import collections
import datetime
import hashlib
import json
import logging
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except ImportError:
    detect = None
    DetectorFactory = None
logger = logging.getLogger(__name__)


def setup_logging(config):
    logging_config = config.get("logging", {})
    level_name = logging_config.get("level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    log_file = logging_config.get("file")

    if log_file:
        log_path = Path(log_file)

        if not log_path.is_absolute():
            log_path = Path(__file__).parent / log_path

        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=level, filename=str(log_path))
    else:
        logging.basicConfig(level=level)


CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
dapt = config["dapt"]
paths = config["paths"]

CLEANING_PATTERNS = [re.compile(r"(?:^|\n)\s*(?:FOLHA|PAG(?:INA|É?)?)\s*\d+\s*/\s*\d+\s*(?:\n|$)", re.I), re.compile(r"(?:^|\n)\s*(?:Página\s+\d+\s+de\s+\d+|Page\s+\d+\s+of\s+\d+)\s*(?:\n|$)", re.I), re.compile(r"CNPJ[:\s]*\d[\d.\-\/]{17,}\s*(?:\n|$)", re.I), re.compile(r"(?:^|\n)\s*(?:[A-ZÁÉÍÓÚÃÕÊÔ][A-ZÁÉÍÓÚÃÕÊÔ\s]{10,})\s*(?:CNPJ|cnpj)\s*[:\-]?\s*[\d.\-\/]+", re.I), re.compile(r"(?:^|\n)\s*(?:O\s+conteúdo\s+deste\s+documento|Declaração\s+de\s+responsabilidade|Este\s+documento\s+(?:foi|contém)\s+objetos?\s+de\s+(?:catequese|transcrição))\s.*?(?:\n\s*\n|\Z)", re.I | re.DOTALL), re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"), re.compile(r"\n{3,}")]
PORTUGUESE_WORDS = {
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "para", "por", "com", "sem", 
    "sob", "sobre", "que", "como", "mais", "ou", "e", "a", "o", "as", "os", "um", "uma", "ser", "estar", 
    "ter", "haver", "fazer", "poder", "dizer", "não", "também", "já", "ainda", "muito", "todo", "cada", 
    "quando", "onde", "porque", "então", "assim", "até", "desde", "durante", "após", "antes"
    }
COMMON_PT_TRIGRAMS = {
    "que", "com", "par", "est", "não", "por", "uma", "dos", "ent", "ção", "ado", "ara", "ver", "ser", "con"
    }


def clean_text(text):
    for pattern in CLEANING_PATTERNS:
        text = pattern.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def deduplicate_documents(documents):
    seen = set()
    exact = []
    for doc in documents:
        doc_hash = hashlib.sha256(doc.get("text", "").encode()).hexdigest()
        if doc_hash not in seen:
            seen.add(doc_hash)
            exact.append(doc)
    return exact


def chunk_text(text, chunk_size=1800, overlap=200):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    cur = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) + 2 > chunk_size:
            chunks.append(cur)
            cur = (cur[-overlap:] + "\n\n" + para) if overlap and len(cur) > overlap else para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def is_portuguese(text):
    if not text or len(text) < 20:
        return False
    threshold = float(dapt.get("language_threshold", 0.05))
    if detect is not None:
        try:
            return detect(text[:1500]) == "pt"
        except Exception:
            pass
    lowered = text.lower()
    words = re.findall(r"[a-záéíóúâêôãõçà]+", lowered)
    if not words:
        return False
    sample = words[:300]
    portuguese_count = sum(1 for word in sample if word in PORTUGUESE_WORDS)
    stop_ratio = portuguese_count / len(sample)
    trigrams = {lowered[i:i+3] for i in range(len(lowered) - 2) if lowered[i:i+3].isalpha()}
    trigram_score = len(trigrams & COMMON_PT_TRIGRAMS) / 15
    return (stop_ratio + trigram_score) >= threshold


def passes_quality_filter(text, min_length=200, max_length=8000):
    if len(text) < min_length:
        return False, "too_short"
    if len(text) > max_length:
        return False, "too_long"
    if not is_portuguese(text):
        return False, "non_portuguese"
    digit_ratio = sum(character.isdigit() for character in text) / max(len(text), 1)
    if digit_ratio > 0.7:
        return False, "high_digit_ratio"
    return True, "ok"


def compute_token_stats(texts, tokenizer_name="answerdotai/ModernBERT-base"):
    if not texts:
        return {"total": 0, "mean": 0, "median": 0, "p50": 0, "p90": 0, "p95": 0, "estimated": True, "tokenizer": tokenizer_name}
    counts = [max(1, len(text) // 4) for text in texts]
    counts_array = np.array(counts)
    return {
        "total": int(counts_array.sum()),
        "mean": round(float(counts_array.mean()), 2),
        "median": int(np.median(counts_array)),
        "p50": int(np.percentile(counts_array, 50)),
        "p90": int(np.percentile(counts_array, 90)),
        "p95": int(np.percentile(counts_array, 95)),
        "min": int(counts_array.min()),
        "max": int(counts_array.max()),
        "estimated": True,
        "tokenizer": tokenizer_name,
    }


def discover_input_files():
    raw_data_dir = Path(paths.get("data_dir", "./data"))
    data_dir = (Path(__file__).parent / raw_data_dir).resolve() if not raw_data_dir.is_absolute() else raw_data_dir.resolve()

    # ponytail: rglob for nested scraper year dirs; sorted for deterministic single-pass
    input_setting = paths.get("input")
    if input_setting:
        candidate = Path(input_setting)
        candidate = (Path(__file__).parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if candidate.is_file():
            return [candidate]
        if candidate.is_dir():
            return sorted(candidate.rglob("*.jsonl"))

    files = sorted(data_dir.rglob("*.jsonl"))

    # exclude output_dir and corpus to avoid re-ingesting prepared shards
    raw_output = Path(paths.get("output_dir", "./data/output"))
    output_resolved = (Path(__file__).parent / raw_output).resolve() if not raw_output.is_absolute() else raw_output.resolve()
    raw_corpus = Path(paths.get("corpus", "./data/corpus/corpus.txt"))
    corpus_path = (Path(__file__).parent / raw_corpus).resolve() if not raw_corpus.is_absolute() else raw_corpus.resolve()
    try:
        files = [
            file_path
            for file_path in files
            if output_resolved not in file_path.parents and file_path.parent != output_resolved and corpus_path.parent not in file_path.parents and file_path != corpus_path and file_path != corpus_path.with_suffix(".jsonl")
        ]
    except Exception:
        pass
    return files


def format_output(chunks, output_dir, stem):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plain_text = "\n".join(chunk["text"].replace("\n", " ") for chunk in chunks)
    (out_path / f"{stem}.txt").write_text(plain_text, encoding="utf-8")
    jsonl_text = "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks)
    (out_path / f"{stem}.jsonl").write_text(jsonl_text, encoding="utf-8")


def run_pipeline(config_override=None):
    pipeline_config = config_override if config_override is not None else yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    dapt_config, paths_config = pipeline_config["dapt"], pipeline_config["paths"]
    files = discover_input_files()
    if not files:
        logger.info(f"No input files found (data_dir={paths_config.get('data_dir')})")
        return
    logger.info(f"Inputs: {[str(file_path) for file_path in files]}")

    documents = []
    for file_path in sorted(files):
        with open(file_path, encoding="utf-8") as file_handle:
            for line in file_handle:
                if line.strip():
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    record["src_file"] = str(file_path)
                    documents.append(record)
    docs_loaded = len(documents)

    for document in documents:
        document["text"] = clean_text(document.get("text", ""))

    docs_before_dedup = len(documents)
    documents = deduplicate_documents(documents)
    docs_deduped = docs_before_dedup - len(documents)
    documents = [document for document in documents if document.get("text", "").strip()]

    chunk_size = int(dapt_config.get("chunk_size", 1800))
    overlap = int(dapt_config.get("overlap", 200))
    min_length = int(dapt_config.get("min_length", 200))
    max_length = int(dapt_config.get("max_length", 8000))

    all_records = []
    chunk_id_counter = 0
    for document in documents:
        document_year = document.get("year", 0)
        try:
            document_year = int(document_year) if document_year not in (None, "") else 0
        except Exception:
            document_year = 0
        if not document_year:
            year_match = re.search(r"(20\d{2})", str(document.get("date", "")))
            document_year = int(year_match.group(1)) if year_match else 0
        document_date = str(document.get("date", ""))
        document_id = document.get("document_id") or hashlib.sha256(document.get("text", "").encode()).hexdigest()[:12]
        for chunk in chunk_text(document["text"], chunk_size, overlap):
            record = {
                "text": chunk,
                "source": "cvm",
                "company": document.get("company") or "",
                "cnpj": document.get("cnpj") or "",
                "category": document.get("category") or "",
                "subject": document.get("subject") or "",
                "date": document_date,
                "year": document_year,
                "document_id": document_id,
                "filename": document.get("filename") or "",
                "source_url": document.get("source_url") or "",
                "chunk_id": chunk_id_counter,
                "extraction_quality": document.get("extraction_quality") or {},
            }
            chunk_id_counter += 1
            all_records.append(record)

    chunks_generated = len(all_records)

    seen_hashes = set()
    filtered = []
    filter_reasons = collections.Counter()
    for record in all_records:
        chunk_text_value = record["text"]
        chunk_hash = hashlib.sha256(chunk_text_value.encode()).hexdigest()
        if chunk_hash in seen_hashes:
            filter_reasons["duplicate_chunk"] += 1
            continue
        passes, reason = passes_quality_filter(chunk_text_value, min_length, max_length)
        if not passes:
            filter_reasons[reason] += 1
            continue
        seen_hashes.add(chunk_hash)
        filtered.append(record)
    filter_reasons["retained"] = len(filtered)

    tokenizer_name = dapt_config.get("tokenizer", "answerdotai/ModernBERT-base")
    token_stats = compute_token_stats([record["text"] for record in filtered], tokenizer_name)
    by_category = collections.Counter(record["category"] or "unknown" for record in filtered)

    raw_output = Path(paths_config["output_dir"])
    out_dir = (Path(__file__).parent / raw_output).resolve() if not raw_output.is_absolute() else raw_output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ponytail: no train/validation split — user splits with sklearn downstream
    stem = "corpus"
    format_output(filtered, out_dir, stem)

    seed = int(dapt_config.get("seed", 42))
    manifest = {
        "dataset": "cvm_ipe",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "documents_loaded": docs_loaded,
        "documents_deduplicated": docs_deduped,
        "chunks_generated": chunks_generated,
        "chunks_retained": len(filtered),
        "filter_reasons": dict(filter_reasons),
        "by_category": dict(by_category),
        "tokens": token_stats,
        "dapt_config": {key: dapt_config[key] for key in sorted(dapt_config.keys())},
        "seed": seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_run = {
        "documents_retained": len(filtered),
        "chunks": len(filtered),
        "tokens": token_stats["total"],
        "chars": sum(len(record["text"]) for record in filtered),
        "by_category": dict(by_category),
    }
    (out_dir / "audit.json").write_text(json.dumps(audit_run, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Done: {len(filtered)} chunks, tokens={token_stats['total']}")


if __name__ == "__main__":
    setup_logging(config)
    run_pipeline()
