import collections
import datetime
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import yaml


logger = logging.getLogger(__name__)


def setup_logging(config):
    logging_config = config.get("logging", {})
    level = getattr(logging, str(logging_config.get("level", "INFO")).upper(), logging.INFO)
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

CLEANING_PATTERNS = [re.compile(r"(?:^|\n)\s*(?:FOLHA|PAG(?:INA|É?)?)\s*\d+\s*/\s*\d+\s*(?:\n|$)", re.I), re.compile(r"(?:^|\n)\s*(?:Página\s+\d+\s+de\s+\d+|Page\s+\d+\s+of\s+\d+)\s*(?:\n|$)", re.I), re.compile(r"CNPJ[:\s]*\d[\d.\-\/]{17,}\s*(?:\n|$)", re.I), re.compile(r"(?:^|\n)\s*(?:[A-ZÁÉÍÓÚÃÕÊÔ][A-ZÁÉÍÓÚÃÕÊÔ\s]{10,})\s*(?:CNPJ|cnpj)\s*[:\-]?\s*[\d.\-\/]+", re.I), re.compile(r"(?:^|\n)\s*(?:O\s+conteúdo\s+deste\s+documento|Declaração\s+de\s+responsabilidade|Este\s+documento\s+(?:foi|contém)\s+objetos?\s+de\s+(?:catequese|transcrição))\s.*?(?:\n\s*\n|\Z)", re.I | re.DOTALL), re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")]


def clean_text(text):
    for pattern in CLEANING_PATTERNS:
        text = pattern.sub("", text)

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text, chunk_size=1800, overlap=200):
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    current_chunk = ""

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()

        if not para:
            continue

        if len(para) > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

                current_chunk = ""

            for i in range(0, len(para), chunk_size - overlap):
                chunks.append(para[i:i+chunk_size])

            continue

        if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
            chunks.append(current_chunk)

            current_chunk = (current_chunk[-overlap:] + "\n\n" + para) if overlap and len(current_chunk) > overlap else para
        else:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def passes_quality_filter(text, min_length=400):
    if len(text) < min_length:
        return False, "too_short"

    return True, "ok"


def compute_token_stats(texts, tokenizer_name="answerdotai/ModernBERT-base"):
    if not texts:
        return {"total": 0, "mean": 0, "median": 0, "p50": 0, "p90": 0, "p95": 0, "estimated": True, "tokenizer": tokenizer_name}

    counts = [max(1, len(text) // 4) for text in texts]
    counts_array = np.array(counts)

    return {
        "total": int(counts_array.sum()), "mean": round(float(counts_array.mean()), 2),
        "median": int(np.median(counts_array)), "p50": int(np.percentile(counts_array, 50)),
        "p90": int(np.percentile(counts_array, 90)), "p95": int(np.percentile(counts_array, 95)),
        "min": int(counts_array.min()), "max": int(counts_array.max()),
        "estimated": True, "tokenizer": tokenizer_name,
    }


def discover_input_files():
    raw_data_dir = Path(paths.get("data_dir", "./data"))

    data_dir = (Path(__file__).parent / raw_data_dir).resolve() if not raw_data_dir.is_absolute() else raw_data_dir.resolve()

    input_setting = paths.get("input")

    if input_setting:
        candidate = Path(input_setting)

        candidate = (Path(__file__).parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

        if candidate.is_file():
            return [candidate]

        if candidate.is_dir():
            return sorted(candidate.rglob("*.jsonl"))

    files = sorted(data_dir.rglob("*.jsonl"))

    raw_output = Path(paths.get("output_dir", "./data/output"))
    output_resolved = (Path(__file__).parent / raw_output).resolve() if not raw_output.is_absolute() else raw_output.resolve()

    raw_corpus = Path(paths.get("corpus", "./data/output/corpus.txt"))
    corpus_path = (Path(__file__).parent / raw_corpus).resolve() if not raw_corpus.is_absolute() else raw_corpus.resolve()

    try:
        files = [file_path for file_path in files if output_resolved not in file_path.parents and file_path.parent != output_resolved and corpus_path.parent not in file_path.parents and file_path != corpus_path and file_path != corpus_path.with_suffix(".jsonl")]
    except Exception:
        pass

    return files


def format_output(chunks, output_dir, stem):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

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

    chunk_size = int(dapt_config.get("chunk_size", 1800))
    overlap = int(dapt_config.get("overlap", 200))
    min_length = int(dapt_config.get("min_length", 200))
    tokenizer_name = dapt_config.get("tokenizer", "answerdotai/ModernBERT-base")

    raw_output = Path(paths_config["output_dir"])
    out_dir = (Path(__file__).parent / raw_output).resolve() if not raw_output.is_absolute() else raw_output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = "corpus"
    out_path = out_dir / f"{stem}.jsonl"

    seen_hashes = set()
    filter_reasons = collections.Counter()
    by_category = collections.Counter()
    token_counts = []
    total_chars = 0
    docs_loaded = 0
    chunks_generated = 0
    chunks_retained = 0
    chunk_id_counter = 0

    # ponytail: stream — one doc at a time, one chunk write at a time, no batch lists
    with open(out_path, "w", encoding="utf-8") as out_handle:
        for file_path in sorted(files):
            with open(file_path, encoding="utf-8") as file_handle:
                for line in file_handle:
                    if not line.strip():
                        continue

                    try:
                        document = json.loads(line)
                    except Exception:
                        continue

                    docs_loaded += 1

                    cleaned = clean_text(document.get("text", ""))

                    if not cleaned.strip():
                        continue

                    document_year = int(document.get("year") or 0)
                    document_date = str(document.get("date", ""))
                    document_id = document.get("document_id") or hashlib.sha256(cleaned.encode()).hexdigest()[:12]

                    for chunk in chunk_text(cleaned, chunk_size, overlap):
                        current_id = chunk_id_counter
                        chunk_id_counter += 1
                        chunks_generated += 1

                        chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()

                        if chunk_hash in seen_hashes:
                            filter_reasons["duplicate_chunk"] += 1
                            continue

                        passes, reason = passes_quality_filter(chunk, min_length)

                        if not passes:
                            filter_reasons[reason] += 1
                            continue

                        seen_hashes.add(chunk_hash)

                        record = {"text": chunk, "source": "cvm", "company": document.get("company") or "", "cnpj": document.get("cnpj") or "", "category": document.get("category") or "", "subject": document.get("subject") or "", "date": document_date, "year": document_year, "document_id": document_id, "filename": document.get("filename") or "", "source_url": document.get("source_url") or "", "chunk_id": current_id, "extraction_quality": document.get("extraction_quality") or {}}

                        by_category[record["category"] or "unknown"] += 1
                        count = max(1, len(chunk) // 4)
                        token_counts.append(count)
                        total_chars += len(chunk)
                        chunks_retained += 1

                        out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    filter_reasons["retained"] = chunks_retained

    if not token_counts:
        token_stats = {"total": 0, "mean": 0, "median": 0, "p50": 0, "p90": 0, "p95": 0, "estimated": True, "tokenizer": tokenizer_name}
    else:
        counts_array = np.array(token_counts)
        token_stats = {
            "total": int(counts_array.sum()), "mean": round(float(counts_array.mean()), 2),
            "median": int(np.median(counts_array)), "p50": int(np.percentile(counts_array, 50)),
            "p90": int(np.percentile(counts_array, 90)), "p95": int(np.percentile(counts_array, 95)),
            "min": int(counts_array.min()), "max": int(counts_array.max()),
            "estimated": True, "tokenizer": tokenizer_name,
        }

    seed = int(dapt_config.get("seed", 42))

    normalized_dapt = {key: dapt_config[key] for key in sorted(dapt_config.keys()) if key != "max_length"}
    normalized_dapt["max_length"] = chunk_size

    manifest = {"dataset": "cvm_ipe", "generated_at": datetime.datetime.utcnow().isoformat() + "Z", "documents_loaded": docs_loaded, "documents_deduplicated": 0, "chunks_generated": chunks_generated, "chunks_retained": chunks_retained, "filter_reasons": dict(filter_reasons), "by_category": dict(by_category), "tokens": token_stats, "dapt_config": normalized_dapt, "seed": seed}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_run = {"documents_retained": chunks_retained, "chunks": chunks_retained, "tokens": token_stats["total"], "chars": total_chars, "by_category": dict(by_category)}
    (out_dir / "audit.json").write_text(json.dumps(audit_run, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Done: {chunks_retained} chunks, tokens={token_stats['total']}")


if __name__ == "__main__":
    setup_logging(config)
    run_pipeline()
