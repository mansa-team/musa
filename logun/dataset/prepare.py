import collections
import datetime
import glob
import hashlib
import json
import logging
import random
import re
import statistics
from pathlib import Path

import yaml

try:
    from langdetect import detect, DetectorFactory

    DetectorFactory.seed = 0
except ImportError:
    detect = None
    DetectorFactory = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
dapt = config["dapt"]
paths = config["paths"]

CLEANING_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:FOLHA|PAG(?:INA|É?)?)\s*\d+\s*/\s*\d+\s*(?:\n|$)", re.I),
    re.compile(
        r"(?:^|\n)\s*(?:Página\s+\d+\s+de\s+\d+|Page\s+\d+\s+of\s+\d+)\s*(?:\n|$)", re.I
    ),
    re.compile(r"CNPJ[:\s]*\d[\d.\-\/]{17,}\s*(?:\n|$)", re.I),
    re.compile(
        r"(?:^|\n)\s*(?:[A-ZÁÉÍÓÚÃÕÊÔ][A-ZÁÉÍÓÚÃÕÊÔ\s]{10,})\s*(?:CNPJ|cnpj)\s*[:\-]?\s*[\d.\-\/]+",
        re.I,
    ),
    re.compile(
        r"(?:^|\n)\s*(?:O\s+conteúdo\s+deste\s+documento|Declaração\s+de\s+responsabilidade|Este\s+documento\s+(?:foi|contém)\s+objetos?\s+de\s+(?:catequese|transcrição))\s.*?(?:\n\s*\n|\Z)",
        re.I | re.DOTALL,
    ),
    re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    re.compile(r"\n{3,}"),
]
PORTUGUESE_WORDS = {
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "para", "por", "com", "sem", "sob", "sobre", "que", "como", "mais",
    "ou", "e", "a", "o", "as", "os", "um", "uma", "ser", "estar", "ter",
    "haver", "fazer", "poder", "dizer", "não", "também", "já", "ainda",
    "muito", "todo", "cada", "quando", "onde", "porque", "então", "assim",
    "até", "desde", "durante", "após", "antes",
}
COMMON_PT_TRIGRAMS = {
    "que", "com", "par", "est", "não", "por", "uma", "dos",
    "ent", "ção", "ado", "ara", "ver", "ser", "con",
}


def percentile_sorted(arr: list, p: float) -> int | float:
    if not arr:
        return 0
    sorted_values = sorted(arr)
    k = (len(sorted_values) - 1) * p / 100
    floor_idx = int(k)
    ceil_idx = min(floor_idx + 1, len(sorted_values) - 1)
    if floor_idx == ceil_idx:
        return sorted_values[floor_idx]
    delta = k - floor_idx
    return int(sorted_values[floor_idx] * (1 - delta) + sorted_values[ceil_idx] * delta)


def clean_text(text: str) -> str:
    for pattern in CLEANING_PATTERNS:
        text = pattern.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def deduplicate_documents(documents: list[dict]) -> list[dict]:
    seen = set()
    exact: list[dict] = []
    for doc in documents:
        doc_hash = hashlib.sha256(doc.get("text", "").encode()).hexdigest()
        if doc_hash not in seen:
            seen.add(doc_hash)
            exact.append(doc)
    return exact


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 200) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    cur = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) + 2 > chunk_size:
            chunks.append(cur)
            cur = (
                (cur[-overlap:] + "\n\n" + para)
                if overlap and len(cur) > overlap
                else para
            )
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def is_portuguese(text: str) -> bool:
    if not text or len(text) < 20:
        return False
    thresh = float(dapt.get("language_threshold", 0.05))
    if detect is not None:
        try:
            return detect(text[:1500]) == "pt"
        except Exception:
            pass
    low = text.lower()
    words = re.findall(r"[a-záéíóúâêôãõçà]+", low)
    if not words:
        return False
    sample = words[:300]
    pt_cnt = sum(1 for w in sample if w in PORTUGUESE_WORDS)
    stop_ratio = pt_cnt / len(sample)
    trigrams = {low[i : i + 3] for i in range(len(low) - 2) if low[i : i + 3].isalpha()}
    trig_score = len(trigrams & COMMON_PT_TRIGRAMS) / 15
    return (stop_ratio + trig_score) >= thresh


def passes_quality_filter(
    text: str, min_length: int = 200, max_length: int = 8000
) -> tuple[bool, str]:
    if len(text) < min_length:
        return False, "too_short"
    if len(text) > max_length:
        return False, "too_long"
    if not is_portuguese(text):
        return False, "non_portuguese"
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.7:
        return False, "high_digit_ratio"
    return True, "ok"


def compute_token_stats(texts: list[str], tokenizer_name: str = "answerdotai/ModernBERT-base") -> dict:
    if not texts:
        return {
            "total": 0,
            "mean": 0,
            "median": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "estimated": True,
            "tokenizer": tokenizer_name,
        }
    counts = None
    estimated = True
    if AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(
                tokenizer_name, local_files_only=False, trust_remote_code=False
            )
            counts = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
            estimated = False
        except Exception:
            counts = None
    if counts is None:
        counts = [max(1, len(t) // 4) for t in texts]
        estimated = True
    return {
        "total": int(sum(counts)),
        "mean": round(statistics.mean(counts), 2),
        "median": int(statistics.median(counts)),
        "p50": int(percentile_sorted(counts, 50)),
        "p90": int(percentile_sorted(counts, 90)),
        "p95": int(percentile_sorted(counts, 95)),
        "p99": int(percentile_sorted(counts, 99)),
        "min": int(min(counts)),
        "max": int(max(counts)),
        "estimated": estimated,
        "tokenizer": tokenizer_name,
    }


def discover_input_files() -> list[Path]:
    out_dir = Path(paths.get("output_dir", "./data/output")).resolve()
    inp = paths.get("input")
    data_dir = Path(paths.get("data_dir", "./data")).resolve()
    if inp:
        candidate = Path(inp)
        candidate = (
            (Path(__file__).parent / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
        if candidate.is_file():
            files = [candidate]
        elif candidate.is_dir():
            files = list(candidate.rglob("*.jsonl"))
        elif any(c in inp for c in "*?["):
            files = [Path(x).resolve() for x in glob.glob(inp, recursive=True)]
        else:
            files = list(data_dir.rglob("*.jsonl"))
    else:
        files = list(data_dir.rglob("*.jsonl"))
    return sorted(
        {
            f
            for f in files
            if out_dir not in f.resolve().parents and f.resolve().parent != out_dir
        }
    )


def format_output(chunks: list[dict], output_dir: str | Path, stem: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(c["text"].replace("\n", " ") for c in chunks)
    (out / f"{stem}.txt").write_text(txt, encoding="utf-8")
    jsl = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks)
    (out / f"{stem}.jsonl").write_text(jsl, encoding="utf-8")


def run_pipeline(config_override: dict | None = None) -> None:
    cfg = (
        config_override
        if config_override is not None
        else yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    )
    globals()["config"], globals()["dapt"], globals()["paths"] = (
        cfg,
        cfg["dapt"],
        cfg["paths"],
    )
    dapt_cfg, paths_cfg = cfg["dapt"], cfg["paths"]
    files = discover_input_files()
    if not files:
        logger.info(
            f"No input files found (input={paths_cfg.get('input')} data_dir={paths_cfg.get('data_dir')})"
        )
        return
    logger.info(f"Inputs: {[str(f) for f in files]}")
    documents: list[dict] = []
    for fp in sorted(files):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    rec["src_file"] = str(fp)
                    documents.append(rec)
    documents = sorted(
        documents,
        key=lambda d: (
            str(d.get("source_url", "")),
            str(d.get("filename", "")),
            str(d.get("date", "")),
        ),
    )
    docs_loaded = len(documents)
    total_chars = sum(len(d.get("text", "")) for d in documents)
    logger.info(f"{docs_loaded} documents, {total_chars:,} characters")

    for d in documents:
        d["text"] = clean_text(d.get("text", ""))

    docs_before_dedup = len(documents)
    documents = deduplicate_documents(documents)
    docs_deduped = docs_before_dedup - len(documents)
    logger.info(
        f"After deduplication: {len(documents)} documents (removed {docs_deduped} exact duplicates)"
    )
    docs_failed_empty = sum(1 for d in documents if not d.get("text", "").strip())
    documents = [d for d in documents if d.get("text", "").strip()]

    chunk_size = int(dapt_cfg.get("chunk_size", 1800))
    overlap = int(dapt_cfg.get("overlap", 200))
    min_length = int(dapt_cfg.get("min_length", 200))
    max_length = int(dapt_cfg.get("max_length", 8000))

    all_records: list[dict] = []
    gid = 0
    for doc in documents:
        doc_year = doc.get("year", 0)
        try:
            doc_year = int(doc_year) if doc_year not in (None, "") else 0
        except Exception:
            doc_year = 0
        if not doc_year:
            m = re.search(r"(20\d{2})", str(doc.get("date", "")))
            doc_year = int(m.group(1)) if m else 0
        doc_date = str(doc.get("date", ""))
        text = doc.get("text", "")
        doc_id = doc.get("document_id") or hashlib.sha256(text.encode()).hexdigest()[:12]
        for ch in chunk_text(doc["text"], chunk_size, overlap):
            rec = {
                "text": ch,
                "source": "cvm",
                "company": doc.get("company") or doc.get("Nome_Companhia") or "",
                "cnpj": doc.get("cnpj") or doc.get("CNPJ_Companhia") or "",
                "category": doc.get("category") or doc.get("Categoria") or "",
                "subject": doc.get("subject") or doc.get("Assunto") or "",
                "date": doc_date,
                "year": doc_year,
                "document_id": doc_id,
                "filename": doc.get("filename") or "",
                "source_url": doc.get("source_url") or doc.get("Link_Download") or "",
                "chunk_id": gid,
                "extraction_quality": doc.get("extraction_quality") or {},
            }
            gid += 1
            all_records.append(rec)

    chunks_generated = len(all_records)
    logger.info(f"Total chunks: {chunks_generated:,}")

    seen_hashes = set()
    filtered: list[dict] = []
    filter_reasons = collections.Counter()
    lang_stats = collections.Counter()
    for rec in all_records:
        ch = rec["text"]
        chunk_hash = hashlib.sha256(ch.encode()).hexdigest()
        if chunk_hash in seen_hashes:
            filter_reasons["duplicate_chunk"] += 1
            continue
        ok, reason = passes_quality_filter(ch, min_length, max_length)
        if not ok:
            filter_reasons[reason] += 1
            if reason == "non_portuguese":
                lang_stats["non_pt"] += 1
            continue
        lang_stats["pt"] += 1
        seen_hashes.add(chunk_hash)
        filtered.append(rec)

    filter_reasons["retained"] = len(filtered)
    logger.info(
        f"After filtering: {len(filtered):,} chunks reasons={dict(filter_reasons)}"
    )

    tok_name = dapt_cfg.get("tokenizer", "answerdotai/ModernBERT-base")
    texts = [r["text"] for r in filtered]
    tok_stats = compute_token_stats(texts, tok_name)
    logger.info(
        f"Tokens: {tok_stats['total']:,} estimated={tok_stats['estimated']} mean={tok_stats['mean']} p50={tok_stats['p50']} p90={tok_stats['p90']}"
    )

    by_year = collections.Counter(r["year"] for r in filtered if r["year"])
    buckets = collections.Counter()
    for r in filtered:
        y = r["year"]
        if not y:
            buckets["unknown"] += 1
        elif 2003 <= y <= 2010:
            buckets["2003-2010"] += 1
        elif 2011 <= y <= 2020:
            buckets["2011-2020"] += 1
        else:
            buckets["2021-2025+"] += 1
    by_category = collections.Counter(r["category"] or "unknown" for r in filtered)
    by_company = collections.Counter(
        r["company"] or r["cnpj"] or "unknown" for r in filtered
    )
    by_qual = collections.Counter(
        (r.get("extraction_quality") or {}).get("quality_flag", "unknown")
        for r in filtered
    )
    doc_lens = [len(d["text"]) for d in documents]
    len_bins = collections.Counter()
    for doc_len in doc_lens:
        if doc_len < 500:
            len_bins["<500"] += 1
        elif doc_len < 1500:
            len_bins["500-1500"] += 1
        elif doc_len < 3000:
            len_bins["1500-3000"] += 1
        elif doc_len < 8000:
            len_bins["3000-8000"] += 1
        else:
            len_bins["8000+"] += 1

    out_dir = Path(paths_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(files) == 1:
        stem = Path(files[0]).stem
    else:
        stem = Path(paths_cfg.get("corpus", "./data/corpus.txt")).stem
        if stem == "corpus":
            stem = "cvm_dapt"
    format_output(filtered, out_dir, stem)
    logger.info(f"Output written to: {out_dir}/{stem}.{{jsonl,txt}}")

    train_split = float(dapt_cfg.get("train_split", 0.95))
    seed = int(dapt_cfg.get("seed", 42))
    if 0 < train_split < 1 and filtered:
        rnd = random.Random(seed)
        shuffled = filtered[:]
        rnd.shuffle(shuffled)
        n_train = int(len(shuffled) * train_split)
        train = shuffled[:n_train]
        val = shuffled[n_train:]
        format_output(train, out_dir, "train")
        format_output(val, out_dir, "validation")
        logger.info(
            f"Split {train_split:.0%} train={len(train)} val={len(val)} seed={seed}"
        )

    manifest = {
        "dataset": "cvm_ipe",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "source_years": sorted(by_year.keys()),
        "documents_downloaded": docs_loaded,
        "documents_extracted": len(documents) + docs_failed_empty,
        "documents_failed": docs_failed_empty,
        "documents_deduplicated": docs_deduped,
        "chunks_generated": chunks_generated,
        "chunks_retained": len(filtered),
        "chunks_filtered": chunks_generated - len(filtered),
        "filter_reasons": dict(filter_reasons),
        "language_stats": dict(lang_stats),
        "language_detector": "langdetect" if detect is not None else "heuristic",
        "tokens": tok_stats,
        "tokens_per_doc": round(tok_stats["total"] / max(len(documents), 1), 2)
        if documents
        else 0,
        "tokens_per_chunk": round(tok_stats["total"] / max(len(filtered), 1), 2)
        if filtered
        else 0,
        "temporal": {"by_year": dict(by_year), "buckets": dict(buckets)},
        "distributions": {
            "by_year": dict(by_year),
            "by_category": dict(by_category),
            "by_company_top20": dict(by_company.most_common(20)),
            "by_doc_length_bins": dict(len_bins),
            "by_language_filter": dict(filter_reasons),
            "by_extraction_quality": dict(by_qual),
        },
        "chunk_chars": {
            "mean": round(statistics.mean([len(r["text"]) for r in filtered]), 2)
            if filtered
            else 0,
            "median": int(statistics.median([len(r["text"]) for r in filtered]))
            if filtered
            else 0,
            "p50": percentile_sorted([len(r["text"]) for r in filtered], 50),
            "p90": percentile_sorted([len(r["text"]) for r in filtered], 90),
            "p95": percentile_sorted([len(r["text"]) for r in filtered], 95),
            "p99": percentile_sorted([len(r["text"]) for r in filtered], 99),
        },
        "dapt_config": {k: dapt_cfg[k] for k in sorted(dapt_cfg.keys())},
        "paths": {k: str(paths_cfg[k]) for k in sorted(paths_cfg.keys())},
        "deterministic": True,
        "seed": seed,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = {
        "dataset": "cvm_ipe",
        "chunks": len(filtered),
        "tokens": tok_stats["total"],
        "tokenizer": tok_name,
        "split": train_split,
        "seed": seed,
        "generated_at": manifest["generated_at"],
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        f"Manifest: {manifest_path} tokens={tok_stats['total']:,} estimated={tok_stats['estimated']}"
    )


class DaptPreparer:
    percentile_sorted = staticmethod(percentile_sorted)
    clean_text = staticmethod(clean_text)
    deduplicate_documents = staticmethod(deduplicate_documents)
    chunk_text = staticmethod(chunk_text)
    is_portuguese = staticmethod(is_portuguese)
    passes_quality_filter = staticmethod(passes_quality_filter)
    compute_token_stats = staticmethod(compute_token_stats)
    discover_input_files = staticmethod(discover_input_files)
    format_output = staticmethod(format_output)
    run_pipeline = staticmethod(run_pipeline)


if __name__ == "__main__":
    run_pipeline()
