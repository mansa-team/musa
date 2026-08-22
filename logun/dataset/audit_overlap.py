import collections
import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text, k=3):
    tokens = re.findall(r"\w+", text)
    if not tokens:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_corpus_metadata(obj: dict, fallback_source: str) -> dict:
    return {"source": obj.get("source") or fallback_source, "company": obj.get("company") or obj.get("Nome_Companhia") or "", "cnpj": obj.get("cnpj") or obj.get("CNPJ_Companhia") or "", "category": obj.get("category") or obj.get("Categoria") or "", "subject": obj.get("subject") or obj.get("Assunto") or "", "date": obj.get("date") or "", "year": obj.get("year") or 0, "document_id": obj.get("document_id") or "", "chunk_id": obj.get("chunk_id") if obj.get("chunk_id") is not None else "", "filename": obj.get("filename") or "", "source_url": obj.get("source_url") or obj.get("Link_Download") or "", "extraction_quality": obj.get("extraction_quality") or {}}


def load_records(path_str: str):
    p = Path(path_str)
    files = []

    if p.is_dir():
        files = sorted(p.rglob("*.jsonl")) or sorted(p.rglob("*.txt"))
    elif p.is_file():
        files = [p]
    elif "*" in path_str:
        files = sorted(Path().glob(path_str))

    recs = []
    for file in files:
        if not file.is_file():
            continue
        sfx = file.suffix.lower()
        try:
            if sfx == ".jsonl":
                with open(file, encoding="utf-8", errors="ignore") as fh:
                    for idx, line in enumerate(fh):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            text = obj.get("text") or obj.get("content") or obj.get("document") or ""
                            if text:
                                meta = build_corpus_metadata(obj, str(file))
                                if not meta["filename"]:
                                    meta["filename"] = file.name
                                recs.append({"id": f"{file.name}:{idx}", "text": str(text), "source": str(file), "corpus_metadata": meta})
                        except json.JSONDecodeError:
                            if line:
                                recs.append({"id": f"{file.name}:{idx}", "text": line, "source": str(file), "corpus_metadata": build_corpus_metadata({}, str(file)) | {"filename": file.name}})
            elif sfx == ".txt":
                text = file.read_text(encoding="utf-8", errors="ignore")
                chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip() and len(c.strip()) >= 50]
                if chunks:
                    for idx, ch in enumerate(chunks):
                        recs.append({"id": f"{file.name}:{idx}", "text": ch, "source": str(file), "corpus_metadata": build_corpus_metadata({}, str(file)) | {"filename": file.name, "chunk_id": idx}})
                elif text.strip() and len(text.strip()) >= 20:
                    recs.append({"id": file.name, "text": text.strip(), "source": str(file), "corpus_metadata": build_corpus_metadata({}, str(file)) | {"filename": file.name, "chunk_id": 0}})
        except Exception:
            continue

    return recs


def build_inverted_index(corpus_shingles_list):
    idx = collections.defaultdict(list)
    for i, sh in enumerate(corpus_shingles_list):
        for s in sh:
            idx[s].append(i)
    return idx


def query_candidates_inverted(bench_shingles, inverted_index):
    candidates = collections.Counter()
    for s in bench_shingles:
        for candidates_index in inverted_index.get(s, []):
            candidates[candidates_index] += 1
    return list(candidates.keys())


def audit_corpus(corpus_records, bench_records, threshold=0.85, k=3):
    exact_set = {sha256_text(r["text"]) for r in corpus_records}
    norm_map = {}
    corpus_shingles_list = []

    for rec in corpus_records:
        n = normalize_text(rec["text"])
        h = sha256_text(n)
        if h not in norm_map:
            norm_map[h] = rec
        corpus_shingles_list.append(shingles(n, k))

    inv = build_inverted_index(corpus_shingles_list)
    total = len(bench_records)
    exact = 0
    normalized = 0
    near = 0
    details = []
    affected = set()

    for idx, bench in enumerate(bench_records):
        bench_text = bench["text"]
        bench_id = bench["id"]
        raw = sha256_text(bench_text)
        if raw in exact_set:
            exact += 1
            m = next((r for r in corpus_records if sha256_text(r["text"]) == raw), corpus_records[0] if corpus_records else None)
            details.append({"benchmark_index": idx, "benchmark_id": bench_id, "match_type": "exact", "jaccard": 1.0, "benchmark_preview": bench_text[:160], "corpus_preview": (m["text"][:160] if m else bench_text[:160]), "corpus_id": m["id"] if m else None, "corpus_metadata": m["corpus_metadata"] if m else {}})
            affected.add(bench_id)
            continue

        n = normalize_text(bench_text)
        h = sha256_text(n)
        if h in norm_map:
            normalized += 1
            m = norm_map[h]
            details.append({"benchmark_index": idx, "benchmark_id": bench_id, "match_type": "normalized", "jaccard": 1.0, "benchmark_preview": bench_text[:160], "corpus_preview": m["text"][:160], "corpus_id": m["id"], "corpus_metadata": m["corpus_metadata"]})
            affected.add(bench_id)
            continue

        bsh = shingles(n, k)
        best = 0.0
        best_idx = None
        for ci in query_candidates_inverted(bsh, inv):
            sc = jaccard(bsh, corpus_shingles_list[ci])
            if sc > best:
                best = sc
                best_idx = ci
            if best == 1.0:
                break

        if best >= threshold:
            near += 1
            m = corpus_records[best_idx] if best_idx is not None else None
            details.append({"benchmark_index": idx, "benchmark_id": bench_id, "match_type": "near_duplicate", "jaccard": round(best, 4), "benchmark_preview": bench_text[:160], "corpus_preview": (m["text"][:160] if m else ""), "corpus_id": m["id"] if m else None, "corpus_metadata": m["corpus_metadata"] if m else {}})
            affected.add(bench_id)
        else:
            details.append({"benchmark_index": idx, "benchmark_id": bench_id, "match_type": "none", "jaccard": round(best, 4), "benchmark_preview": bench_text[:160]})

    overlap_percent = round((exact + normalized + near) / total * 100, 2) if total else 0.0

    return {"total": total, "exact": exact, "normalized": normalized, "near": {"count": near, "jaccard_threshold": threshold, "shingle": k}, "threshold": threshold, "shingle_k": k, "corpus_size": len(corpus_records), "affected_benchmark_ids": sorted(affected), "affected_count": len(affected), "overlap_percent": overlap_percent, "details": details}


audit = audit_corpus


def main():
    config_path = Path(__file__).parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit_config = config.get("audit", {})
    corpus = audit_config.get("corpus", "./data/output")
    benchmark = audit_config.get("benchmark")
    threshold = audit_config.get("threshold", 0.85)
    shingle = audit_config.get("shingle", 3)
    output = audit_config.get("output", "audit_report.json")

    corpus_records = load_records(corpus)
    if not corpus_records:
        logger.error(f"[audit] no corpus records found for corpus \"{corpus}\"")
        sys.exit(1)

    if not benchmark:
        logger.error("[audit] benchmark not set in config.yaml -> audit.benchmark")
        sys.exit(1)

    benchmark_records = load_records(benchmark)
    if not benchmark_records:
        logger.error(f"[audit] benchmark not found or empty: \"{benchmark}\"")
        sys.exit(1)

    report = audit_corpus(corpus_records, benchmark_records, threshold=threshold, k=shingle)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    overlap_percent = report["overlap_percent"]
    logger.info(f"[audit] total={report['total']} exact={report['exact']} normalized={report['normalized']} near={report['near']['count']} overlap={overlap_percent}%")
    logger.info(f"[audit] report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
