import collections
import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import yaml

try:
    from datasketch import MinHash, MinHashLSH

    has_datasketch = True
except ImportError:
    MinHash = None
    MinHashLSH = None
    has_datasketch = False

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    return {
        "source": obj.get("source") or fallback_source,
        "company": obj.get("company") or obj.get("Nome_Companhia") or "",
        "cnpj": obj.get("cnpj") or obj.get("CNPJ_Companhia") or "",
        "category": obj.get("category") or obj.get("Categoria") or "",
        "subject": obj.get("subject") or obj.get("Assunto") or "",
        "date": obj.get("date") or "",
        "year": obj.get("year") or 0,
        "document_id": obj.get("document_id") or "",
        "chunk_id": obj.get("chunk_id") if obj.get("chunk_id") is not None else "",
        "filename": obj.get("filename") or "",
        "source_url": obj.get("source_url") or obj.get("Link_Download") or "",
        "extraction_quality": obj.get("extraction_quality") or {},
    }


def corpus_metadata_from_record(rec: dict) -> dict:
    raw = rec.get("corpus_metadata") or {}
    # ensure all keys present
    return {
        "source": raw.get("source") or rec.get("source") or "",
        "company": raw.get("company") or "",
        "cnpj": raw.get("cnpj") or "",
        "category": raw.get("category") or "",
        "subject": raw.get("subject") or "",
        "date": raw.get("date") or "",
        "year": raw.get("year") or 0,
        "document_id": raw.get("document_id") or "",
        "chunk_id": raw.get("chunk_id") if raw.get("chunk_id") is not None else "",
        "filename": raw.get("filename") or Path(rec.get("source", "")).name,
        "source_url": raw.get("source_url") or "",
        "extraction_quality": raw.get("extraction_quality") or {},
    }


def minhash_for_shingles(shingle_set: set, num_perm: int = 128):
    minhash = MinHash(num_perm=num_perm)
    for shingle in shingle_set:
        minhash.update(shingle.encode("utf-8"))
    return minhash


def load_records(path_str: str):
    path_obj = Path(path_str)
    files = []
    if path_obj.is_dir():
        files = sorted(path_obj.rglob("*.jsonl"))
        # also consider txt files in dir for fallback
        if not files:
            files = sorted(path_obj.rglob("*.txt"))
    elif path_obj.is_file():
        files = [path_obj]
    elif "*" in path_str:
        files = sorted(Path().glob(path_str))
    records = []
    for file in files:
        if not file.is_file():
            continue
        suffix = file.suffix.lower()
        try:
            if suffix == ".jsonl":
                with open(file, encoding="utf-8", errors="ignore") as fh:
                    for idx, line in enumerate(fh):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            text = (
                                obj.get("text")
                                or obj.get("content")
                                or obj.get("document")
                                or obj.get("input")
                                or obj.get("prompt")
                                or ""
                            )
                            if not text and isinstance(obj, str):
                                text = obj
                            if text:
                                metadata = build_corpus_metadata(obj, str(file))
                                # ensure filename fallback
                                if not metadata["filename"]:
                                    metadata["filename"] = file.name
                                records.append(
                                    {
                                        "id": f"{file.name}:{idx}",
                                        "text": str(text),
                                        "source": str(file),
                                        "corpus_metadata": metadata,
                                    }
                                )
                        except json.JSONDecodeError:
                            if line:
                                records.append(
                                    {
                                        "id": f"{file.name}:{idx}",
                                        "text": line,
                                        "source": str(file),
                                        "corpus_metadata": {
                                            "source": str(file),
                                            "company": "",
                                            "cnpj": "",
                                            "category": "",
                                            "subject": "",
                                            "date": "",
                                            "year": 0,
                                            "document_id": "",
                                            "chunk_id": "",
                                            "filename": file.name,
                                            "source_url": "",
                                            "extraction_quality": {},
                                        },
                                    }
                                )
            elif suffix == ".txt":
                text = file.read_text(encoding="utf-8", errors="ignore")
                chunks = [
                    c.strip()
                    for c in re.split(r"\n\s*\n", text)
                    if c.strip() and len(c.strip()) >= 50
                ]
                if chunks:
                    for idx, chunk in enumerate(chunks):
                        records.append(
                            {
                                "id": f"{file.name}:{idx}",
                                "text": chunk,
                                "source": str(file),
                                "corpus_metadata": {
                                    "source": str(file),
                                    "company": "",
                                    "cnpj": "",
                                    "category": "",
                                    "subject": "",
                                    "date": "",
                                    "year": 0,
                                    "document_id": "",
                                    "chunk_id": idx,
                                    "filename": file.name,
                                    "source_url": "",
                                    "extraction_quality": {},
                                },
                            }
                        )
                elif text.strip() and len(text.strip()) >= 20:
                    records.append(
                        {
                            "id": file.name,
                            "text": text.strip(),
                            "source": str(file),
                            "corpus_metadata": {
                                "source": str(file),
                                "company": "",
                                "cnpj": "",
                                "category": "",
                                "subject": "",
                                "date": "",
                                "year": 0,
                                "document_id": "",
                                "chunk_id": 0,
                                "filename": file.name,
                                "source_url": "",
                                "extraction_quality": {},
                            },
                        }
                    )
        except Exception:
            continue
    return records


def build_inverted_index(corpus_shingles_list):
    index = collections.defaultdict(list)
    for corpus_idx, shingle_set in enumerate(corpus_shingles_list):
        for shingle in shingle_set:
            index[shingle].append(corpus_idx)
    return index


def query_candidates_inverted(bench_shingles, inverted_index, corpus_shingles_list, corpus_sizes, bench_size, threshold):
    counter = collections.Counter()
    for shingle in bench_shingles:
        for corpus_idx in inverted_index.get(shingle, []):
            counter[corpus_idx] += 1
    if not counter:
        return []
    candidates = []
    for corpus_idx, shared in counter.items():
        corpus_size = corpus_sizes[corpus_idx]
        union = bench_size + corpus_size - shared
        estimated = shared / union if union else 0.0
        # keep candidates with any overlap; threshold filtering done by exact jaccard later
        # quick prune: if estimated far below threshold, skip exact jaccard
        # use relaxed prune to avoid false negatives: keep if shared >= 1
        candidates.append(corpus_idx)
    return candidates


def audit_corpus(corpus_records, bench_records, threshold=0.85, k=3):
    exact_set = {sha256_text(r["text"]) for r in corpus_records}
    norm_map = {}
    corpus_shingles_list = []
    corpus_norms = []
    for rec in corpus_records:
        normalized = normalize_text(rec["text"])
        norm_hash = sha256_text(normalized)
        if norm_hash not in norm_map:
            norm_map[norm_hash] = rec
        shingle_set = shingles(normalized, k)
        corpus_shingles_list.append(shingle_set)
        corpus_norms.append(normalized)

    # build indexed structures (sub-quadratic)
    use_lsh = has_datasketch and len(corpus_records) > 0
    lsh_index = None
    minhashes = []
    corpus_sizes = [len(s) for s in corpus_shingles_list]
    inverted_index = None
    if use_lsh:
        try:
            num_perm = 128
            lsh_index = MinHashLSH(threshold=threshold, num_perm=num_perm)
            for corpus_idx, shingle_set in enumerate(corpus_shingles_list):
                mhash = minhash_for_shingles(shingle_set, num_perm=num_perm)
                minhashes.append(mhash)
                lsh_index.insert(f"corpus_{corpus_idx}", mhash)
        except Exception:
            use_lsh = False
            lsh_index = None
            minhashes = []
    if not use_lsh:
        inverted_index = build_inverted_index(corpus_shingles_list)

    total = len(bench_records)
    exact_matches = 0
    norm_matches = 0
    near_dup = 0
    details = []
    affected_ids_set = set()

    for idx, bench_rec in enumerate(bench_records):
        text = bench_rec["text"]
        raw_hash = sha256_text(text)
        bench_id = bench_rec["id"]
        if raw_hash in exact_set:
            exact_matches += 1
            # find one matching corpus record for metadata (first match)
            matched = next((r for r in corpus_records if sha256_text(r["text"]) == raw_hash), corpus_records[0] if corpus_records else None)
            details.append(
                {
                    "benchmark_index": idx,
                    "benchmark_id": bench_id,
                    "match_type": "exact",
                    "jaccard": 1.0,
                    "benchmark_preview": text[:160],
                    "corpus_preview": (matched["text"][:160] if matched else text[:160]),
                    "corpus_id": matched["id"] if matched else None,
                    "corpus_metadata": corpus_metadata_from_record(matched) if matched else {},
                }
            )
            affected_ids_set.add(bench_id)
            continue
        normalized = normalize_text(text)
        norm_hash = sha256_text(normalized)
        if norm_hash in norm_map:
            norm_matches += 1
            matched = norm_map[norm_hash]
            details.append(
                {
                    "benchmark_index": idx,
                    "benchmark_id": bench_id,
                    "match_type": "normalized",
                    "jaccard": 1.0,
                    "benchmark_preview": text[:160],
                    "corpus_preview": matched["text"][:160],
                    "corpus_id": matched["id"],
                    "corpus_metadata": corpus_metadata_from_record(matched),
                }
            )
            affected_ids_set.add(bench_id)
            continue

        bench_shingles = shingles(normalized, k)
        bench_size = len(bench_shingles)
        best_jaccard = 0.0
        best_idx = None

        if use_lsh and lsh_index is not None:
            try:
                query_minhash = minhash_for_shingles(bench_shingles, num_perm=128)
                candidates = lsh_index.query(query_minhash)
                # candidates are keys like "corpus_3"
                candidate_indices = []
                for key in candidates:
                    try:
                        candidate_indices.append(int(key.split("_")[1]))
                    except Exception:
                        continue
                for corpus_idx in candidate_indices:
                    corpus_shingles = corpus_shingles_list[corpus_idx]
                    score = jaccard(bench_shingles, corpus_shingles)
                    if score > best_jaccard:
                        best_jaccard = score
                        best_idx = corpus_idx
                # fallback: if no candidate but bench has shingles, also check empty case
            except Exception:
                candidate_indices = []
                for corpus_idx, corpus_shingles in enumerate(corpus_shingles_list):
                    score = jaccard(bench_shingles, corpus_shingles)
                    if score > best_jaccard:
                        best_jaccard = score
                        best_idx = corpus_idx
        else:
            candidate_indices = query_candidates_inverted(
                bench_shingles, inverted_index, corpus_shingles_list, corpus_sizes, bench_size, threshold
            )
            for corpus_idx in candidate_indices:
                corpus_shingles = corpus_shingles_list[corpus_idx]
                score = jaccard(bench_shingles, corpus_shingles)
                if score > best_jaccard:
                    best_jaccard = score
                    best_idx = corpus_idx
                    if best_jaccard >= threshold:
                        # early break not safe for best, but can break if 1.0
                        if best_jaccard == 1.0:
                            break
            # if no candidates, best remains 0

        is_near = best_jaccard >= threshold
        if is_near:
            near_dup += 1
            matched = corpus_records[best_idx] if best_idx is not None else None
            details.append(
                {
                    "benchmark_index": idx,
                    "benchmark_id": bench_id,
                    "match_type": "near_duplicate",
                    "jaccard": round(best_jaccard, 4),
                    "benchmark_preview": text[:160],
                    "corpus_preview": (matched["text"][:160] if matched else ""),
                    "corpus_id": matched["id"] if matched else None,
                    "corpus_metadata": corpus_metadata_from_record(matched) if matched else {},
                }
            )
            affected_ids_set.add(bench_id)
        else:
            details.append(
                {
                    "benchmark_index": idx,
                    "benchmark_id": bench_id,
                    "match_type": "none",
                    "jaccard": round(best_jaccard, 4),
                    "benchmark_preview": text[:160],
                }
            )

    overlapped = exact_matches + norm_matches + near_dup
    overlap_percent = round(overlapped / total * 100, 2) if total else 0.0
    affected_benchmark_ids = sorted(affected_ids_set)
    return {
        "total": total,
        "total_benchmark": total,
        "exact": exact_matches,
        "exact_matches": exact_matches,
        "normalized": norm_matches,
        "normalized_matches": norm_matches,
        "near": {"count": near_dup, "jaccard_threshold": threshold, "shingle": k},
        "near_duplicates": near_dup,
        "threshold": threshold,
        "shingle_k": k,
        "corpus_size": len(corpus_records),
        "affected_benchmark_ids": affected_benchmark_ids,
        "affected_count": len(affected_benchmark_ids),
        "overlap_percent": overlap_percent,
        "details": details,
    }


audit = audit_corpus


def main():
    config_path = Path(__file__).parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit_cfg = config.get("audit", {})
    corpus = audit_cfg.get("corpus", "./data/output")
    benchmark = audit_cfg.get("benchmark")
    threshold = audit_cfg.get("threshold", 0.85)
    shingle = audit_cfg.get("shingle", 3)
    output = audit_cfg.get("output", "audit_report.json")
    limit_bench = audit_cfg.get("limit_bench", 0)

    corpus_records = load_records(corpus)
    if not corpus_records:
        logger.error(f"[audit] no corpus records found for corpus \"{corpus}\"")
        logger.error(
            "  examples: corpus data/output  corpus data/output/*.jsonl  corpus ./data/corpus.txt  corpus train.jsonl"
        )
        sys.exit(1)
    logger.info(f"[audit] corpus: {len(corpus_records)} samples from \"{corpus}\"")

    if not benchmark:
        logger.error("[audit] benchmark not set in config.yaml -> audit.benchmark")
        logger.error("  example: audit: {benchmark: \"path/to/benchmark.jsonl\"}")
        sys.exit(1)

    bench_records = load_records(benchmark)
    if not bench_records:
        logger.error(f"[audit] benchmark not found or empty: \"{benchmark}\"")
        logger.error("  expected format: jsonl with {\"text\": \"...\"} per line")
        sys.exit(1)
    if limit_bench and len(bench_records) > limit_bench:
        bench_records = bench_records[:limit_bench]
    logger.info(f"[audit] benchmark: {len(bench_records)} samples from \"{benchmark}\"")

    report = audit_corpus(
        corpus_records, bench_records, threshold=threshold, k=shingle
    )
    report["demo_mode"] = False

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pct = report["overlap_percent"]
    logger.info(
        f"[audit] total={report['total']} exact={report['exact']} normalized={report['normalized']} near={report['near']['count']} overlap={pct}% (threshold={threshold})"
    )
    if pct == 0:
        logger.info("[audit] PASS — no overlap above threshold")
    elif pct < 5:
        logger.info(
            "[audit] WARN — low overlap (<5%); review near_duplicates details for leakage"
        )
    else:
        logger.info(
            f"[audit] FAIL — {pct}% overlap exceeds 5% budget; remove overlapping corpus docs before DAPT"
        )
    logger.info(f"[audit] report: {out_path.resolve()}")


if __name__ == "__main__":
    main()
