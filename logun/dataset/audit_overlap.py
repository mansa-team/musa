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

    text = "".join(char for char in text if unicodedata.category(char) != "Mn")

    text = text.lower()

    text = re.sub(r"[^\w\s]", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text, shingle_size=3):
    tokens = re.findall(r"\w+", text)

    if not tokens:
        return set()

    if len(tokens) < shingle_size:
        return {" ".join(tokens)}

    return {" ".join(tokens[i : i + shingle_size]) for i in range(len(tokens) - shingle_size + 1)}


def jaccard(first_set, second_set):
    if not first_set and not second_set:
        return 1.0

    if not first_set or not second_set:
        return 0.0

    return len(first_set & second_set) / len(first_set | second_set)


def build_corpus_metadata(obj: dict, fallback_source: str) -> dict:
    return {"source": obj.get("source") or fallback_source, "company": obj.get("company") or "", "cnpj": obj.get("cnpj") or "", "category": obj.get("category") or "", "subject": obj.get("subject") or "", "date": obj.get("date") or "", "year": obj.get("year") or 0, "document_id": obj.get("document_id") or "", "chunk_id": obj.get("chunk_id") if obj.get("chunk_id") is not None else "", "filename": obj.get("filename") or "", "source_url": obj.get("source_url") or "", "extraction_quality": obj.get("extraction_quality") or {}}


def load_records(path_str: str):
    input_path = Path(path_str)

    files = []

    if input_path.is_dir():
        files = sorted(input_path.rglob("*.jsonl"))
    elif input_path.is_file():
        files = [input_path]

    recs = []

    for file in files:
        if not file.is_file():
            continue

        sfx = file.suffix.lower()

        try:
            if sfx == ".jsonl":
                with open(file, encoding="utf-8", errors="ignore") as file_handle:
                    for idx, line in enumerate(file_handle):
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

            # .txt splitter removed — corpus is jsonl only
        except Exception:
            continue

    return recs


def build_inverted_index(corpus_shingles_list):
    inverted_index = collections.defaultdict(list)

    for i, shingle_set in enumerate(corpus_shingles_list):
        for shingle in shingle_set:
            inverted_index[shingle].append(i)

    return inverted_index


def query_candidates_inverted(bench_shingles, inverted_index):
    candidates = collections.Counter()

    for shingle in bench_shingles:
        for candidates_index in inverted_index.get(shingle, []):
            candidates[candidates_index] += 1

    return list(candidates.keys())


def audit_corpus(corpus_records, bench_records, threshold=0.85, shingle_size=3):
    exact_set = {sha256_text(record["text"]) for record in corpus_records}

    norm_map = {}

    corpus_shingles_list = []

    for rec in corpus_records:
        normalized_text = normalize_text(rec["text"])

        normalized_hash = sha256_text(normalized_text)

        if normalized_hash not in norm_map:
            norm_map[normalized_hash] = rec

        corpus_shingles_list.append(shingles(normalized_text, shingle_size))

    inverted_index = build_inverted_index(corpus_shingles_list)

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

            matched_record = next((record for record in corpus_records if sha256_text(record["text"]) == raw), corpus_records[0] if corpus_records else None)

            details.append({"benchmark_index": idx, "benchmark_id": bench_id, "match_type": "exact", "jaccard": 1.0, "benchmark_preview": bench_text[:160], "corpus_preview": (matched_record["text"][:160] if matched_record else bench_text[:160]), "corpus_id": matched_record["id"] if matched_record else None, "corpus_metadata": matched_record["corpus_metadata"] if matched_record else {}})

            affected.add(bench_id)

            continue

        normalized_text = normalize_text(bench_text)

        normalized_hash = sha256_text(normalized_text)

        if normalized_hash in norm_map:
            normalized += 1

            matched_record = norm_map[normalized_hash]

            details.append({"benchmark_index": idx, "benchmark_id": bench_id, "match_type": "normalized", "jaccard": 1.0, "benchmark_preview": bench_text[:160], "corpus_preview": matched_record["text"][:160], "corpus_id": matched_record["id"], "corpus_metadata": matched_record["corpus_metadata"]})

            affected.add(bench_id)

            continue

        bench_shingle_set = shingles(normalized_text, shingle_size)

        best = 0.0

        best_idx = None

        for candidate_index in query_candidates_inverted(bench_shingle_set, inverted_index):
            similarity_score = jaccard(bench_shingle_set, corpus_shingles_list[candidate_index])

            if similarity_score > best:
                best = similarity_score

                best_idx = candidate_index

            if best == 1.0:
                break

        if best >= threshold:
            near += 1

            matched_record = corpus_records[best_idx] if best_idx is not None else None

            details.append({"benchmark_index": idx, "benchmark_id": bench_id, "match_type": "near_duplicate", "jaccard": round(best, 4), "benchmark_preview": bench_text[:160], "corpus_preview": (matched_record["text"][:160] if matched_record else ""), "corpus_id": matched_record["id"] if matched_record else None, "corpus_metadata": matched_record["corpus_metadata"] if matched_record else {}})

            affected.add(bench_id)
        else:
            details.append({"benchmark_index": idx, "benchmark_id": bench_id, "match_type": "none", "jaccard": round(best, 4), "benchmark_preview": bench_text[:160]})

    overlap_percent = round((exact + normalized + near) / total * 100, 2) if total else 0.0

    return {"total": total, "exact": exact, "normalized": normalized, "near": {"count": near, "jaccard_threshold": threshold, "shingle": shingle_size}, "threshold": threshold, "shingle_k": shingle_size, "corpus_size": len(corpus_records), "affected_benchmark_ids": sorted(affected), "affected_count": len(affected), "overlap_percent": overlap_percent, "details": details}


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

    report = audit_corpus(corpus_records, benchmark_records, threshold=threshold, shingle_size=shingle)

    output_path = Path(output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    overlap_percent = report["overlap_percent"]

    logger.info(f"[audit] total={report['total']} exact={report['exact']} normalized={report['normalized']} near={report['near']['count']} overlap={overlap_percent}%")

    logger.info(f"[audit] report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
