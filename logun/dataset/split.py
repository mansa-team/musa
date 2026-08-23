import collections
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import yaml

# DSIR hashed 1+2gram importance resampling — Xie et al 2023 NeurIPS ArXiv 2302.03169
# Generative feature model over hashed n-grams, KL reduction via importance weight w = p/q.

CONFIG_DEFAULT = Path(__file__).parent / "config.yaml"
BUCKET_COUNT = 10000
SMOOTH_ALPHA = 1.0
TOKEN_PATTERN = re.compile(r"\w+")
TARGET_CATEGORIES = {"fato relevante", "comunicado ao mercado", "aviso aos acionistas"}


def _load_tokenizer(model_name):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None


def _count_tokens(text_value, tokenizer_obj):
    if tokenizer_obj is not None:
        try:
            return len(tokenizer_obj.encode(text_value, add_special_tokens=False))
        except Exception:
            pass
    return max(1, len(text_value) // 4)


def parse_token_target(raw_value):
    if isinstance(raw_value, int):
        if raw_value <= 0:
            raise ValueError(f"token_target must be positive got {raw_value}")
        return raw_value

    if isinstance(raw_value, float):
        if raw_value <= 0:
            raise ValueError(f"token_target must be positive got {raw_value}")
        return int(raw_value)

    text_value = str(raw_value).strip()

    if not text_value:
        raise ValueError("token_target is empty")

    upper_value = text_value.upper()
    suffix_map = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    suffix = upper_value[-1]

    if suffix in suffix_map:
        number_part = upper_value[:-1].strip()

        if not number_part:
            raise ValueError(f"invalid token_target '{raw_value}'")

        number_value = float(number_part)
        result = int(number_value * suffix_map[suffix])

        if result <= 0:
            raise ValueError(f"token_target must be positive got {raw_value}")

        return result

    result = int(float(text_value))

    if result <= 0:
        raise ValueError(f"token_target must be positive got {raw_value}")

    return result


def normalize_token_suffix(token_count):
    if token_count % 1_000_000_000 == 0:
        return f"{token_count // 1_000_000_000}B"

    if token_count % 1_000_000 == 0:
        return f"{token_count // 1_000_000}M"

    return str(token_count)


def hash_bucket(ngram_text):
    return int(hashlib.md5(ngram_text.encode()).hexdigest(), 16) % BUCKET_COUNT


def resolve_path(raw_path, base_dir):
    candidate = Path(raw_path)

    if candidate.is_absolute():
        return candidate

    return (base_dir / candidate).resolve()


def main():
    config_override = None

    if len(sys.argv) > 1:
        if sys.argv[1] == "--config" and len(sys.argv) >= 3:
            config_override = sys.argv[2]
        elif sys.argv[1].startswith("--config="):
            config_override = sys.argv[1].split("=", 1)[1]

    config_path = Path(config_override) if config_override else CONFIG_DEFAULT

    if not config_path.is_file():
        print(f"[split] config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(config_data, dict):
        print("[split] invalid config expected mapping", file=sys.stderr)
        sys.exit(1)

    dapt_config = config_data.get("dapt")

    if not isinstance(dapt_config, dict):
        print("[split] missing dapt section in config.yaml", file=sys.stderr)
        sys.exit(1)

    raw_target = dapt_config.get("token_target")

    if raw_target is None:
        print("[split] missing dapt.token_target in config.yaml (example: 250M)", file=sys.stderr)
        sys.exit(1)

    target_tokens = parse_token_target(raw_target)
    suffix_label = normalize_token_suffix(target_tokens)
    tokenizer_name = dapt_config.get("tokenizer", "answerdotai/ModernBERT-base")
    tokenizer_obj = _load_tokenizer(tokenizer_name)

    if tokenizer_obj is None:
        print(f"[split] tokenizer {tokenizer_name} not available fallback len//4", file=sys.stderr)
    else:
        print(f"[split] tokenizer {tokenizer_name} loaded", file=sys.stderr)

    paths_config = config_data.get("paths") or {}
    raw_output_dir = paths_config.get("output_dir", "./data/output")
    base_dir = Path(__file__).parent
    output_dir = resolve_path(raw_output_dir, base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"

    if not corpus_path.is_file():
        print(f"[split] corpus not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    output_path = output_dir / f"corpus-{suffix_label}.jsonl"
    manifest_path = output_dir / f"split-{suffix_label}.json"

    # first pass: collect source and target hashed distributions + bucket ngram decode
    source_counts = [0] * BUCKET_COUNT
    target_counts = [0] * BUCKET_COUNT
    bucket_to_ngram = collections.defaultdict(collections.Counter)
    by_category_total = collections.Counter()
    stored_entries = []
    stored_lines = []
    target_seed_size = 0

    with open(corpus_path, encoding="utf-8") as corpus_handle:
        for raw_line in corpus_handle:
            stripped = raw_line.strip()

            if not stripped:
                continue

            try:
                chunk_obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            text_value = chunk_obj.get("text") or ""
            category_value = (chunk_obj.get("category") or "").strip()
            category_lower = category_value.lower()
            by_category_total[category_value or "unknown"] += 1
            token_estimate = _count_tokens(text_value, tokenizer_obj)
            normalized = unicodedata.normalize("NFC", text_value).lower()
            token_list = TOKEN_PATTERN.findall(normalized)
            counter = collections.Counter()

            for token_item in token_list:
                bucket_index = hash_bucket(token_item)
                counter[bucket_index] += 1
                bucket_to_ngram[bucket_index][token_item] += 1

            for token_index in range(len(token_list) - 1):
                bigram_text = token_list[token_index] + " " + token_list[token_index + 1]
                bucket_index = hash_bucket(bigram_text)
                counter[bucket_index] += 1
                bucket_to_ngram[bucket_index][bigram_text] += 1

            for bucket_index, count_value in counter.items():
                source_counts[bucket_index] += count_value

            is_target = category_lower in TARGET_CATEGORIES

            if is_target:
                target_seed_size += 1

                for bucket_index, count_value in counter.items():
                    target_counts[bucket_index] += count_value

            stored_entries.append((counter, token_estimate, category_value or "unknown"))
            stored_lines.append(stripped)

    total_chunks = len(stored_entries)

    if target_seed_size < 2000:
        print(f"[split] target seed {target_seed_size} <2000 using available", file=sys.stderr)

    # DSIR log ratio gamma/nu with Laplace smoothing — Xie 2023 Eq.3 KL-proven r=0.82-0.89
    total_source = sum(source_counts)
    total_target = sum(target_counts)
    denom_source = total_source + SMOOTH_ALPHA * BUCKET_COUNT
    denom_target = total_target + SMOOTH_ALPHA * BUCKET_COUNT
    log_ratio = [0.0] * BUCKET_COUNT

    for bucket_index in range(BUCKET_COUNT):
        gamma = (target_counts[bucket_index] + SMOOTH_ALPHA) / denom_target if denom_target else 1.0 / BUCKET_COUNT
        nu_value = (source_counts[bucket_index] + SMOOTH_ALPHA) / denom_source if denom_source else 1.0 / BUCKET_COUNT
        log_ratio[bucket_index] = math.log(gamma / nu_value) if gamma > 0 and nu_value > 0 else 0.0

    # second pass: score each chunk w = sum cnt * log(p/q) — generative > discriminative 0.6-0.7%
    scored_entries = []

    for entry_index in range(total_chunks):
        counter, token_estimate, category_value = stored_entries[entry_index]
        weight = 0.0

        for bucket_index, count_value in counter.items():
            weight += count_value * log_ratio[bucket_index]

        scored_entries.append((weight, token_estimate, entry_index, category_value))

    scored_entries.sort(key=lambda scored: scored[0], reverse=True)

    selected_indices = []
    by_category_selected = collections.Counter()
    accumulated_tokens = 0

    for weight, token_estimate, entry_index, category_value in scored_entries:
        if accumulated_tokens >= target_tokens:
            break

        selected_indices.append(entry_index)
        accumulated_tokens += token_estimate
        by_category_selected[category_value] += 1

    # write selected in score order preserving original json line
    with open(output_path, "w", encoding="utf-8") as output_handle:
        for entry_index in selected_indices:
            output_handle.write(stored_lines[entry_index] + "\n")

    # top buckets for interpretability
    top_buckets = sorted(range(BUCKET_COUNT), key=lambda bucket_index: log_ratio[bucket_index], reverse=True)[:10]
    top_bucket_info = []

    for bucket_index in top_buckets:
        ngram_counter = bucket_to_ngram.get(bucket_index)

        if ngram_counter:
            top_ngram, top_count = ngram_counter.most_common(1)[0]
        else:
            top_ngram, top_count = "", 0

        top_bucket_info.append({"bucket": bucket_index, "log_ratio": round(log_ratio[bucket_index], 4), "top_ngram": top_ngram, "count": top_count})

    # KL divergence pre/post on hashed buckets — paper stats (Xie 2023 KL reduction r=0.82)
    selected_counts = [0] * BUCKET_COUNT
    selected_total = 0

    for entry_index in selected_indices:
        counter, _, _ = stored_entries[entry_index]

        for bucket_index, count_value in counter.items():
            selected_counts[bucket_index] += count_value
            selected_total += count_value

    denom_selected = selected_total + SMOOTH_ALPHA * BUCKET_COUNT if selected_total else BUCKET_COUNT
    kl_pre = 0.0
    kl_post = 0.0

    for bucket_index in range(BUCKET_COUNT):
        gamma = (target_counts[bucket_index] + SMOOTH_ALPHA) / denom_target if denom_target else 1.0 / BUCKET_COUNT
        nu_value = (source_counts[bucket_index] + SMOOTH_ALPHA) / denom_source if denom_source else 1.0 / BUCKET_COUNT
        sel_n = (selected_counts[bucket_index] + SMOOTH_ALPHA) / denom_selected

        if gamma > 0 and nu_value > 0:
            kl_pre += gamma * math.log(gamma / nu_value)

        if gamma > 0 and sel_n > 0:
            kl_post += gamma * math.log(gamma / sel_n)

    kl_reduction = kl_pre - kl_post
    kl_ratio = (kl_reduction / kl_pre) if kl_pre else 0.0
    manifest_data = {
        "target_tokens": target_tokens,
        "target_label": suffix_label,
        "actual_tokens": accumulated_tokens,
        "chunks_selected": len(selected_indices),
        "chunks_total": total_chunks,
        "target_seed_size": target_seed_size,
        "bucket_count": BUCKET_COUNT,
        "method": "DSIR hashed 1+2gram M=10000 KL-proven",
        "kl_divergence": {
            "pre": round(kl_pre, 4),
            "post": round(kl_post, 4),
            "reduction": round(kl_reduction, 4),
            "ratio": round(kl_ratio, 4),
        },
        "by_category_total": dict(by_category_total),
        "by_category_selected": dict(by_category_selected),
        "top_buckets": top_bucket_info,
        "input": str(corpus_path),
        "output": str(output_path),
    }

    manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[split] target={target_tokens} ({suffix_label}) actual={accumulated_tokens} selected={len(selected_indices)}/{total_chunks} seed={target_seed_size}")
    print(f"[split] wrote {output_path}")
    print(f"[split] manifest {manifest_path}")


if __name__ == "__main__":
    main()
