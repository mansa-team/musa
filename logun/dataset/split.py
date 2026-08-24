import argparse
import collections
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import yaml

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except Exception:
    torch = None  # type: ignore
    HAS_CUDA = False

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


def _batch_token_counts(texts, tokenizer_obj, batch_size=2048):
    if tokenizer_obj is None or not texts:
        return [max(1, len(t) // 4) for t in texts]
    # ponytail: Rust encode_batch ~10k/s via tokenizers — no cuda move, tokenizer is CPU
    chunk_size = 4096
    out = []
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        # fastest: Rust encode_batch
        try:
            if hasattr(tokenizer_obj, "encode_batch"):
                encodings = tokenizer_obj.encode_batch(chunk)
                out.extend([max(1, len(e.ids)) if hasattr(e, "ids") else max(1, len(e)) for e in encodings])
                continue
        except Exception:
            pass
        try:
            if hasattr(tokenizer_obj, "batch_encode_plus"):
                res = tokenizer_obj.batch_encode_plus(chunk, add_special_tokens=False)
                out.extend([max(1, len(ids)) for ids in res["input_ids"]])
                continue
        except Exception:
            pass
        try:
            if hasattr(tokenizer_obj, "__call__"):
                res = tokenizer_obj(chunk, add_special_tokens=False, truncation=False)
                ids = res["input_ids"] if isinstance(res, dict) else getattr(res, "input_ids", None)
                if ids is not None:
                    out.extend([max(1, len(x)) for x in ids])
                    continue
        except Exception:
            pass
        for text_value in chunk:
            try:
                out.append(len(tokenizer_obj.encode(text_value, add_special_tokens=False)) or 1)
            except Exception:
                out.append(max(1, len(text_value) // 4))
    if len(out) == len(texts):
        return out
    # fallback sequential for any remainder
    while len(out) < len(texts):
        text_value = texts[len(out)]
        try:
            out.append(len(tokenizer_obj.encode(text_value, add_special_tokens=False)) or 1)
        except Exception:
            out.append(max(1, len(text_value) // 4))
    return out


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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", dest="config", default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4096)
    # keep legacy positional --config handling
    args, _unknown = parser.parse_known_args()
    # legacy: python split.py --config path  or  python split.py path
    config_override = args.config
    if config_override is None and len(sys.argv) > 1:
        if sys.argv[1] == "--config" and len(sys.argv) >= 3:
            config_override = sys.argv[2]
        elif sys.argv[1].startswith("--config="):
            config_override = sys.argv[1].split("=", 1)[1]
        elif not sys.argv[1].startswith("-"):
            # bare positional treated as config path for compat
            config_override = sys.argv[1]

    batch_size = max(1, args.batch_size)

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
        cuda_tag = " cuda" if HAS_CUDA else " cpu"
        print(f"[split] tokenizer {tokenizer_name} loaded{ cuda_tag} (batched, batch_size={batch_size})", file=sys.stderr)
    if HAS_CUDA:
        print("[split] cuda available — batched tokenization on GPU (chunks 4096)", file=sys.stderr)
    else:
        print("[split] cuda not available — cpu fallback", file=sys.stderr)

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

    # first pass: hashed distributions + defer token counts
    source_counts = [0] * BUCKET_COUNT
    target_counts = [0] * BUCKET_COUNT
    bucket_to_ngram = collections.defaultdict(collections.Counter)
    by_category_total = collections.Counter()
    stored_entries = []
    file_offsets = []
    target_seed_size = 0

    # micro-opts: bind locals, cache hash, fast digest
    md5_digest = hashlib.md5
    bucket_cache = {}
    findall = TOKEN_PATTERN.findall
    normalize = unicodedata.normalize

    def _hash_bucket_cached(ngram_text):
        cached = bucket_cache.get(ngram_text)
        if cached is not None:
            return cached
        # int.from_bytes big-endian == int(hexdigest,16) but avoids hex alloc/parse
        digest = md5_digest(ngram_text.encode()).digest()
        bucket = int.from_bytes(digest, "big") % BUCKET_COUNT
        bucket_cache[ngram_text] = bucket
        return bucket

    # read as binary to capture exact offsets without RAM duplicating lines
    with open(corpus_path, "rb") as corpus_handle:
        offset = 0
        for raw_bytes in corpus_handle:
            line_len = len(raw_bytes)
            curr_offset = offset
            offset += line_len
            stripped_bytes = raw_bytes.strip()
            if not stripped_bytes:
                continue
            try:
                chunk_obj = json.loads(stripped_bytes)
            except json.JSONDecodeError:
                continue
            text_value = chunk_obj.get("text") or ""
            category_value = (chunk_obj.get("category") or "").strip()
            category_lower = category_value.lower()
            by_category_total[category_value or "unknown"] += 1

            # placeholder estimate; overwritten by batched accurate counts before accumulation when tokenizer available
            token_estimate = max(1, len(text_value) // 4)

            normalized = normalize("NFC", text_value).lower()
            token_list = findall(normalized)
            # plain dict faster than Counter for per-chunk
            counter = {}
            for token_item in token_list:
                bucket_index = _hash_bucket_cached(token_item)
                counter[bucket_index] = counter.get(bucket_index, 0) + 1
                bucket_to_ngram[bucket_index][token_item] += 1
            # bigrams
            for token_index in range(len(token_list) - 1):
                bigram_text = token_list[token_index] + " " + token_list[token_index + 1]
                bucket_index = _hash_bucket_cached(bigram_text)
                counter[bucket_index] = counter.get(bucket_index, 0) + 1
                bucket_to_ngram[bucket_index][bigram_text] += 1
            for bucket_index, count_value in counter.items():
                source_counts[bucket_index] += count_value
            is_target = category_lower in TARGET_CATEGORIES
            if is_target:
                target_seed_size += 1
                for bucket_index, count_value in counter.items():
                    target_counts[bucket_index] += count_value
            stored_entries.append((counter, token_estimate, category_value or "unknown"))
            file_offsets.append(curr_offset)

    total_chunks = len(stored_entries)
    if target_seed_size < 2000:
        print(f"[split] target seed {target_seed_size} <2000 using available", file=sys.stderr)

    # DSIR log ratio
    total_source = sum(source_counts)
    total_target = sum(target_counts)
    denom_source = total_source + SMOOTH_ALPHA * BUCKET_COUNT
    denom_target = total_target + SMOOTH_ALPHA * BUCKET_COUNT
    log_ratio = [0.0] * BUCKET_COUNT
    for bucket_index in range(BUCKET_COUNT):
        gamma = (target_counts[bucket_index] + SMOOTH_ALPHA) / denom_target if denom_target else 1.0 / BUCKET_COUNT
        nu_value = (source_counts[bucket_index] + SMOOTH_ALPHA) / denom_source if denom_source else 1.0 / BUCKET_COUNT
        log_ratio[bucket_index] = math.log(gamma / nu_value) if gamma > 0 and nu_value > 0 else 0.0

    # second pass: score
    scored_entries = []
    for entry_index in range(total_chunks):
        counter, token_estimate, category_value = stored_entries[entry_index]
        weight = 0.0
        for bucket_index, count_value in counter.items():
            weight += count_value * log_ratio[bucket_index]
        scored_entries.append((weight, token_estimate, entry_index, category_value))
    scored_entries.sort(key=lambda scored: scored[0], reverse=True)

    # tokenizer path: recompute token counts for selected subset via batched GPU/CPU (deferred)
    if tokenizer_obj is not None and scored_entries:
        # Use fast estimate to get initial candidate set, then refine with batch accurate counts.
        # Collect candidate texts in batches until fast accumulated >= target (over-estimate), then batch-count those candidates.
        # Then walk sorted order with accurate counts until target met.
        # To avoid double reading, read selected candidate lines by offset.
        # First, determine how many fast-estimated chunks reach target (fast selection size)
        fast_selected = []
        fast_accum = 0
        for weight, token_estimate, entry_index, category_value in scored_entries:
            if fast_accum >= target_tokens:
                break
            fast_selected.append(entry_index)
            fast_accum += token_estimate
        # Expand candidate window a bit to handle under-estimation (fast may underestimate)
        # Estimate avg chars/token ~4, so accurate ~ fast. Expand by 20% buffer
        expand = int(len(fast_selected) * 0.2) + 500
        candidate_count = min(total_chunks, len(fast_selected) + expand)
        candidate_indices = [scored_entries[i][2] for i in range(candidate_count)]
        # sequential single-pass read: O(N) scan, no random seeks — replaces 108k thrashing seeks
        # ponytail: candidate_indices are in importance order; seek per idx in that order is random I/O
        if candidate_indices:
            needed_offsets = {file_offsets[idx] for idx in candidate_indices}
            offset_to_idx = {file_offsets[idx]: idx for idx in candidate_indices}
            max_needed = max(needed_offsets) if needed_offsets else 0
            idx_to_text = {}
            with open(corpus_path, "rb") as corpus_handle:
                curr_off = 0
                for raw_line in corpus_handle:
                    if curr_off in needed_offsets:
                        try:
                            obj = json.loads(raw_line.strip())
                            txt = obj.get("text") or ""
                        except Exception:
                            txt = ""
                        idx_to_text[offset_to_idx[curr_off]] = txt
                        if len(idx_to_text) == len(candidate_indices) and curr_off >= max_needed:
                            break
                    curr_off += len(raw_line)
                    if len(idx_to_text) == len(candidate_indices) and curr_off > max_needed:
                        break
            candidate_texts = [idx_to_text.get(idx, "") for idx in candidate_indices]
        else:
            candidate_texts = []
        # batch count (GPU batched when HAS_CUDA, CPU fallback otherwise; chunked 4096 internally)
        accurate_counts = _batch_token_counts(candidate_texts, tokenizer_obj, batch_size=batch_size)
        accurate_map = {candidate_indices[i]: accurate_counts[i] for i in range(len(candidate_indices))}
        # Now re-accumulate using accurate where available
        selected_indices = []
        by_category_selected = collections.Counter()
        accumulated_tokens = 0
        for weight, token_estimate, entry_index, category_value in scored_entries:
            if accumulated_tokens >= target_tokens:
                break
            actual = accurate_map.get(entry_index, token_estimate)
            # patch stored_entries
            counter_prev, _, cat_prev = stored_entries[entry_index]
            stored_entries[entry_index] = (counter_prev, actual, cat_prev)
            selected_indices.append(entry_index)
            accumulated_tokens += actual
            by_category_selected[category_value] += 1
        # If still not reaching target and we truncated candidate window, fall back to expanding — sequential scan
        if accumulated_tokens < target_tokens and candidate_count < total_chunks:
            remaining_indices = [scored_entries[pos][2] for pos in range(candidate_count, total_chunks)]
            if remaining_indices:
                needed_offsets_rem = {file_offsets[idx] for idx in remaining_indices}
                offset_to_idx_rem = {file_offsets[idx]: idx for idx in remaining_indices}
                idx_to_text_rem = {}
                with open(corpus_path, "rb") as corpus_handle:
                    curr_off = 0
                    for raw_line in corpus_handle:
                        if curr_off in needed_offsets_rem:
                            try:
                                obj = json.loads(raw_line.strip())
                                txt = obj.get("text") or ""
                            except Exception:
                                txt = ""
                            idx_to_text_rem[offset_to_idx_rem[curr_off]] = txt
                            if len(idx_to_text_rem) == len(remaining_indices):
                                break
                        curr_off += len(raw_line)
                batch_texts = []
                batch_idxs = []
                for pos in range(candidate_count, total_chunks):
                    if accumulated_tokens >= target_tokens:
                        break
                    _weight, _est, entry_index, _cat = scored_entries[pos]
                    txt = idx_to_text_rem.get(entry_index, "")
                    batch_texts.append(txt)
                    batch_idxs.append(entry_index)
                    if len(batch_texts) >= batch_size or pos == total_chunks - 1:
                        counts = _batch_token_counts(batch_texts, tokenizer_obj, batch_size=batch_size)
                        for bi, cnt in enumerate(counts):
                            ei = batch_idxs[bi]
                            counter_prev2, _, cat_prev2 = stored_entries[ei]
                            stored_entries[ei] = (counter_prev2, cnt, cat_prev2)
                            selected_indices.append(ei)
                            accumulated_tokens += cnt
                            by_category_selected[cat_prev2] += 1
                            if accumulated_tokens >= target_tokens:
                                break
                        batch_texts = []
                        batch_idxs = []
                        if accumulated_tokens >= target_tokens:
                            break
    else:
        selected_indices = []
        by_category_selected = collections.Counter()
        accumulated_tokens = 0
        for weight, token_estimate, entry_index, category_value in scored_entries:
            if accumulated_tokens >= target_tokens:
                break
            selected_indices.append(entry_index)
            accumulated_tokens += token_estimate
            by_category_selected[category_value] += 1

    # write selected in score order — sequential single-pass then emit in importance order (no random seeks)
    if selected_indices:
        needed_write = {file_offsets[i] for i in selected_indices}
        max_needed_write = max(needed_write) if needed_write else 0
        offset_to_line = {}
        with open(corpus_path, "rb") as corpus_handle:
            curr_off = 0
            for raw_line in corpus_handle:
                if curr_off in needed_write:
                    offset_to_line[curr_off] = raw_line.strip() + b"\n"
                    if len(offset_to_line) == len(needed_write) and curr_off >= max_needed_write:
                        break
                curr_off += len(raw_line)
                if len(offset_to_line) == len(needed_write) and curr_off > max_needed_write:
                    break
        with open(output_path, "wb") as output_handle:
            for entry_index in selected_indices:
                line = offset_to_line.get(file_offsets[entry_index])
                if line is not None:
                    output_handle.write(line)
    else:
        Path(output_path).write_bytes(b"")

    # ponytail: cap bucket_to_ngram — keep only top 500 log_ratio buckets (was 10k * Counter, huge for 111k chunks)
    if len(bucket_to_ngram) > 500:
        _keep = set(sorted(range(BUCKET_COUNT), key=lambda b: log_ratio[b], reverse=True)[:500])
        for _b in list(bucket_to_ngram.keys()):
            if _b not in _keep:
                del bucket_to_ngram[_b]
    top_buckets = sorted(range(BUCKET_COUNT), key=lambda bucket_index: log_ratio[bucket_index], reverse=True)[:10]
    top_bucket_info = []
    for bucket_index in top_buckets:
        ngram_counter = bucket_to_ngram.get(bucket_index)
        if ngram_counter:
            top_ngram, top_count = ngram_counter.most_common(1)[0]
        else:
            top_ngram, top_count = "", 0
        top_bucket_info.append({"bucket": bucket_index, "log_ratio": round(log_ratio[bucket_index], 4), "top_ngram": top_ngram, "count": top_count})

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
