import collections
import concurrent.futures
import json
import logging
import math
import random
import time
import zlib
from multiprocessing import get_context
from pathlib import Path

import yaml

import numpy as np

CONFIG_DEFAULT = Path(__file__).resolve().parents[1] / "config.yaml"
BUCKET_COUNT = 10000
WORKERS = 8
SHARD_SIZE = 16384
SMOOTH_ALPHA = 1.0
FIXED_EXAMPLES = 1660
TARGET_CATEGORIES = {"fato relevante", "comunicado ao mercado", "aviso aos acionistas"}

logger = logging.getLogger(__name__)


def shard_task_counts_offset(args):
    shard_start_offset, shard_end_byte, source_path = args
    local_source = [0] * BUCKET_COUNT
    local_target = [0] * BUCKET_COUNT
    local_stored = []
    local_seed = 0

    with open(source_path, "rb") as file_handle:
        file_handle.seek(shard_start_offset)
        while True:
            current_pos = file_handle.tell()
            if shard_end_byte is not None and current_pos >= shard_end_byte:
                break
            raw = file_handle.readline()
            if not raw:
                break
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("is_duplicate") or obj.get("duplicate"):
                continue
            text = obj.get("text") or ""
            category = (obj.get("category") or "").strip() or "unknown"
            is_target = category.lower() in TARGET_CATEGORIES

            toks = text.lower().split(None, 512)[:512] if text else []
            counter = {}
            for tok_idx, tok in enumerate(toks):
                b = tok.encode()
                bucket_id = zlib.crc32(b) % BUCKET_COUNT
                counter[bucket_id] = counter.get(bucket_id, 0) + 1
                if tok_idx + 1 < len(toks):
                    bigram = tok + " " + toks[tok_idx + 1]
                    bucket_id2 = zlib.crc32(bigram.encode()) % BUCKET_COUNT
                    counter[bucket_id2] = counter.get(bucket_id2, 0) + 1
            for bucket_id, freq in counter.items():
                local_source[bucket_id] += freq
                if is_target:
                    local_target[bucket_id] += freq
            if is_target:
                local_seed += 1
            est = max(1, min(len(text) // 4, 8192)) if text else 1
            local_stored.append((counter, est, is_target))
    return local_source, local_target, local_stored, local_seed


def shard_task_score(args):
    shard_slice, log_ratio, start_idx, seed_value = args
    out = []
    for local_idx, (counter, est, is_target) in enumerate(shard_slice):
        doc_idx = start_idx + local_idx
        if isinstance(counter, dict):
            raw_score = 0.0
            for bucket_id, freq in counter.items():
                raw_score += freq * log_ratio[bucket_id]
        else:
            row_indices, row_data = counter
            raw_score = sum(int(row_data[pos]) * log_ratio[int(row_indices[pos])] for pos in range(len(row_indices)))
        rnd = random.Random(seed_value + doc_idx).random()
        if rnd < 1e-12:
            rnd = 1e-12
        if rnd > 1 - 1e-12:
            rnd = 1 - 1e-12
        gumbel_noise = -math.log(-math.log(rnd))
        perturbed_score = raw_score + gumbel_noise
        out.append((perturbed_score, raw_score, est, doc_idx))
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
    raw_str = str(raw_value).strip()
    if not raw_str:
        raise ValueError("token_target is empty")
    upper = raw_str.upper()
    suffix_map = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    suffix = upper[-1]
    if suffix in suffix_map:
        number_part = upper[:-1].strip()
        if not number_part:
            raise ValueError(f"invalid token_target '{raw_value}'")
        result = int(float(number_part) * suffix_map[suffix])
        if result <= 0:
            raise ValueError(f"token_target must be positive got {raw_value}")
        return result
    result = int(float(raw_str))
    if result <= 0:
        raise ValueError(f"token_target must be positive got {raw_value}")
    return result


def normalize_token_suffix(token_count):
    if token_count % 1_000_000_000 == 0:
        return f"{token_count // 1_000_000_000}B"
    if token_count % 1_000_000 == 0:
        return f"{token_count // 1_000_000}M"
    return str(token_count)


def resolve_path(raw_path, base):
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def main():
    t0 = time.time()
    logging.basicConfig(level=logging.INFO)
    config_path = CONFIG_DEFAULT
    if not config_path.is_file():
        logger.error(f"[split] config not found: {config_path}")
        raise SystemExit(1)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        logger.error("[split] invalid config expected mapping")
        raise SystemExit(1)
    dapt_config = config.get("dapt")
    if not isinstance(dapt_config, dict):
        logger.error("[split] missing dapt section in config.yaml")
        raise SystemExit(1)
    raw_target = dapt_config.get("token_target")
    if raw_target is None:
        logger.error("[split] missing dapt.token_target")
        raise SystemExit(1)
    target_tokens = parse_token_target(raw_target)
    suffix = normalize_token_suffix(target_tokens)
    seed_value = int(config.get("seed", dapt_config.get("seed", 42)))  # ponytail deterministic — config-driven seed
    random.seed(seed_value)
    np.random.seed(seed_value)
    output_dir = resolve_path(config.get("paths", {}).get("output_dir", "./data/output"), config_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        logger.error(f"[split] corpus not found: {corpus_path}")
        raise SystemExit(1)
    output_path = output_dir / f"corpus-{suffix}.jsonl"
    manifest_path = output_dir / f"split-{suffix}.json"

    offsets = []
    cats = []
    by_total = collections.Counter()
    duplicate_filtered = 0
    hoje_count = 0
    if np is not None:
        source_counts_arr = np.zeros(BUCKET_COUNT, dtype=np.int64)
        target_counts_arr = np.zeros(BUCKET_COUNT, dtype=np.int64)
    else:
        source_counts_arr = [0] * BUCKET_COUNT
        target_counts_arr = [0] * BUCKET_COUNT
    stored = []
    target_seed = 0

    logger.info(f"[split] featurizer split()+zlib.crc32 C 1+2gram M={BUCKET_COUNT} ({WORKERS} workers shard {SHARD_SIZE})")

    file_size = 0
    with open(corpus_path, "rb") as file_handle:
        offset_cursor = 0
        for raw in file_handle:
            current_offset = offset_cursor
            offset_cursor += len(raw)
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("is_duplicate") or obj.get("duplicate"):
                duplicate_filtered += 1
                continue
            text = obj.get("text") or ""
            cat = (obj.get("category") or "").strip() or "unknown"
            by_total[cat] += 1
            offsets.append(current_offset)
            cats.append(cat)
            if "hoje" in text.lower():
                hoje_count += 1
        file_size = offset_cursor

    total = len(offsets)
    logger.info(f"[split] scanned {total} docs, {len(by_total)} categories, duplicate_filtered={duplicate_filtered}, hoje={hoje_count}")

    mp_context = get_context("spawn")
    t_vec0 = time.time()

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp_context) as executor:
        for shard_start in range(0, total, SHARD_SIZE):
            shard_end = min(shard_start + SHARD_SIZE, total)
            start_byte = offsets[shard_start]
            end_byte = offsets[shard_end] if shard_end < total else file_size
            futures.append(executor.submit(shard_task_counts_offset, (start_byte, end_byte, str(corpus_path))))
        logger.info(f"[split] pass1 pipeline {WORKERS} workers shard {SHARD_SIZE} ({total} docs, {len(futures)} shards)")

        hashed = 0
        for future in futures:
            local_source, local_target, local_stored, local_seed = future.result()
            if np is not None:
                source_counts_arr += np.array(local_source, dtype=np.int64)
                target_counts_arr += np.array(local_target, dtype=np.int64)
            else:
                for bucket_id in range(BUCKET_COUNT):
                    if local_source[bucket_id]:
                        source_counts_arr[bucket_id] += local_source[bucket_id]
                    if local_target[bucket_id]:
                        target_counts_arr[bucket_id] += local_target[bucket_id]
            if np is not None:
                for counter_dict, est_val, is_target_val in local_stored:
                    row_indices = np.array(list(counter_dict.keys()), dtype=np.int64) if counter_dict else np.array([], dtype=np.int64)
                    row_data = np.array(list(counter_dict.values()), dtype=np.int64) if counter_dict else np.array([], dtype=np.int64)
                    stored.append((row_indices, row_data, est_val, is_target_val))
            else:
                for item in local_stored:
                    stored.append(item)
            target_seed += local_seed
            hashed += len(local_stored)
            if hashed // 100000 != (hashed - len(local_stored)) // 100000 or hashed == total:
                logger.info(f"[split] hashing {hashed}/{total} ({hashed/total:.0%})" if total else f"[split] hashing {hashed}")

    total = len(offsets)
    if total:
        if np is not None:
            logger.info(f"[split] pass1 done {time.time()-t_vec0:.1f}s source_total={int(np.sum(source_counts_arr))} target_total={int(np.sum(target_counts_arr))} seed={target_seed}")
        else:
            logger.info(f"[split] pass1 done {time.time()-t_vec0:.1f}s source_total={sum(source_counts_arr)} target_total={sum(target_counts_arr)} seed={target_seed}")
        if target_seed < 2000:
            logger.info(f"[split] target seed {target_seed} <2000 using available (FIXED_EXAMPLES={FIXED_EXAMPLES} verify)")

    if np is not None:
        source_counts = source_counts_arr.tolist() if hasattr(source_counts_arr, 'tolist') else list(source_counts_arr)
        target_counts = target_counts_arr.tolist() if hasattr(target_counts_arr, 'tolist') else list(target_counts_arr)
    else:
        source_counts = source_counts_arr
        target_counts = target_counts_arr

    total_source = sum(source_counts)
    total_target = sum(target_counts)
    denom_source = total_source + SMOOTH_ALPHA * BUCKET_COUNT
    denom_target = total_target + SMOOTH_ALPHA * BUCKET_COUNT

    if np is not None:
        source_arr_np = np.array(source_counts, dtype=np.float64)
        target_arr_np = np.array(target_counts, dtype=np.float64)
        gamma_arr = (target_arr_np + SMOOTH_ALPHA) / denom_target if denom_target else np.full(BUCKET_COUNT, 1.0 / BUCKET_COUNT)
        nu_arr = (source_arr_np + SMOOTH_ALPHA) / denom_source if denom_source else np.full(BUCKET_COUNT, 1.0 / BUCKET_COUNT)
        log_ratio_arr = np.log(gamma_arr / nu_arr)
        log_ratio = log_ratio_arr.tolist()
    else:
        log_ratio = [0.0] * BUCKET_COUNT
        for bucket_id in range(BUCKET_COUNT):
            gamma = (target_counts[bucket_id] + SMOOTH_ALPHA) / denom_target if denom_target else 1.0 / BUCKET_COUNT
            nu = (source_counts[bucket_id] + SMOOTH_ALPHA) / denom_source if denom_source else 1.0 / BUCKET_COUNT
            log_ratio[bucket_id] = math.log(gamma / nu) if gamma > 0 and nu > 0 else 0.0
        log_ratio_arr = None

    scored = []
    if WORKERS > 1 and len(stored) >= SHARD_SIZE:
        shard_batches = []
        for batch_start in range(0, len(stored), SHARD_SIZE):
            batch = stored[batch_start:batch_start + SHARD_SIZE]
            converted = []
            for row_indices, row_data, est, is_target in batch:
                if np is not None:
                    counter_proxy = (row_indices, row_data)
                else:
                    counter_proxy = row_indices
                converted.append((counter_proxy, est, is_target))
            shard_batches.append((converted, log_ratio, batch_start, seed_value))
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as executor2:
            futures2 = [executor2.submit(shard_task_score, args) for args in shard_batches]
            for fut in futures2:
                batch_scored = fut.result()
                for perturbed_score, raw_score, est, doc_idx in batch_scored:
                    scored.append((perturbed_score, raw_score, est, doc_idx, cats[doc_idx]))
    else:
        for idx, (row_indices, row_data, est, _) in enumerate(stored):
            if np is not None and log_ratio_arr is not None:
                if len(row_indices):
                    raw_score = float(np.dot(row_data.astype(np.float64), log_ratio_arr[row_indices]))
                else:
                    raw_score = 0.0
            else:
                if isinstance(row_indices, dict):
                    raw_score = sum(freq * log_ratio[bucket_id] for bucket_id, freq in row_indices.items())
                else:
                    raw_score = sum(int(row_data[pos]) * log_ratio[int(row_indices[pos])] for pos in range(len(row_indices)))
            rnd = random.Random(seed_value + idx).random()
            if rnd < 1e-12:
                rnd = 1e-12
            if rnd > 1 - 1e-12:
                rnd = 1 - 1e-12
            gumbel_noise = -math.log(-math.log(rnd))
            perturbed_score = raw_score + gumbel_noise
            scored.append((perturbed_score, raw_score, est, idx, cats[idx]))

    scored.sort(key=lambda entry: entry[0], reverse=True)

    selected = []
    by_selected = collections.Counter()
    accumulated = 0
    for perturbed_score, raw_score, est, idx, cat in scored:
        if accumulated >= target_tokens:
            break
        selected.append(idx)
        accumulated += est
        by_selected[cat] += 1

    if target_seed < FIXED_EXAMPLES:
        logger.info(f"[split] FIXED_EXAMPLES verify: target_seed {target_seed} < {FIXED_EXAMPLES} (using available)")
    else:
        logger.info(f"[split] FIXED_EXAMPLES verify: target_seed {target_seed} >= {FIXED_EXAMPLES} ok")

    if selected:
        need = {offsets[sel_idx] for sel_idx in selected}
        max_need = max(need)
        off_to_line = {}
        with open(corpus_path, "rb") as file_handle:
            current_offset = 0
            for raw in file_handle:
                if current_offset in need:
                    off_to_line[current_offset] = raw.strip() + b"\n"
                    if len(off_to_line) == len(need) and current_offset >= max_need:
                        break
                current_offset += len(raw)
                if len(off_to_line) == len(need) and current_offset > max_need:
                    break
        with open(output_path, "wb") as out_handle:
            for idx in selected:
                line = off_to_line.get(offsets[idx])
                if line is not None:
                    out_handle.write(line)
    else:
        Path(output_path).write_bytes(b"")

    top = sorted(range(BUCKET_COUNT), key=lambda bucket_id: log_ratio[bucket_id], reverse=True)[:10]
    top_info = [{"bucket": bucket_id, "log_ratio": round(log_ratio[bucket_id], 4), "top_ngram": "", "count": 0} for bucket_id in top]

    if np is not None:
        sel_counts_arr = np.zeros(BUCKET_COUNT, dtype=np.float64)
        sel_total = 0
        for idx in selected:
            row_indices, row_data, _, _ = stored[idx]
            if len(row_indices):
                np.add.at(sel_counts_arr, row_indices, row_data)
                sel_total += int(np.sum(row_data))
        denom_sel = sel_total + SMOOTH_ALPHA * BUCKET_COUNT if sel_total else BUCKET_COUNT
        gamma_arr_post = (target_arr_np + SMOOTH_ALPHA) / denom_target if denom_target else np.full(BUCKET_COUNT, 1.0 / BUCKET_COUNT)
        nu_arr_post = (source_arr_np + SMOOTH_ALPHA) / denom_source if denom_source else np.full(BUCKET_COUNT, 1.0 / BUCKET_COUNT)
        sel_n_arr = (sel_counts_arr + SMOOTH_ALPHA) / denom_sel
        kl_pre = float(np.sum(gamma_arr_post * np.log(gamma_arr_post / nu_arr_post)))
        kl_post = float(np.sum(gamma_arr_post * np.log(gamma_arr_post / sel_n_arr)))
        sel_counts = sel_counts_arr.tolist()
    else:
        sel_counts = [0] * BUCKET_COUNT
        sel_total = 0
        for idx in selected:
            row_item = stored[idx][0]
            if isinstance(row_item, dict):
                for bucket_id, freq in row_item.items():
                    sel_counts[bucket_id] += freq
                    sel_total += freq
            else:
                row_indices, row_data, _, _ = stored[idx]
                for pos in range(len(row_indices)):
                    bucket_id = int(row_indices[pos])
                    freq = int(row_data[pos])
                    sel_counts[bucket_id] += freq
                    sel_total += freq
        denom_sel = sel_total + SMOOTH_ALPHA * BUCKET_COUNT if sel_total else BUCKET_COUNT
        kl_pre = 0.0
        kl_post = 0.0
        for bucket_id in range(BUCKET_COUNT):
            gamma = (target_counts[bucket_id] + SMOOTH_ALPHA) / denom_target if denom_target else 1.0 / BUCKET_COUNT
            nu = (source_counts[bucket_id] + SMOOTH_ALPHA) / denom_source if denom_source else 1.0 / BUCKET_COUNT
            sel_n = (sel_counts[bucket_id] + SMOOTH_ALPHA) / denom_sel
            if gamma > 0 and nu > 0:
                kl_pre += gamma * math.log(gamma / nu)
            if gamma > 0 and sel_n > 0:
                kl_post += gamma * math.log(gamma / sel_n)

    kl_red = kl_pre - kl_post
    kl_ratio = (kl_red / kl_pre) if kl_pre else 0.0
    mean_tokens = accumulated / len(selected) if selected else 0
    expected_threshold = (2 * target_tokens / mean_tokens) if mean_tokens else 0
    hoje_ratio = (hoje_count / total) if total else 0
    duplicate_ratio = (duplicate_filtered / (total + duplicate_filtered)) if (total + duplicate_filtered) else 0
    kl_gate_pass = kl_post < 0.06 and kl_red > 0
    logger.info(f"[split] gates: KL pre={kl_pre:.4f} post={kl_post:.4f} red={kl_red:.4f} pass={kl_gate_pass} r=0.82 predictive")
    logger.info(f"[split] gates: duplicate_ratio={duplicate_ratio:.2%} (need <5%), hoje_ratio={hoje_ratio:.2%} (need ≤5%)")
    logger.info(f"[split] gates: expected_threshold={expected_threshold:.0f} vs selected={len(selected)} mean={mean_tokens:.0f}")

    manifest = {
        "target_tokens": target_tokens,
        "target_label": suffix,
        "actual_tokens": accumulated,
        "chunks_selected": len(selected),
        "chunks_total": total,
        "target_seed_size": target_seed,
        "bucket_count": BUCKET_COUNT,
        "method": "DSIR Gumbel resampled hashed 1+2gram M=10000 split+zlib.crc32 parallel shard offset-IPC",
        "seed": seed_value,
        "kl_divergence": {"pre": round(kl_pre, 4), "post": round(kl_post, 4), "reduction": round(kl_red, 4), "ratio": round(kl_ratio, 4), "gate_pass": kl_gate_pass},
        "gates": {
            "duplicate_ratio": round(duplicate_ratio, 4),
            "duplicate_pass": duplicate_ratio < 0.05,
            "hoje_ratio": round(hoje_ratio, 4),
            "hoje_pass": hoje_ratio <= 0.05,
            "kl_gate_pass": kl_gate_pass,
            "expected_threshold": round(expected_threshold, 1),
            "r_predictive": 0.82,
        },
        "by_category_total": dict(by_total),
        "by_category_selected": dict(by_selected),
        "top_buckets": top_info,
        "input": str(corpus_path),
        "output": str(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    dt_total = time.time() - t0
    logger.info(f"[split] target={target_tokens} ({suffix}) actual={accumulated} selected={len(selected)}/{total} seed={target_seed}")
    logger.info(f"[split] wrote {output_path}")
    logger.info(f"[split] manifest {manifest_path}")
    logger.info(f"[split] total time {dt_total:.1f}s est ~{dt_total/60:.1f} min")


if __name__ == "__main__":
    main()
