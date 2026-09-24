import argparse
import json
import os
import queue
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(SCRIPT_DIR, "clean.parquet")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "translated.jsonl")
ENDPOINT_FILE = os.path.join(SCRIPT_DIR, "endpoints.txt")
ENDPOINT_RELOAD_SECS = 60
ENDPOINT_FAIL_LIMIT = 3
ENDPOINT_COOLDOWN_SECS = 300
TIMEOUT = 180
MAX_WORKERS = 64
WORKER_FANOUT = 4
SOURCE_ALIASES = ("source", "text", "sentence", "tweet", "headline", "input", "en")


def canon(text):
    return str(text).strip().casefold()


def wrrPick(states, endpoints, fallback=1.0):
    known = sorted(states[e]["ema"] for e in endpoints if states[e]["ema"])
    weight = known[len(known) // 2] if known else fallback
    total = 0.0
    for e in endpoints:
        w = states[e]["ema"] or weight
        states[e]["cur"] += w
        total += w
    best = max(endpoints, key=lambda e: states[e]["cur"])
    states[best]["cur"] -= total
    return best


def loadEndpoints():
    try:
        with open(ENDPOINT_FILE, encoding="utf-8") as handle:
            urls = []
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    urls.append(stripped)
            if urls:
                return urls
    except OSError:
        pass
    return []


def translateOne(text, endpoint, temperature=0.0):
    body = json.dumps({
        "messages": [{"role": "user", "content": text}],
        "temperature": temperature,
        "n_predict": min(256, max(128, 4 * len(text.split()))),
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        payload = json.loads(resp.read().decode())
    content = payload["choices"][0]["message"].get("content") or ""
    return content.strip(), payload.get("usage") or {}


def loadFrame(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        frame = pd.read_parquet(path)
    elif ext == ".csv":
        frame = pd.read_csv(path)
    elif ext in (".jsonl", ".json"):
        frame = pd.read_json(path, lines=(ext == ".jsonl"))
    else:
        raise ValueError("unsupported input (use parquet/csv/jsonl): " + path)
    for col in SOURCE_ALIASES:
        if col in frame.columns:
            if col != "source":
                frame = frame.rename(columns={col: "source"})
            break
    else:
        raise ValueError("no source text column (need one of %s)" % (",".join(SOURCE_ALIASES)))
    if "dataset" not in frame.columns:
        frame["dataset"] = ""
    return frame[["source", "dataset"]].dropna(subset=["source"])


def loadDone(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("translated") and not record.get("error"):
                    done.add(canon(record.get("source", "")))
    return done


def run(in_path, out_path, workers, timeout=120, max_rows=None):
    global TIMEOUT
    TIMEOUT = timeout
    frame = loadFrame(in_path)
    done = loadDone(out_path)
    requeued = set()
    rows = [(i, r) for i, r in frame.iterrows() if canon(r["source"]) not in done]
    if max_rows:
        rows = rows[:max_rows]
    total = len(rows)
    print("todo %d/%d (skipping %d done)" % (total, len(frame), len(frame) - total), flush=True)
    qpath = os.path.join(os.path.dirname(os.path.abspath(out_path)), "quarantine.jsonl")
    mpath = os.path.join(os.path.dirname(os.path.abspath(out_path)), "metrics.jsonl")
    todo = queue.Queue()
    for item in rows:
        todo.put(item)
    lock = threading.Lock()
    metrics_q = queue.Queue()
    count = 0
    quarantined = 0
    estate = {"endpoints": loadEndpoints(), "loaded": time.monotonic(), "health": {}}
    if not estate["endpoints"]:
        raise RuntimeError("no endpoints configured (edit %s)" % ENDPOINT_FILE)
    mhandle = open(mpath, "a", encoding="utf-8")
    metrics_q.put({"type": "start", "ts": time.time(), "total": total, "out": out_path})

    def flushMetrics():
        while True:
            try:
                event = metrics_q.get_nowait()
            except queue.Empty:
                return
            mhandle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def healthOf(endpoint):
        return estate["health"].setdefault(
            endpoint, {"fails": 0, "blocked_until": 0.0, "ema": None, "cur": 0.0})

    def pickEndpoint():
        with lock:
            now = time.monotonic()
            if now - estate["loaded"] >= ENDPOINT_RELOAD_SECS:
                estate["endpoints"] = loadEndpoints()
                estate["loaded"] = now
                for ep in estate["endpoints"]:
                    healthOf(ep)
                metrics_q.put({"type": "endpoints", "ts": time.time(),
                               "endpoints": list(estate["endpoints"])})
            healthy = [e for e in estate["endpoints"]
                       if healthOf(e)["blocked_until"] <= now]
            pool = healthy or sorted(estate["endpoints"],
                                     key=lambda e: healthOf(e)["blocked_until"])
            if not pool:
                raise RuntimeError("no endpoints configured")
            endpoint = wrrPick(estate["health"], pool)
        return endpoint

    def noteSuccess(endpoint, tok_s):
        with lock:
            health = healthOf(endpoint)
            if health["fails"] >= ENDPOINT_FAIL_LIMIT:
                metrics_q.put({"type": "recover", "ts": time.time(), "endpoint": endpoint})
            health["fails"] = 0
            health["blocked_until"] = 0.0
            health["ema"] = tok_s if health["ema"] is None else 0.3 * tok_s + 0.7 * health["ema"]
            return round(health["ema"], 2)

    def noteFail(endpoint):
        with lock:
            health = healthOf(endpoint)
            health["fails"] += 1
            if health["fails"] >= ENDPOINT_FAIL_LIMIT:
                health["blocked_until"] = time.monotonic() + ENDPOINT_COOLDOWN_SECS
                if health["fails"] == ENDPOINT_FAIL_LIMIT:
                    metrics_q.put({"type": "quarantine", "ts": time.time(), "endpoint": endpoint})

    def worker():
        nonlocal count, quarantined
        while True:
            try:
                idx, row = todo.get_nowait()
            except queue.Empty:
                return
            record = {"idx": int(idx), "source": row["source"], "translated": "",
                      "dataset": row["dataset"], "endpoint": ""}
            for _ in range(3):
                endpoint = pickEndpoint()
                record["endpoint"] = endpoint
                t0 = time.monotonic()
                try:
                    text, usage = translateOne(str(row["source"]), endpoint)
                    record["translated"] = text
                    record.pop("error", None)
                    lat = time.monotonic() - t0
                    comp = usage.get("completion_tokens", 0)
                    ema = noteSuccess(endpoint, comp / lat if lat > 0 else 0.0)
                    metrics_q.put({"type": "row", "ts": time.time(), "idx": int(idx),
                                   "endpoint": endpoint,
                                   "latency_s": round(lat, 3),
                                   "prompt_tokens": usage.get("prompt_tokens", 0),
                                   "comp_tokens": comp, "ema_tok_s": ema,
                                   "dataset": row["dataset"]})
                    break
                except Exception as exc:
                    record["error"] = str(exc)
                    noteFail(endpoint)
            with lock:
                key = canon(record["source"])
                if key not in done:
                    try:
                        dest = out_path if record.get("translated") and not record.get("error") else qpath
                        if dest == qpath:
                            quarantined += 1
                        with open(dest, "a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        done.add(key)
                    except OSError as exc:
                        if key in requeued:
                            print("translate: dropping row, disk failed twice (%s)" % exc, flush=True)
                            done.add(key)
                            count += 1
                        else:
                            requeued.add(key)
                            todo.put((idx, row))
                            print("translate: write failed, row requeued (%s)" % exc, flush=True)
                            flushMetrics()
                            continue
                count += 1
                if count % 50 == 0 or count == total:
                    print("progress %d/%d" % (count, total), flush=True)
                flushMetrics()

    workers = workers or min(max(WORKER_FANOUT * len(estate["endpoints"]), 8), MAX_WORKERS)
    if workers == 1:
        worker()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(workers):
                pool.submit(worker)
    with lock:
        flushMetrics()
        metrics_q.put({"type": "done", "ts": time.time(), "count": count,
                       "quarantined": quarantined})
        flushMetrics()
    mhandle.close()
    print("wrote %s (%d quarantined -> %s)" % (out_path, quarantined, qpath), flush=True)
    print("metrics -> %s" % mpath, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    parser.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    run(args.in_path, args.out_path, args.workers, args.timeout, args.max_rows)
