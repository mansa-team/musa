"""SFT translation fan-out: clean sentiment frame -> pt-BR JSONL via llama-server pool.

LOCKED prompt config (MiniCPM5 noex): raw source sentence as the user prompt,
temperature 0. Request shape mirrors logun/bench/comet_duel/translate.py
(POST {base}/v1/chat/completions, urllib, TIMEOUT).
"""
import argparse
import json
import os
import queue
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ENDPOINTS = [
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8081"
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(SCRIPT_DIR, "clean.parquet")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "translated.jsonl")
TIMEOUT = 180
SOURCE_ALIASES = ("source", "text", "sentence", "tweet", "headline", "input", "en")


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
        content = json.loads(resp.read().decode())["choices"][0]["message"].get("content") or ""
    return content.strip()


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
                    done.add(record.get("source"))
    return done


def run(in_path, out_path, workers):
    frame = loadFrame(in_path)
    done = loadDone(out_path)
    rows = [(i, r) for i, r in frame.iterrows() if r["source"] not in done]
    total = len(rows)
    print("todo %d/%d (skipping %d done)" % (total, len(frame), len(frame) - total), flush=True)
    qpath = out_path + ".quarantine.jsonl"
    todo = queue.Queue()
    for item in rows:
        todo.put(item)
    lock = threading.Lock()
    count = 0
    quarantined = 0
    ep_idx = 0

    def pickEndpoint():
        nonlocal ep_idx
        with lock:
            endpoint = ENDPOINTS[ep_idx % len(ENDPOINTS)]
            ep_idx += 1
        return endpoint

    def worker():
        nonlocal count, quarantined
        while True:
            try:
                idx, row = todo.get_nowait()
            except queue.Empty:
                return
            record = {"source": row["source"], "translated": "",
                      "dataset": row["dataset"], "endpoint": ""}
            for _ in range(3):
                endpoint = pickEndpoint()
                record["endpoint"] = endpoint
                try:
                    record["translated"] = translateOne(str(row["source"]), endpoint)
                    record.pop("error", None)
                    break
                except Exception as exc:
                    record["error"] = str(exc)
            with lock:
                if record["source"] not in done:
                    done.add(record["source"])
                    dest = out_path if record.get("translated") and not record.get("error") else qpath
                    if dest == qpath:
                        quarantined += 1
                    with open(dest, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if count % 50 == 0 or count == total:
                    print("progress %d/%d" % (count, total), flush=True)

    workers = workers or len(ENDPOINTS)
    if workers == 1:
        worker()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(workers):
                pool.submit(worker)
    print("wrote %s (%d quarantined -> %s)" % (out_path, quarantined, qpath), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    parser.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=len(ENDPOINTS))
    args = parser.parse_args()
    run(args.in_path, args.out_path, args.workers)
