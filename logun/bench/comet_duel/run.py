import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

from translate import run as translate_run
from score import run as score_run

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGUN_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
MODELS_FILE = os.path.join(LOGUN_DIR, "models.json")
LAUNCH = os.path.join(LOGUN_DIR, "inference", "launch.py")
URL = "http://127.0.0.1:8080"
RESULTS = os.path.join(SCRIPT_DIR, "results")
SCORER = "Unbabel/wmt22-cometkiwi-da"


def run(cmd: list) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def wait_healthy(timeout: int = 180) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise RuntimeError("server never became healthy")


def paired(first: list, second: list) -> dict:
    a = {r["id"]: r["score"] for r in first}
    b = {r["id"]: r["score"] for r in second}
    ids = sorted(set(a) & set(b))
    d = [b[i] - a[i] for i in ids]
    m = statistics.mean(d)
    se = statistics.stdev(d) / (len(d) ** 0.5) if len(d) > 1 else 0.0
    return {"vs": None, "diff": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "wins": sum(1 for x in d if x > 0), "n": len(ids)}


def main(argv: list = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=os.path.join(SCRIPT_DIR, "samples.json"))
    parser.add_argument("--only", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    with open(MODELS_FILE, "r", encoding="utf-8") as handle:
        models = json.load(handle)
    if args.only:
        want = set(args.only.split(","))
        models = [m for m in models if m["name"] in want]
    os.makedirs(RESULTS, exist_ok=True)

    means, scoreRows = {}, {}
    for m in models:
        name = m["name"]
        raw = os.path.join(RESULTS, f"{name}.json")
        clean = os.path.join(RESULTS, f"{name}_clean.json")
        scores = os.path.join(RESULTS, f"{name}_scores.json")
        if not (args.resume and os.path.exists(raw)):
            run([sys.executable, LAUNCH, "--model", m["gguf"]] + m.get("flags", []))
            try:
                wait_healthy()
                translate_run({"url": URL, "samples": args.samples,
                               "out": raw, "model_name": name})
            finally:
                run([sys.executable, LAUNCH, "stop"])
        run([sys.executable, os.path.join(SCRIPT_DIR, "clean.py"),
             "--in", raw, "--out", clean])
        score_run({"in": clean, "out": scores})
        with open(scores, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        means[name] = data["mean"]
        scoreRows[name] = data["rows"]

    base = models[0]["name"]
    table = {"samples": os.path.basename(args.samples), "scorer": SCORER,
             "models": [{"name": n, "mean": means[n]} for n in means], "paired": []}
    for n in means:
        if n == base:
            continue
        p = paired(scoreRows[base], scoreRows[n])
        p["vs"] = f"{n}-minus-{base}"
        table["paired"].append(p)
    tablePath = os.path.join(RESULTS, "table.json")
    with open(tablePath, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2)
    for n in means:
        print(f"{n} mean={means[n]:.4f}")
    for p in table["paired"]:
        print(f"{p['vs']}: diff={p['diff']:+.4f} 95CI [{p['lo']:+.4f},{p['hi']:+.4f}] "
              f"wins={p['wins']}/{p['n']}")
    print(f"-> {tablePath}")


if __name__ == "__main__":
    main()
