import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGUN_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
MODELS_FILE = os.path.join(LOGUN_DIR, "models.json")
LAUNCH = os.path.join(SCRIPT_DIR, "launch.py")
URL = "http://127.0.0.1:8080"
RESULTS = os.path.join(SCRIPT_DIR, "results")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    with open(MODELS_FILE, "r", encoding="utf-8") as handle:
        models = json.load(handle)
    if args.only:
        want = set(args.only.split(","))
        models = [m for m in models if m["name"] in want]
    os.makedirs(RESULTS, exist_ok=True)

    table = []
    for m in models:
        out = os.path.join(RESULTS, f"speed_{m['name']}.json")
        run([sys.executable, LAUNCH, "--model", m["gguf"]] + m.get("flags", []))
        try:
            wait_healthy()
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "measure.py"), "--out", out]
            run(cmd)
        finally:
            run([sys.executable, LAUNCH, "stop"])
        with open(out, "r", encoding="utf-8") as handle:
            row = json.loads(handle.read().splitlines()[0])
        row["model"] = m["name"]
        table.append(row)
        print(f"{m['name']}: {row.get('toksPerSec')} tok/s sane={row.get('sane')}")

    tablePath = os.path.join(RESULTS, "speed_table.json")
    with open(tablePath, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2)
    print(f"-> {tablePath}")
