import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def readMean(path: str) -> float:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["mean"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfm-url", required=True)
    parser.add_argument("--minicpm-url", required=True)
    args = parser.parse_args()

    samples = os.path.join(SCRIPT_DIR, "samples.json")
    lfm = os.path.join(SCRIPT_DIR, "lfm.json")
    minicpm = os.path.join(SCRIPT_DIR, "minicpm.json")
    lfmScore = os.path.join(SCRIPT_DIR, "lfm_comet.json")
    minicpmScore = os.path.join(SCRIPT_DIR, "minicpm_comet.json")

    run([sys.executable, os.path.join(SCRIPT_DIR, "sample.py"), "--out", samples])
    run([sys.executable, os.path.join(SCRIPT_DIR, "translate.py"),
         "--url", args.lfm_url, "--in", samples, "--out", lfm, "--model-name", "lfm"])
    run([sys.executable, os.path.join(SCRIPT_DIR, "translate.py"),
         "--url", args.minicpm_url, "--in", samples, "--out", minicpm,
         "--model-name", "minicpm"])
    run([sys.executable, os.path.join(SCRIPT_DIR, "score_comet.py"),
         "--in", lfm, "--out", lfmScore])
    run([sys.executable, os.path.join(SCRIPT_DIR, "score_comet.py"),
         "--in", minicpm, "--out", minicpmScore])

    lfmMean = readMean(lfmScore)
    minicpmMean = readMean(minicpmScore)
    print(f"lfm mean={lfmMean:.4f} minicpm mean={minicpmMean:.4f}")
