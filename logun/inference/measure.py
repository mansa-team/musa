import json
import os
import subprocess
import sys
import urllib.request

import yaml
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))


CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8"))


URL = f"http://{CONFIG['llm']['host']}:{CONFIG['llm']['port']}"
PROMPT = (
    "O Banco Central manteve a taxa Selic em dois digitos e o mercado de juros futuros "
    "reagiu com alta nos vencimentos longos enquanto a bolsa recuou com bancos e varejo "
    "liderando as perdas do dia diante do pessimismo externo sobre inflacao e credito."
)
URL += "/completion"
N_PREDICT = 128
TEMPERATURE = 0
DEFAULT_OUT = os.path.join(HERE, "last-measure.json")


def postOnce(promptText):
    body = json.dumps(
        {"prompt": promptText, "n_predict": N_PREDICT, "temperature": TEMPERATURE, "stream": False,},
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def vramMiB():
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return int(proc.stdout.strip().split("\n")[0].strip())
    except Exception:
        return -1


# ponytail: stdlib urllib only, no requests dep.
try:
    args = sys.argv[1:]
    outPath = DEFAULT_OUT
    if "--out" in args:
        outIdx = args.index("--out")
        if outIdx + 1 >= len(args):
            raise ValueError("missing value for --out")
        outPath = args[outIdx + 1]
        
    print("warmup request sent")
    postOnce(PROMPT)  # run 1: first-touch warmup, discarded
    print("measured request sent")
    resp = postOnce(PROMPT)  # run 2: measured
    text = resp.get("content", "")
    timings = resp.get("timings", {}) or {}
    predN, predMs = timings.get("predicted_n", 0), timings.get("predicted_ms", 0)
    toksPerSec = (predN / (predMs / 1000.0)) if predN and predMs else 0.0
    sane = bool(text) and len(text) > 50
    print("decode n=" + str(predN) + " ms=" + str(predMs) + " sane=" + str(sane))
    result = {
        "toksPerSec": round(toksPerSec, 2),
        "vramMiB": vramMiB(),
        "sane": sane,
        "nPredict": N_PREDICT,
        "temperature": TEMPERATURE,
        "url": URL,
    }
    line = json.dumps(result, separators=(",", ":"),)
    with open(outPath, "w", encoding="utf-8",) as handle:
        handle.write(line + "\n")
    print(line)
except Exception as err:
    print(json.dumps({"error": str(err),}, separators=(",", ":"),))
    sys.exit(1)
