import argparse
import glob
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "models"))
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

import torch
from comet import download_model, load_from_checkpoint
from comet.models.download_utils import available_legacy_metrics, download_model_legacy

DEFAULT_MODEL_ID = "Unbabel/wmt22-cometkiwi-da"
DEFAULT_IN = os.path.join(SCRIPT_DIR, "translated.jsonl")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "scored.jsonl")
DEFAULT_RETRY = os.path.join(SCRIPT_DIR, "retry.jsonl")


def loadModel(model_id):
    short = model_id.split("/")[-1]
    local = glob.glob(os.path.join(
        CACHE_DIR, f"models--*--{short}", "snapshots", "*",
        "checkpoints", "model.ckpt"))
    if local:
        ckpt = sorted(local)[0]
    elif short in available_legacy_metrics:
        ckpt = download_model_legacy(short, CACHE_DIR)
    else:
        ckpt = download_model(model_id)
    return load_from_checkpoint(ckpt)


def run(inp, out_path, retry_path, threshold, model_id):
    rows = []
    with open(inp, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    idx = [i for i, r in enumerate(rows)
           if (r.get("translated") or "").strip() and not r.get("error")]
    data = [{"src": rows[i]["source"], "mt": rows[i]["translated"]} for i in idx]
    scores = []
    if data:
        model = loadModel(model_id)
        gpus = 1 if torch.cuda.is_available() else 0
        scores = [float(s) for s in
                  model.predict(data, batch_size=8, gpus=gpus).scores]
    for i, s in zip(idx, scores):
        rows[i]["score"] = s
    with open(out_path, "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    retry = [r for i, r in ((i, rows[i]) for i in idx)
             if r["score"] < threshold]
    with open(retry_path, "w", encoding="utf-8") as handle:
        for r in retry:
            handle.write(json.dumps({"source": r["source"],
                                     "dataset": r.get("dataset", "")},
                                    ensure_ascii=False) + "\n")
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"n_scored={len(scores)} mean={mean:.4f} "
          f"n_below={len(retry)} (thr={threshold}) -> {out_path} + {retry_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default=DEFAULT_IN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--retry", default=DEFAULT_RETRY)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()
    run(args.inp, args.out, args.retry, args.threshold, args.model_id)
