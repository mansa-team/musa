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

# PAPER.md gate scorer; checkpoint cached under logun/models (see resolver below).
DEFAULT_MODEL_ID = "Unbabel/wmt22-cometkiwi-da"


def run(cfg: dict) -> None:
    name = cfg.get("model_name", "lfm")
    inp = cfg.get("in") or os.path.join(SCRIPT_DIR, "results", name + "_clean.json")
    out_path = cfg.get("out") or os.path.join(SCRIPT_DIR, "results", name + "_scores.json")
    model_id = cfg.get("model_id", DEFAULT_MODEL_ID)
    with open(inp, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    short = model_id.split("/")[-1]
    # Local HF snapshot first (offline-safe, no S3): a snapshot_download of
    # e.g. Unbabel/wmt20-comet-qe-da lays down
    # models--<org>--<short>/snapshots/*/checkpoints/model.ckpt.
    local = glob.glob(os.path.join(
        CACHE_DIR, f"models--*--{short}", "snapshots", "*",
        "checkpoints", "model.ckpt"))
    if local:
        ckpt = sorted(local)[0]
    elif short in available_legacy_metrics:
        # Legacy S3 tarball (e.g. wmt20-comet-qe-da): ignores HF_HOME, so route
        # explicitly into logun/models (C: cannot fit the ~2GB download).
        ckpt = download_model_legacy(short, CACHE_DIR)
    else:
        ckpt = download_model(model_id)
    model = load_from_checkpoint(ckpt)
    data = [{"src": row["src"], "mt": row["mt"]} for row in rows]
    gpus = 1 if torch.cuda.is_available() else 0
    scores = model.predict(data, batch_size=8, gpus=gpus).scores
    scored = [{"id": row["id"], "score": float(score)}
              for row, score in zip(rows, scores)]
    mean = sum(row["score"] for row in scored) / len(scored) if scored else 0.0
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"mean": mean, "rows": scored}, handle, indent=2)
    print(f"mean={mean:.4f} n={len(scored)} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--model-name", default="lfm")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()
    run({"in": args.inp or None, "out": args.out or None,
         "model_name": args.model_name, "model_id": args.model_id})
