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

# PAPER.md gate stays Unbabel/wmt22-cometkiwi-da once HF access is granted for
# account heitorrosa; default below is the ungated apache-2.0 fallback.
DEFAULT_MODEL_ID = "Unbabel/wmt20-comet-qe-da"


def main(argv: list = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    args = parser.parse_args(argv)
    with open(args.inp, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    short = args.model.split("/")[-1]
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
        ckpt = download_model(args.model)
    model = load_from_checkpoint(ckpt)
    data = [{"src": row["src"], "mt": row["mt"]} for row in rows]
    gpus = 1 if torch.cuda.is_available() else 0
    scores = model.predict(data, batch_size=8, gpus=gpus).scores
    scored = [{"id": row["id"], "score": float(score)}
              for row, score in zip(rows, scores)]
    mean = sum(row["score"] for row in scored) / len(scored) if scored else 0.0
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"mean": mean, "rows": scored}, handle, indent=2)
    print(f"mean={mean:.4f} n={len(scored)} -> {args.out}")


if __name__ == "__main__":
    main()
