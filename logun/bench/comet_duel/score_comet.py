import argparse
import json
import os

os.environ["HF_HOME"] = "D:/temp/hf"
os.environ["HF_HUB_CACHE"] = "D:/temp/hf"
os.environ["TRANSFORMERS_CACHE"] = "D:/temp/hf"
os.environ["XDG_CACHE_HOME"] = "D:/temp/hfcache"

import torch
from comet import download_model, load_from_checkpoint

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
    model = load_from_checkpoint(download_model(args.model))
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
