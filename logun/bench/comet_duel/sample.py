import argparse
import csv
import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "financial_phrasebank", "financial_phrase_bank_pt_br.csv"))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "samples.json")
N = 20
SEED = 42


def sample(csvPath: str = None, n: int = N, seed: int = SEED) -> list:
    path = csvPath or DEFAULT_CSV
    with open(path, "r", encoding="latin-1") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(seed)
    picks = rng.sample(rows, min(n, len(rows)))
    return [
        {"id": idx, "text": row["text"], "ref": row["text_pt"], "y": row["y"]}
        for idx, row in enumerate(picks)
    ]


def main(argv: list = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default=DEFAULT_CSV)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    samples = sample(args.inp, args.n, args.seed)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2, ensure_ascii=False)
    print(f"wrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()
