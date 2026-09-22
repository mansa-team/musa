import csv
import json
import os
import random

import yaml
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parent.parent.parent / "config.yaml").read_text(encoding="utf-8"))


SEED = CONFIG["seed"]
IN_CSV = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "financial_phrasebank", "financial_phrase_bank_pt_br.csv"))
OUT = os.path.join(SCRIPT_DIR, "samples.json")
N = 20


def sample(csvPath: str = None, n: int = N, seed: int = SEED) -> list:
    path = csvPath or IN_CSV
    with open(path, "r", encoding="latin-1") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(seed)
    picks = rng.sample(rows, min(n, len(rows)))
    picks = sorted(picks, key=lambda row: row["text"])
    return [
        {"id": idx, "text": row["text"], "ref": row["text_pt"], "y": row["y"]}
        for idx, row in enumerate(picks)
    ]


def main(argv: list = None) -> None:
    samples = sample(IN_CSV, N, SEED)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2, ensure_ascii=False)
    print(f"wrote {len(samples)} samples to {OUT}")


if __name__ == "__main__":
    main()
