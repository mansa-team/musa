import hashlib
import json
import logging
import os
import random

logger = logging.getLogger(__name__)

CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "data",
    "output",
    "corpus-250M.jsonl",
)
N_DOCS = 100
SEED = 42


def normalize(text: str) -> str:
    return " ".join(text.split())


def docHash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def sampleHashes(corpusPath: str) -> list[str]:
    rng = random.Random(SEED)
    picked = []
    with open(corpusPath, "r", encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line).get("text", "")
            if not text.strip():
                continue
            if len(picked) < N_DOCS:
                picked.append(text)
            elif rng.random() < N_DOCS / (len(picked) + 1):
                picked[rng.randrange(N_DOCS)] = text
    rng.shuffle(picked)
    return [docHash(text) for text in picked[:N_DOCS]]


def countCopies(corpusPath: str, wanted: set) -> dict:
    counts = {key: 0 for key in wanted}
    with open(corpusPath, "r", encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line).get("text", "")
            if not text.strip():
                continue
            key = docHash(text)
            if key in counts:
                counts[key] += 1
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    wanted = set(sampleHashes(CORPUS_PATH))
    counts = countCopies(CORPUS_PATH, wanted)
    duped = sum(1 for key in wanted if counts[key] > 1)
    rate = duped / len(wanted)
    
    print(f"held-out docs: {len(wanted)}, with verbatim copy in pool: {duped}")
    print(f"leakage: {duped}/{len(wanted)} held-out docs occur verbatim in the training pool ({rate:.2%})")
