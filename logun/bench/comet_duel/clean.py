import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent / "results"

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_mt(mt: str) -> str:
    mt = THINK_BLOCK.sub("", mt)
    if "</think>" in mt:
        mt = mt.rsplit("</think>", 1)[-1]
    if "<think>" in mt:  # stray open tag without close
        mt = mt.rsplit("<think>", 1)[-1]
    mt = mt.strip()
    # prefill rule: translate.py prefills assistant with ... é: " so the model
    # never emits the opening quote and leaves one unbalanced trailing quote;
    # strip exactly one trailing quote only when unbalanced (count == 1).
    if mt.endswith('"') and mt.count('"') == 1:
        mt = mt[:-1].strip()
    idx_para = mt.find("\n\n")
    idx_q = mt.find("Question:")
    cuts = [i for i in (idx_para, idx_q) if i != -1]
    if cuts:
        mt = mt[: min(cuts)]
    return mt.strip()


def clean_file(model: str) -> None:
    src = BASE / f"{model}.json"
    dst = BASE / f"{model}_clean.json"
    rows = json.loads(src.read_text(encoding="utf-8"))
    for row in rows:
        row["mt"] = clean_mt(row["mt"])
    dst.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        raw = json.loads(src.read_text(encoding="utf-8"))[0]["mt"]
        print(f"{model} row0: before={len(raw)} after={len(rows[0]['mt'])}")
    print(f"{model}: {len(rows)} rows -> {dst}")


if __name__ == "__main__":
    for m in ("lfm", "minicpm"):
        clean_file(m)
