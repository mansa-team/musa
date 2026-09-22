import argparse
import json
import re

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_mt(mt: str) -> str:
    mt = THINK_BLOCK.sub("", mt)
    if "</think>" in mt:
        mt = mt.rsplit("</think>", 1)[-1]
    if "<think>" in mt:  # stray open tag without close
        mt = mt.rsplit("<think>", 1)[-1]
    mt = mt.strip()
    if mt.endswith('"') and mt.count('"') == 1:
        mt = mt[:-1].strip()
    idx_para = mt.find("\n\n")
    idx_q = mt.find("Question:")
    cuts = [i for i in (idx_para, idx_q) if i != -1]
    if cuts:
        mt = mt[: min(cuts)]
    return mt.strip()


def main(argv: list = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with open(args.inp, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        row["mt"] = clean_mt(row["mt"])
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    if rows:
        with open(args.inp, "r", encoding="utf-8") as handle:
            raw = json.load(handle)[0]["mt"]
        print(f"row0: before={len(raw)} after={len(rows[0]['mt'])}")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
