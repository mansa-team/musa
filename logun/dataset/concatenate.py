import json
from pathlib import Path

import yaml

config = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text(encoding="utf-8"))
paths = config["paths"]


def main():
    raw_out = Path(paths["output_dir"])
    output_dir = (Path(__file__).parent / raw_out).resolve() if not raw_out.is_absolute() else raw_out.resolve()

    corpus_txt = (Path(__file__).parent / Path(paths["corpus"])).resolve()
    corpus_jsonl = corpus_txt.with_suffix(".jsonl")
    corpus_jsonl.parent.mkdir(parents=True, exist_ok=True)

    legacy = output_dir / "cvm_dapt.jsonl"
    files = [legacy] if legacy.exists() else [f for f in sorted(output_dir.glob("*.jsonl")) if f.name not in ("manifest.json", "audit.json", "scrape_audit.json", "failures.jsonl", corpus_jsonl.name)]
    if not files:
        print(f"No files in {output_dir}" if not corpus_jsonl.exists() else f"corpus at {corpus_jsonl}")
        return

    recs = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
                if o.get("text"):
                    recs.append(o)
            except Exception:
                continue

    if not recs:
        print("No records")
        return

    with open(corpus_jsonl, "w", encoding="utf-8") as jf, open(corpus_txt, "w", encoding="utf-8") as tf:
        for o in recs:
            jf.write(json.dumps(o, ensure_ascii=False) + "\n")
            tf.write(o["text"].replace("\n", " ") + "\n")

    print(f"{len(recs)} chunks -> {corpus_jsonl}")


if __name__ == "__main__":
    main()
