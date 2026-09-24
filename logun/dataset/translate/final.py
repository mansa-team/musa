import argparse
import json
import os
import sys

import pandas as pd

SCHEMA_KEYS = ("sentiment", "source", "translated", "model", "quality")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Merge translations with labels into locked JSONL.")
    parser.add_argument("--translations", required=True, help="Input translations JSONL path.")
    parser.add_argument("--labels", required=True, help="Labels parquet or CSV path.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--model", required=True, help="Model id stamped on every row.")
    return parser.parse_args(argv)


def load_label_map(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    missing = {"sentiment", "source"} - set(frame.columns)
    if missing:
        raise KeyError("labels file missing columns: %s" % sorted(missing))
    label_map = {}
    for row in frame.itertuples(index=False):
        source = row.source
        if isinstance(source, str) and source not in label_map:
            label_map[source] = row.sentiment
    return label_map


def main(argv=None):
    args = parse_args(argv)
    try:
        label_map = load_label_map(args.labels)
    except (KeyError, OSError, ValueError) as exc:
        print("final.py: cannot load labels: %s" % exc, file=sys.stderr)
        return 1
    try:
        fin = open(args.translations, "r", encoding="utf-8")
    except OSError as exc:
        print("final.py: cannot open translations: %s" % exc, file=sys.stderr)
        return 1
    try:
        fout = open(args.out, "w", encoding="utf-8")
    except OSError as exc:
        print("final.py: cannot open output: %s" % exc, file=sys.stderr)
        fin.close()
        return 1
    n = 0
    with fin, fout:
        for lineno, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print("final.py: line %d: bad JSON: %s" % (lineno, exc), file=sys.stderr)
                return 1
            source = obj.get("source")
            translated = obj.get("translated")
            if not isinstance(source, str) or not source:
                print("final.py: line %d: missing/empty source" % lineno, file=sys.stderr)
                return 1
            if not isinstance(translated, str) or not translated:
                print("final.py: line %d: missing/empty translated" % lineno, file=sys.stderr)
                return 1
            if source not in label_map:
                print("final.py: line %d: no label for source %r" % (lineno, source), file=sys.stderr)
                return 1
            out = {
                "sentiment": label_map[source],
                "source": source,
                "translated": translated,
                "model": args.model,
                "quality": None,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d rows to %s" % (n, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
