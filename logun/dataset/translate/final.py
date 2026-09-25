import argparse
import json
import os
import sys

import pandas as pd

ALLOWED_LABELS = ("positive", "negative", "neutral")


def canon(text):
    return str(text).strip().casefold()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Merge translations with labels into locked JSONL.")
    parser.add_argument("--translations", required=True, help="Input translations JSONL path.")
    parser.add_argument("--labels", required=True, help="Labels parquet or CSV path.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--model", required=True, help="Model id stamped on every row.")
    parser.add_argument("--scores", default=None, help="Optional scored JSONL carrying a score per source.")
    parser.add_argument("--echoes-out", default=None, help="Optional path to write verbatim-echo rows for retranslation.")
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
    dupes = 0
    for row in frame.itertuples(index=False):
        source = row.source
        if not isinstance(source, str):
            continue
        if canon(source) not in label_map:
            label_map[canon(source)] = row.sentiment
        else:
            dupes += 1
    if dupes:
        print("final.py: %d duplicate label sources (first wins)" % dupes, file=sys.stderr)
    return label_map


def load_score_map(path):
    scores = {}
    if not path:
        return scores
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print("final.py: scores line %d: bad JSON: %s" % (lineno, exc), file=sys.stderr)
                continue
            source = obj.get("source")
            score = obj.get("score")
            if not isinstance(source, str) or not source.strip():
                print("final.py: scores line %d: missing/empty source" % lineno, file=sys.stderr)
                continue
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                continue
            key = canon(source)
            scores[key] = float(score)  # last wins: matches the kept re-fed translation
    return scores


def main(argv=None):
    args = parse_args(argv)
    try:
        label_map = load_label_map(args.labels)
    except (KeyError, OSError, ValueError) as exc:
        print("final.py: cannot load labels: %s" % exc, file=sys.stderr)
        return 1
    try:
        score_map = load_score_map(args.scores)
    except OSError as exc:
        print("final.py: cannot open scores: %s" % exc, file=sys.stderr)
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
    eout = None
    if args.echoes_out:
        try:
            eout = open(args.echoes_out, "w", encoding="utf-8")
        except OSError as exc:
            print("final.py: cannot open echoes output: %s" % exc, file=sys.stderr)
            fin.close()
            fout.close()
            return 1
    n = 0
    skipped = 0
    echoes = 0
    rows = {}
    with fin:
        for lineno, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print("final.py: line %d: bad JSON: %s" % (lineno, exc), file=sys.stderr)
                skipped += 1
                continue
            source = obj.get("source")
            translated = obj.get("translated")
            if not isinstance(source, str) or not source.strip():
                print("final.py: line %d: missing/empty source" % lineno, file=sys.stderr)
                skipped += 1
                continue
            if not isinstance(translated, str) or not translated.strip():
                print("final.py: line %d: missing/empty translated" % lineno, file=sys.stderr)
                skipped += 1
                continue
            key = canon(source)
            if canon(translated) == key:
                print("final.py: line %d: verbatim echo, quarantined for retranslation" % lineno, file=sys.stderr)
                skipped += 1
                echoes += 1
                if eout is not None:
                    eout.write(json.dumps({"idx": obj.get("idx"), "source": source,
                                           "dataset": obj.get("dataset", "")}, ensure_ascii=False) + "\n")
                continue
            if key not in label_map:
                print("final.py: line %d: no label for source %r" % (lineno, source), file=sys.stderr)
                skipped += 1
                continue
            sentiment = label_map[key]
            if sentiment not in ALLOWED_LABELS:
                print("final.py: line %d: unmapped sentiment %r" % (lineno, sentiment), file=sys.stderr)
                skipped += 1
                continue
            out = {
                "sentiment": sentiment,
                "source": source,
                "translated": translated,
                "model": args.model,
                "quality": score_map.get(key),
                "dataset": obj.get("dataset", ""),
                "endpoint": obj.get("endpoint", ""),
                "idx": obj.get("idx"),
            }
            rows[key] = out  # last good wins: re-fed corrections overwrite stale rows
    if eout is not None:
        eout.close()
    with fout:
        for out in rows.values():
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d rows to %s (%d skipped, %d echoes)" % (n, args.out, skipped, echoes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
