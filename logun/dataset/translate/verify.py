import argparse
import json
import sys

REQUIRED_KEYS = ("sentiment", "source", "translated", "model", "quality", "dataset", "endpoint", "idx")
ALLOWED_LABELS = {"positive", "negative", "neutral"}


def fail(msg):
    print("verify.py: %s" % msg, file=sys.stderr)
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify locked 8-field SFT JSONL.")
    parser.add_argument("--in", dest="inp", required=True, help="Input JSONL path.")
    args = parser.parse_args(argv)
    try:
        fin = open(args.inp, "r", encoding="utf-8")
    except OSError as exc:
        print("verify.py: cannot open input: %s" % exc, file=sys.stderr)
        return 1
    ok = True
    n = 0
    with fin:
        for lineno, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                ok = fail("line %d: bad JSON: %s" % (lineno, exc))
                continue
            if not isinstance(obj, dict) or tuple(sorted(obj.keys())) != tuple(sorted(REQUIRED_KEYS)):
                ok = fail("line %d: keys must be exactly %s" % (lineno, list(REQUIRED_KEYS)))
                continue
            if obj["sentiment"] not in ALLOWED_LABELS:
                ok = fail("line %d: bad sentiment %r" % (lineno, obj["sentiment"]))
            for key in ("source", "translated", "model", "dataset", "endpoint"):
                if not isinstance(obj[key], str) or not obj[key].strip():
                    ok = fail("line %d: %s must be a non-empty string" % (lineno, key))
            if obj["idx"] is None or isinstance(obj["idx"], bool) or not isinstance(obj["idx"], int):
                ok = fail("line %d: idx must be an integer" % lineno)
            quality = obj["quality"]
            if quality is not None and (isinstance(quality, bool) or not isinstance(quality, (int, float))):
                ok = fail("line %d: quality must be float or null" % lineno)
            n += 1
    print("%d rows checked in %s" % (n, args.inp))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
