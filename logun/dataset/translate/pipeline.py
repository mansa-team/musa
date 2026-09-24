import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from translate import run as run_translate, ENDPOINT_FILE, loadEndpoints
from final import main as final_main
from verify import main as verify_main

MODEL_ID = "minicpm5-2b"


def main():
    in_path = os.path.join(SCRIPT_DIR, "clean.parquet")
    translated = os.path.join(SCRIPT_DIR, "translated.jsonl")
    final_out = os.path.join(SCRIPT_DIR, "final.jsonl")
    for path in (in_path, ENDPOINT_FILE):
        if not os.path.exists(path):
            print("pipeline: missing %s" % path, file=sys.stderr)
            return 1
    urls = loadEndpoints()
    if not urls:
        print("pipeline: no endpoints in %s" % ENDPOINT_FILE, file=sys.stderr)
        return 1
    print("pipeline: %d endpoint(s), translating %s" % (len(urls), in_path), flush=True)
    try:
        run_translate(in_path, translated, None)
    except RuntimeError as exc:
        print("pipeline: translate failed: %s" % exc, file=sys.stderr)
        return 1
    qpath = os.path.join(SCRIPT_DIR, "quarantine.jsonl")
    if os.path.exists(qpath):
        print("pipeline: quarantined rows present (see %s) - continuing with translated rows" % qpath)
    rc = None
    scored = os.path.join(SCRIPT_DIR, "scored.jsonl")
    scores_arg = []
    try:
        from rescore import run as run_rescore, DEFAULT_MODEL_ID as KIWI_ID
        run_rescore(translated, scored, os.path.join(SCRIPT_DIR, "retry.jsonl"), 0.5, KIWI_ID)
        scores_arg = ["--scores", scored]
    except Exception as exc:
        print("pipeline: rescore failed (%s) - continuing with null quality" % exc, file=sys.stderr)
    rc = final_main(["--translations", translated, "--labels", in_path,
                     "--out", final_out, "--model", MODEL_ID] + scores_arg)
    if rc != 0:
        return rc
    rc = verify_main(["--in", final_out])
    if rc != 0:
        return rc
    print("pipeline: DONE -> %s (metrics: %s)" % (final_out, os.path.join(SCRIPT_DIR, "metrics.jsonl")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
