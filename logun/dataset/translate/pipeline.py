import json
import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from translate import run as run_translate, ENDPOINT_FILE, loadEndpoints, canon
from final import main as final_main
from verify import main as verify_main

MODEL_ID = "minicpm5-2b"


def _stripBanked(src, dst):
    """Drop dst rows whose canon matches src rows (atomic tmp+rename).

    Re-fed rows are already banked, so resume would skip them. Stripping
    first lets the plain run() retranslate exactly those rows. No flags."""
    if not os.path.exists(dst):
        return 0
    doomed = set()
    with open(src, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                doomed.add(canon(json.loads(line).get("source", "")))
            except ValueError:
                continue
    if not doomed:
        return 0
    kept = []
    with open(dst, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if canon(record.get("source", "")) not in doomed:
                kept.append(line if line.endswith("\n") else line + "\n")
    tmp = dst + ".strip.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.writelines(kept)
    os.replace(tmp, dst)
    return len(doomed)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args(argv)
    in_path = os.path.join(SCRIPT_DIR, "clean.parquet")
    suffix = "_test" if args.max_rows else ""
    translated = os.path.join(SCRIPT_DIR, "translated%s.jsonl" % suffix)
    scored = os.path.join(SCRIPT_DIR, "scored%s.jsonl" % suffix)
    retry = os.path.join(SCRIPT_DIR, "retry%s.jsonl" % suffix)
    echoes = os.path.join(SCRIPT_DIR, "echoes%s.jsonl" % suffix)
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
        run_translate(in_path, translated, None, 120, args.max_rows)
    except RuntimeError as exc:
        print("pipeline: translate failed: %s" % exc, file=sys.stderr)
        return 1
    qpath = os.path.join(SCRIPT_DIR, "quarantine.jsonl")
    if os.path.exists(qpath):
        print("pipeline: quarantined rows present (see %s) - continuing with translated rows" % qpath)
    rc = None
    scores_arg = []
    try:
        from rescore import run as run_rescore, DEFAULT_MODEL_ID as KIWI_ID
    except Exception as exc:
        run_rescore = None
        print("pipeline: rescore unavailable (%s) - continuing with null quality" % exc, file=sys.stderr)
    if run_rescore is not None:
        try:
            run_rescore(translated, scored, retry, 0.5, KIWI_ID)
            scores_arg = ["--scores", scored]
        except Exception as exc:
            print("pipeline: rescore failed (%s) - continuing with null quality" % exc, file=sys.stderr)
    # Re-feed loop: retry + quarantine rows go back through translate.
    # translate appends and resume skips banked canons, so this is idempotent.
    # New failures re-quarantine themselves; one bounded pass, no loop.
    try:
        fed_any = False
        for pending in (retry, qpath, echoes):
            if os.path.exists(pending) and os.path.getsize(pending) > 0:
                print("pipeline: re-feeding %s" % pending, flush=True)
                _stripBanked(pending, translated)
                run_translate(pending, translated, None, 120)
                os.remove(pending)
                fed_any = True
        if fed_any and run_rescore is not None:
            run_rescore(translated, scored, retry, 0.5, KIWI_ID)
            scores_arg = ["--scores", scored]
    except RuntimeError as exc:
        print("pipeline: re-feed failed (%s) - continuing with translated rows" % exc, file=sys.stderr)
    rc = final_main(["--translations", translated, "--labels", in_path,
                     "--out", final_out, "--model", MODEL_ID,
                     "--echoes-out", echoes] + scores_arg)
    if rc != 0:
        return rc
    # Echoes (verbatim copies quarantined by final) get one retranslation pass,
    # then final rebuilds deterministically from translated.jsonl.
    try:
        if os.path.exists(echoes) and os.path.getsize(echoes) > 0:
            print("pipeline: re-feeding echoes", flush=True)
            _stripBanked(echoes, translated)
            run_translate(echoes, translated, None, 120)
            os.remove(echoes)
            if run_rescore is not None:
                try:
                    run_rescore(translated, scored, retry, 0.5, KIWI_ID)
                except Exception as exc:
                    print("pipeline: rescore failed (%s)" % exc, file=sys.stderr)
            rc = final_main(["--translations", translated, "--labels", in_path,
                             "--out", final_out, "--model", MODEL_ID,
                             "--echoes-out", echoes] + scores_arg)
            if rc != 0:
                return rc
    except RuntimeError as exc:
        print("pipeline: echo re-feed failed (%s)" % exc, file=sys.stderr)
    rc = verify_main(["--in", final_out])
    if rc != 0:
        return rc
    print("pipeline: DONE -> %s (metrics: %s)" % (final_out, os.path.join(SCRIPT_DIR, "metrics.jsonl")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
