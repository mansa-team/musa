"""Delete 2003-2006 pdfs spam folders from heitorrosa/cvm-corpus (supervised, dry by default)."""
import os
import sys
from dotenv import load_dotenv


load_dotenv()


REPO_ID = "heitorrosa/cvm-corpus"
TARGETS = ["2003/pdfs", "2004/pdfs", "2005/pdfs", "2006/pdfs",]


def fail(msg: str) -> None:
    print(msg, file=sys.stderr,)
    sys.exit(1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--help" in args:
        print("usage: hf_repo_cleanup.py [--yes]  (dry run by default, deletes 2003-2006 pdfs folders)")
        sys.exit(0)
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        fail("missing HF_TOKEN env (set it before running supervised)")
    confirmed = "--yes" in args
    from huggingface_hub import HfApi
    api = HfApi(token=token,)
    try:
        remote = list(api.list_repo_files(REPO_ID, repo_type="dataset",))
    except Exception as e:
        fail(f"remote list failed: {e}")
    found, missing = [], []
    for target in TARGETS:
        if not target.endswith("/pdfs"):
            fail(f"refusing non-pdfs path: {target}")
        hits = [f for f in remote if f == target or f.startswith(target + "/",)]
        (found if hits else missing).append(target)
        print(f"{'found' if hits else 'empty/missing'}: {target} ({len(hits)} files)")
    if not confirmed:
        print(f"dry run: would delete {found}, run again with --yes to confirm")
        sys.exit(0)
    deleted, failed = [], []
    for target in found:
        try:
            api.delete_folder(
                repo_id=REPO_ID, path_in_repo=target, repo_type="dataset",
                commit_message=f"remove {target} spam",
            )
            deleted.append(target)
            print(f"deleted: {target}")
        except Exception as e:
            failed.append(target)
            print(f"FAILED {target}: {e}", file=sys.stderr,)
    print(f"summary: {len(deleted)}/{len(found)} deleted {deleted}, missing {missing}, failed {failed}")
    sys.exit(1 if failed else 0)
