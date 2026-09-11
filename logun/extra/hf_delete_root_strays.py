import os
import sys
import time
from dotenv import load_dotenv
from huggingface_hub import CommitOperationDelete, HfApi
load_dotenv()
REPO = "heitorrosa/cvm-corpus"
PDFS = "pdfs"
SLEEP_BETWEEN_YEARS = 5


def isDir(entry) -> bool:
    return entry.__class__.__name__ == "RepoFolder"


def partitionedYears(api: HfApi) -> dict:
    top = list(api.list_repo_tree(repo_id=REPO, repo_type="dataset", recursive=False))
    years = sorted(e.path for e in top if isDir(e) and e.path[:4].isdigit())
    found = {}
    for year in years:
        entries = list(api.list_repo_tree(repo_id=REPO, repo_type="dataset", path_in_repo=f"{year}/{PDFS}", recursive=False))
        partDirs = sorted(e.path.split("/")[-1] for e in entries if isDir(e) and e.path.split("/")[-1].startswith("part-"))
        if partDirs:
            found[year] = partDirs
    return found


def rootStrays(api: HfApi, year: str, partDirs: list) -> tuple:
    partFiles = set()
    for part in partDirs:
        for e in api.list_repo_tree(repo_id=REPO, repo_type="dataset", path_in_repo=f"{year}/{PDFS}/{part}", recursive=True):
            if not isDir(e) and e.path.lower().endswith(".pdf"):
                partFiles.add(e.path.split("/")[-1])
    entries = list(api.list_repo_tree(repo_id=REPO, repo_type="dataset", path_in_repo=f"{year}/{PDFS}", recursive=False))
    rootFiles = sorted(e.path.split("/")[-1] for e in entries if not isDir(e) and e.path.lower().endswith(".pdf"))
    overlap = sorted(set(rootFiles) & partFiles)
    orphans = sorted(set(rootFiles) - partFiles)
    return overlap, orphans


if __name__ == "__main__":
    execute = "--yes" in sys.argv
    only = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--year=")), None)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    found = partitionedYears(api)
    print(f"partitioned years: {sorted(found)}")
    for year, partDirs in sorted(found.items()):
        if only and year != only:
            continue
        overlap, orphans = rootStrays(api, year, partDirs)
        print(f"{year}: root strays with part copy={len(overlap)}, orphans (never touched)={len(orphans)}")
        if orphans:
            print(f"  orphans kept: {orphans[:10]}")
        if not overlap:
            continue
        if not execute:
            print(f"  dry-run: would delete {len(overlap)} root files under {year}/{PDFS}/ (re-run with --yes)")
            continue
        ops = [CommitOperationDelete(path_in_repo=f"{year}/{PDFS}/{name}") for name in overlap]
        url = api.create_commit(
            repo_id=REPO,
            repo_type="dataset",
            operations=ops,
            commit_message=f"remove {len(ops)} duplicated root pdfs in {year}/pdfs (kept in part-/)",
        )
        print(f"  deleted {len(ops)} root files: {url}")
        time.sleep(SLEEP_BETWEEN_YEARS)
    print("done (default is dry-run; pass --yes to execute, --year=YYYY for one year)")
