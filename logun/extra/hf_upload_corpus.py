"""Supervised, stoppable/resumable upload of logun/dataset/data/ to heitorrosa/cvm-corpus via `hf` CLI."""
import fnmatch
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


logger = logging.getLogger(__name__)

REPO_ID = "heitorrosa/cvm-corpus"
DATA_DIR = Path(__file__).resolve().parent.parent / "dataset" / "data"
STATE_PATH = Path(__file__).resolve().with_name("hf_upload_state.json")
SCRATCH_DIR = Path(__file__).resolve().parent / ".tmp"
README_TEXT = (
    "# cvm-corpus\n\nCVM filings PT-BR corpus.\n\nDSIR 250M subset included "
    "(`output/corpus-250M.jsonl`).\n\nSource: https://github.com/mansa-team/musa\n"
)

WORKERS = 2
MAX_SHARD_FILES = 10000
SCAN_CACHE: dict = {}
PDF_DIR_CACHE: dict = {}


def run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env,)


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def entrySize(local: Path) -> int:
    if local.is_file():
        return local.stat().st_size
    return sum(f.stat().st_size for f in local.rglob("*") if f.is_file())


def scanEntry(name: str, local: Path, remoteName: str, extra: list) -> tuple:
    hit = SCAN_CACHE.get(name)
    if hit is not None:
        return hit
    extra = extra or []
    files: list = []
    total = 0
    if local.is_file():
        try:
            total = local.stat().st_size
        except OSError:
            total = 0
        files = [(local, remoteName,)]
    elif "/pdfs/part-" in remoteName:
        incs = [extra[i + 1] for i, arg in enumerate(extra) if arg == "--include" and i + 1 < len(extra)]
        key = str(local)
        names = PDF_DIR_CACHE.get(key)
        if names is None:
            names = []
            try:
                with os.scandir(local) as it:
                    for entry in it:
                        if not entry.is_file():
                            continue
                        try:
                            size = entry.stat().st_size
                        except OSError:
                            size = 0
                        names.append((entry.name, size, Path(entry.path),))
            except OSError:
                names = []
            PDF_DIR_CACHE[key] = names
        for fileName, size, absPath in names:
            if not any(fnmatch.fnmatch(fileName, pat) for pat in incs):
                continue
            files.append((absPath, f"{remoteName}/{fileName}",))
            total += size
    elif name.endswith("/root"):
        for f in local.rglob("*"):
            if not f.is_file():
                continue
            try:
                rel = f.relative_to(local)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] == "pdfs":
                continue
            try:
                total += f.stat().st_size
            except OSError:
                pass
            files.append((f, f.relative_to(DATA_DIR).as_posix(),))
    else:
        for f in local.rglob("*"):
            if not f.is_file():
                continue
            try:
                total += f.stat().st_size
            except OSError:
                pass
            files.append((f, f.relative_to(DATA_DIR).as_posix(),))
    SCAN_CACHE[name] = (files, total,)
    return files, total


def fmtSize(num: int) -> str:
    val, unit = float(num), "B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024 or unit == "TB":
            break
        val /= 1024
    return f"{val:.1f}{unit}" if unit == "B" else f"{val:.2f}{unit}"


def loadState() -> set:
    if not STATE_PATH.exists():
        return set()
    try:
        return set(json.loads(STATE_PATH.read_text()).get("completed", []))
    except (OSError, ValueError) as e:
        logger.warning(f"ignoring unreadable state file: {e}")
        return set()


def saveState(done: set) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"completed": sorted(done)}, indent=2) + "\n")
    os.replace(tmp, STATE_PATH)


def remotePaths() -> set:
    code = (f"from huggingface_hub import HfApi; api = HfApi(); "  # ponytail: hf 1.28 lacks files cmd
            f"print(chr(10).join(api.list_repo_files('{REPO_ID}', repo_type='dataset')))")
    try:
        proc = run(["python", "-c", code,])
    except OSError as e:
        logger.warning(f"remote file list unavailable: {e}")
        return set()
    if proc.returncode != 0:
        logger.warning(f"remote file list unavailable: {proc.stderr.strip()[:120]}")
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def pdfShards(pdfDir: Path) -> list:
    buckets: dict = {}
    with os.scandir(pdfDir) as it:
        for entry in it:
            if not entry.is_file() or not entry.name:
                continue
            buckets.setdefault(entry.name[0], []).append(entry.name)
    split: dict = {}
    for key, names in buckets.items():
        if len(names) > MAX_SHARD_FILES:
            for name in names:
                split.setdefault(name[:2], []).append(name)
        else:
            split[key] = names
    shards, cur, curCount = [], [], 0
    for key in sorted(split):
        if cur and curCount + len(split[key]) > MAX_SHARD_FILES:
            shards.append(cur)
            cur, curCount = [], 0
        cur.append(key)
        curCount += len(split[key])
    if cur:
        shards.append(cur)
    return shards


def shardIncludes(keys: list) -> list:
    singles = sorted(k for k in keys if len(k) == 1)
    doubles = sorted(k for k in keys if len(k) > 1)
    out = []
    if len(singles) == 1:
        out.append(f"{singles[0]}*")
    elif singles:
        out.append(f"[{''.join(singles)}]*")
    out += [f"{key}*" for key in doubles]
    return out


def partLabel(keys: list) -> str:
    first, last = min(keys), max(keys)
    if first == last:
        return f"part-{first}"
    return f"part-{first}-{last}"


def entryPaths(local: Path, remoteName: str, extra: list | None = None, name: str = "") -> set:
    if name:
        files, _ = scanEntry(name, local, remoteName, extra or [])
        return {remote for _, remote in files}
    if local.is_file():
        return {remoteName}
    extra = extra or []
    if "/pdfs/part-" in remoteName:
        incs = [extra[i + 1] for i, arg in enumerate(extra) if arg == "--include" and i + 1 < len(extra)]
        out: set = set()
        with os.scandir(local) as it:
            for entry in it:
                if entry.is_file() and any(fnmatch.fnmatch(entry.name, pat) for pat in incs):
                    out.add(f"{remoteName}/{entry.name}")
        return out
    return {f.relative_to(DATA_DIR).as_posix() for f in local.rglob("*") if f.is_file()}


def entryCountSize(name: str, local: Path, remoteName: str, extra: list) -> tuple:
    files, total = scanEntry(name, local, remoteName, extra)
    if not (name.endswith("/root") or name.endswith("/pdfs") or "/pdfs/part-" in remoteName):
        return None, total
    return len(files), total


def buildEntries(only: str | None) -> list:
    found = []
    for child in sorted(DATA_DIR.iterdir()):
        if child.is_dir() and child.name != "output":
            pdfDir = child / "pdfs"
            if not pdfDir.is_dir():
                found.append((child.name, child, child.name, []))
                continue
            found.append((f"{child.name}/root", child, child.name, ["--exclude", "pdfs/*",]))
            shards = pdfShards(pdfDir)
            if len(shards) == 1:
                dest = f"{child.name}/pdfs"
                found.append((dest, pdfDir, dest, []))
            else:
                for keys in shards:
                    extra: list = []
                    for glob in shardIncludes(keys):
                        extra += ["--include", glob,]
                    dest = f"{child.name}/pdfs/{partLabel(keys)}"
                    found.append((dest, pdfDir, dest, extra))
    outDir = DATA_DIR / "output"
    if outDir.is_dir():
        for child in sorted(outDir.iterdir()):
            if child.is_file():
                found.append((f"output/{child.name}", child, f"output/{child.name}", []))
    if only:
        found = [e for e in found if only in e[0]]
    for name, local, rn, extra in found:
        print(f"scanning {name}...")
        scanEntry(name, local, rn, extra)
    return found


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    # ponytail: C: TEMP full kills large folder uploads, stage hf temp files under repo instead
    scratchEnv = {**os.environ, "TEMP": str(SCRATCH_DIR), "TMP": str(SCRATCH_DIR), "TMPDIR": str(SCRATCH_DIR),}
    args = sys.argv[1:]
    if "--help" in args:
        print("usage: hf_upload_corpus.py [--dry-run] [--only=SUBSTR] [--redo=Y1,Y2]  (pdfs <=9000 single, 429 exits 75)")
        sys.exit(0)
    dryRun = "--dry-run" in args
    only = None
    redoRaw = None
    for i, arg in enumerate(args):
        if arg.startswith("--only="):
            only = arg.split("=", 1)[1]
        elif arg.startswith("--include="):
            only = arg.split("=", 1)[1]
        elif arg in ("--only", "--include") and i + 1 < len(args):
            only = args[i + 1]
        elif arg.startswith("--redo="):
            redoRaw = arg.split("=", 1)[1]
        elif arg == "--redo" and i + 1 < len(args):
            redoRaw = args[i + 1]
    redoSubs = [s for s in (redoRaw or "").split(",") if s]
    entries = buildEntries(only)
    if redoSubs:
        redone = loadState()
        dropped = sorted(n for n in redone if any(s in n for s in redoSubs))
        if dropped:
            saveState(redone - set(dropped))
        print(f"redo: {dropped} dropped from state, will re-upload")
    if not entries:
        fail("no entries found (check dataset/data layout or --only filter)")
    if dryRun:
        totalBytes = 0
        for name, local, rn, extra in entries:
            count, size = entryCountSize(name, local, rn, extra)
            totalBytes += size
            if count is None:
                print(f"would upload: {name} ({fmtSize(size)})")
            else:
                print(f"would upload: {name} ({count} files, {fmtSize(size)})")
        print(f"dry run: {len(entries)} entries, {fmtSize(totalBytes)} total, remote untouched")
        sys.exit(0)
    if run(["hf", "auth", "whoami"]).returncode != 0:
        fail("not authenticated with huggingface: run: hf auth login")
    proc = run(["hf", "repos", "create", REPO_ID, "--repo-type", "dataset", "--public",])
    out = (proc.stderr + proc.stdout).lower()  # ponytail: 409 means exists, retry would also 409
    if proc.returncode != 0 and ("409" in out or "conflict" in out or "exist" in out):
        print("repo exists, continuing")
    elif proc.returncode != 0:
        fail(f"repo create failed: {(proc.stderr or proc.stdout).strip()[:300]}")
    remote = remotePaths()
    if "README.md" not in remote:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tmpReadme:
            tmpReadme.write(README_TEXT)
        proc = run(
            ["hf", "upload", REPO_ID, tmpReadme.name, "README.md",
             "--repo-type", "dataset", "--commit-message", "add README",],
            env=scratchEnv,
        )
        os.unlink(tmpReadme.name)
        if proc.returncode != 0:
            fail(f"README upload failed: {proc.stderr.strip()[:300]}")
        print("uploaded: README.md")
    done = loadState()
    for n, local, rn, extra in entries:
        if any(s in n for s in redoSubs):
            continue
        paths = entryPaths(local, rn, extra, n)
        if paths and paths <= remote:
            done.add(n)
    total = len(entries)
    stateLock = threading.Lock()
    rateHit = threading.Event()
    todo = []
    for idx, (name, local, remoteName, extra) in enumerate(entries, 1):
        paths = entryPaths(local, remoteName, extra, name)
        if not any(s in name for s in redoSubs) and (name in done or (paths and paths <= remote)):
            done.add(name)
            print(f"[{idx}/{total}] skip {name} (already done)")
            continue
        todo.append((idx, name, local, remoteName, extra))

    def uploadOne(item: tuple) -> None:
        idx, name, local, remoteName, extra = item
        size = fmtSize(entryCountSize(name, local, remoteName, extra)[1])
        start = time.monotonic()
        try:
            proc = run(
                ["hf", "upload", REPO_ID, str(local), remoteName,
                 "--repo-type", "dataset", *extra, "--commit-message", f"upload {name}",],
                env=scratchEnv,
            )
        except OSError as e:
            print(f"[{idx}/{total}] FAIL {name} ({size}): {e}")
            return
        elapsed = time.monotonic() - start
        if proc.returncode != 0:
            err = proc.stderr.strip()
            if "429" in err or "Too Many Requests" in err:
                print(f"[{idx}/{total}] RATE LIMITED — state saved, wait ~1h and re-run")
                with stateLock:
                    saveState(done)
                rateHit.set()
                return
            print(f"[{idx}/{total}] FAIL {name} ({size}) after {elapsed:.0f}s: {err[-3000:]}")
            return
        with stateLock:
            done.add(name)
            saveState(done)
        print(f"[{idx}/{total}] done {name} ({size}) in {elapsed:.0f}s, state saved")

    futures = {}
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(uploadOne, item): item for item in todo}
            for fut in as_completed(futures):
                fut.result()
    except KeyboardInterrupt:
        for fut in futures:
            fut.cancel()  # ponytail: in-progress uploads restart that entry next run
        saveState(done)
        print(f"interrupted, state saved ({len(done)}/{total} done)")
        sys.exit(130)
    if rateHit.is_set():
        sys.exit(75)
    saveState(done)
    pending = [n for n, _, _, _ in entries if n not in done]
    uploadedBytes = sum(entryCountSize(n, local, rn, extra)[1] for n, local, rn, extra in entries if n in done)
    print(f"summary: {len(done)}/{total} done, {fmtSize(uploadedBytes)} uploaded, {len(pending)} pending: {pending}")
