"""Downloads a pinned release of the PropBank frame files.

    python download_propbank.py                 # first time: resolves --ref, writes the lock
    python download_propbank.py                 # later: re-downloads exactly the locked commit
    python download_propbank.py --ref v3.4.0 --update   # deliberately move to another release

The repository's main branch keeps changing, so results built on "whatever main was that day"
cannot be reproduced. This script resolves a release tag (or any branch name or commit SHA) to
an exact commit, downloads that commit's archive, and records in data/propbank_lock.json:

    repo, ref, commit SHA, archive SHA-256, number of frame files, download date.

Commit the lock file. Every later download fetches the locked commit and fails if the archive
checksum differs. The experiment manifest records the lock and the paper cites the commit.

If the requested tag does not exist, the script lists the tags that do and stops; it never falls
back silently to the main branch.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO = "propbank/propbank-frames"
DEFAULT_REF = "v3.4.0"  # PropBank 3.4, the latest release per the repository README
DATA_DIR = Path("data")
FRAMES_DIR = DATA_DIR / "propbank-frames" / "frames"
LOCK_PATH = DATA_DIR / "propbank_lock.json"
CACHE_PATH = DATA_DIR / "propbank_cache.json"

Fetch = Callable[[str], bytes]


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "contextcanvas-propbank-downloader"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def list_tags(fetch: Fetch = http_get) -> List[str]:
    data = json.loads(fetch(f"https://api.github.com/repos/{REPO}/tags?per_page=100"))
    return [t["name"] for t in data]


def resolve_commit(ref: str, fetch: Fetch = http_get) -> str:
    """Resolves a tag, branch or SHA to a full commit SHA via the GitHub API."""
    try:
        data = json.loads(fetch(f"https://api.github.com/repos/{REPO}/commits/{ref}"))
    except urllib.error.HTTPError as e:
        if e.code in (404, 422):
            try:
                tags = list_tags(fetch)
            except Exception:
                tags = []
            hint = f"Available tags: {', '.join(tags)}" if tags else "No release tags are published."
            raise SystemExit(
                f"'{ref}' is not a tag, branch or commit of {REPO}. {hint}\n"
                f"Re-run with --ref <tag>, or pin the current release line with --ref <commit SHA>."
            ) from e
        raise
    return data["sha"]


def extract_frames(archive: bytes, frames_dir: Path) -> int:
    """Extracts <repo-root>/frames/*.xml from a GitHub zip archive; returns the file count."""
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for name in zf.namelist():
            parts = name.split("/")
            # GitHub archives have one top-level directory: <repo>-<sha>/frames/<file>.xml
            if len(parts) == 3 and parts[1] == "frames" and parts[2].endswith(".xml"):
                (frames_dir / parts[2]).write_bytes(zf.read(name))
                count += 1
    if count == 0:
        raise SystemExit("The archive contains no frames/*.xml files; is this the right repository/ref?")
    return count


def download(ref: Optional[str], update: bool, fetch: Fetch = http_get,
             lock_path: Path = LOCK_PATH, frames_dir: Path = FRAMES_DIR, cache_path: Path = CACHE_PATH) -> Dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    if lock and not update:
        if ref and ref != lock["ref"]:
            raise SystemExit(f"Locked to {lock['ref']} ({lock['commit'][:10]}); pass --update to change release.")
        commit, ref = lock["commit"], lock["ref"]
        print(f"Using locked PropBank frames {ref} @ {commit[:10]}")
    else:
        ref = ref or DEFAULT_REF
        commit = resolve_commit(ref, fetch)
        print(f"Resolved {ref} -> {commit}")

    archive = fetch(f"https://codeload.github.com/{REPO}/zip/{commit}")
    digest = hashlib.sha256(archive).hexdigest()
    if lock and not update and digest != lock["archive_sha256"]:
        raise SystemExit(f"Checksum mismatch for locked commit {commit[:10]}: expected "
                         f"{lock['archive_sha256'][:16]}..., got {digest[:16]}... Investigate before continuing.")
    n_files = extract_frames(archive, frames_dir)

    new_lock = {
        "repo": REPO, "ref": ref, "commit": commit, "archive_sha256": digest, "frame_files": n_files,
        "downloaded": lock["downloaded"] if (lock and not update) else time.strftime("%Y-%m-%d"),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(new_lock, indent=2) + "\n", encoding="utf-8")
    if cache_path.exists():
        cache_path.unlink()  # the catalog rebuilds it from the pinned XML on next use
    print(f"Extracted {n_files} frame files to {frames_dir}; lock written to {lock_path}")
    return new_lock


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=None, help=f"Release tag, branch or commit SHA (default: {DEFAULT_REF}).")
    ap.add_argument("--update", action="store_true", help="Ignore the existing lock and pin --ref anew.")
    ap.add_argument("--list-tags", action="store_true", help="Print the repository's tags and exit.")
    args = ap.parse_args()
    if args.list_tags:
        print("\n".join(list_tags()) or "(no tags)")
        return
    try:
        download(args.ref, args.update)
    except urllib.error.URLError as e:
        sys.exit(f"Network error: {e}")


if __name__ == "__main__":
    main()
