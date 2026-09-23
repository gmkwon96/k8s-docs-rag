"""Fetch Kubernetes docs from kubernetes/website release branches.

Each version is a shallow, blobless, sparse checkout under data/raw/release-<version>/
containing only the paths in SPARSE_PATHS. The exact commit per version is pinned in
ingest/sources.lock.json so every run (and the golden set built on it) sees the same docs.

Usage:
    uv run python -m ingest.fetch             # check out the commits pinned in the lock file
    uv run python -m ingest.fetch --update    # move pins to the current branch tips
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_URL = "https://github.com/kubernetes/website.git"
DEFAULT_VERSIONS = ["1.34", "1.35", "1.36"]
DOCS_PATH = "content/en/docs"
EXAMPLES_PATH = "content/en/examples"  # pulled into pages by the code_sample shortcode
SPARSE_PATHS = [DOCS_PATH, EXAMPLES_PATH]

ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = ROOT / "ingest" / "sources.lock.json"
DATA_DIR = ROOT / "data" / "raw"


@dataclass
class Source:
    branch: str
    commit: str
    commit_date: str


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def resolve_branch_tip(repo: str, branch: str) -> str:
    out = git("ls-remote", repo, f"refs/heads/{branch}")
    if not out:
        raise SystemExit(f"branch not found in {repo}: {branch}")
    return out.split()[0]


def checkout(repo: str, commit: str, dest: Path) -> bool:
    """Ensure dest is a sparse checkout of repo at commit. Returns True if anything changed."""
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        git("init", "-q", cwd=dest)
        git("remote", "add", "origin", repo, cwd=dest)
        # Same settings `git clone --filter=blob:none` would write: fetch blobs lazily.
        git("config", "remote.origin.promisor", "true", cwd=dest)
        git("config", "remote.origin.partialclonefilter", "blob:none", cwd=dest)
        git("sparse-checkout", "set", *SPARSE_PATHS, cwd=dest)
    elif head_commit(dest) == commit:
        return False

    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit, cwd=dest)
    git("-c", "advice.detachedHead=false", "checkout", "-q", "--detach", commit, cwd=dest)
    return True


def head_commit(dest: Path) -> str | None:
    try:
        return git("rev-parse", "HEAD", cwd=dest)
    except subprocess.CalledProcessError:
        return None


def count_files(dest: Path) -> dict[str, int]:
    return {
        "docs_md": sum(1 for _ in (dest / DOCS_PATH).rglob("*.md")),
        "examples": sum(1 for p in (dest / EXAMPLES_PATH).rglob("*") if p.is_file()),
    }


def load_lock(path: Path) -> dict:
    data = json.loads(path.read_text())
    data["sources"] = {v: Source(**s) for v, s in data["sources"].items()}
    return data


def write_lock(path: Path, repo: str, sources: dict[str, Source]) -> None:
    data = {"repo": repo, "sources": {v: asdict(s) for v, s in sorted(sources.items())}}
    path.write_text(json.dumps(data, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--update", action="store_true", help="re-pin to current branch tips")
    parser.add_argument("--versions", nargs="+", help="versions to pin (with --update)")
    parser.add_argument("--repo", help=f"git remote (default: lock file's, else {REPO_URL})")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)

    lock = load_lock(args.lock) if args.lock.exists() else None
    if lock is None and not args.update:
        parser.error(f"{args.lock} not found; run with --update to create it")
    if args.versions and not args.update:
        parser.error("--versions requires --update")

    repo = args.repo or (lock["repo"] if lock else REPO_URL)
    if args.update:
        versions = args.versions or (list(lock["sources"]) if lock else DEFAULT_VERSIONS)
        sources = {}
        for v in versions:
            branch = f"release-{v}"
            sources[v] = Source(branch, resolve_branch_tip(repo, branch), commit_date="")
    else:
        sources = lock["sources"]

    for v, src in sources.items():
        dest = args.data_dir / src.branch
        changed = checkout(repo, src.commit, dest)
        if not src.commit_date:
            src.commit_date = git("show", "-s", "--format=%cI", src.commit, cwd=dest)
        counts = count_files(dest)
        status = "fetched" if changed else "up to date"
        print(
            f"{v}: {src.commit[:12]} ({src.commit_date}) {status}, "
            f"{counts['docs_md']} docs .md, {counts['examples']} example files"
        )

    if args.update:
        write_lock(args.lock, repo, sources)
        print(f"wrote {args.lock}", file=sys.stderr)


if __name__ == "__main__":
    main()
