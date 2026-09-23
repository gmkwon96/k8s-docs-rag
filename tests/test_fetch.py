import json
import subprocess

import pytest

from ingest.fetch import load_lock, main


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def commit_files(repo, files, message):
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def hugo(version):
    return f'[params]\nversion = "v{version}"\n'


@pytest.fixture
def origin(tmp_path, monkeypatch):
    """A local stand-in for kubernetes/website: release-1.36 cut, 1.37 on main."""
    for var in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{var}_NAME", "test")
        monkeypatch.setenv(f"GIT_{var}_EMAIL", "test@example.com")
    repo = tmp_path / "origin"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    # GitHub allows these; a local repo needs them for SHA fetches and --filter.
    git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    git(repo, "config", "uploadpack.allowFilter", "true")
    commit_files(
        repo,
        {
            "hugo.toml": hugo("1.37"),
            "content/en/docs/concepts/pods.md": "# Pods\n",
            "content/en/docs/_index.md": "# Docs\n",
            "content/en/examples/pods/simple-pod.yaml": "kind: Pod\n",
            "content/en/includes/task-tutorial-prereqs.md": "You need a cluster.\n",
            "content/ko/docs/concepts/pods.md": "# 파드\n",
            "static/logo.svg": "<svg/>",
        },
        "v1",
    )
    git(repo, "checkout", "-q", "-b", "release-1.36")
    commit_files(repo, {"hugo.toml": hugo("1.36")}, "cut release-1.36")
    git(repo, "checkout", "-q", "main")
    return repo


def run(origin, tmp_path, *extra, version="1.36"):
    lock = tmp_path / "sources.lock.json"
    data = tmp_path / "data"
    main(["--repo", f"file://{origin}", "--lock", str(lock), "--data-dir", str(data), *extra])
    return lock, data / f"v{version}"


def tip(origin, branch):
    return git(origin, "rev-parse", branch)


def test_requires_update_when_lock_missing(origin, tmp_path):
    with pytest.raises(SystemExit):
        run(origin, tmp_path)


def test_update_pins_branch_tip_and_checks_out_sparse_paths(origin, tmp_path, capsys):
    lock, dest = run(origin, tmp_path, "--update", "--versions", "1.36")

    src = load_lock(lock)["sources"]["1.36"]
    assert src.branch == "release-1.36"
    assert src.commit == tip(origin, "release-1.36")
    assert src.commit_date
    assert git(dest, "rev-parse", "HEAD") == src.commit

    files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if ".git" not in p.parts)
    files = [f for f in files if (dest / f).is_file()]
    assert files == [
        "content/en/docs/_index.md",
        "content/en/docs/concepts/pods.md",
        "content/en/examples/pods/simple-pod.yaml",
        "content/en/includes/task-tutorial-prereqs.md",
        "hugo.toml",
    ]
    assert "2 docs .md, 1 examples, 1 includes" in capsys.readouterr().out


def test_version_without_release_branch_maps_to_main(origin, tmp_path):
    lock, dest = run(origin, tmp_path, "--update", "--versions", "1.37", version="1.37")

    src = load_lock(lock)["sources"]["1.37"]
    assert src.branch == "main"
    assert src.commit == tip(origin, "main")
    assert 'version = "v1.37"' in (dest / "hugo.toml").read_text()


def test_hugo_version_mismatch_fails(origin, tmp_path):
    # No release-1.38 branch, so 1.38 maps to main, whose hugo.toml says v1.37.
    with pytest.raises(SystemExit, match="expected v1.38"):
        run(origin, tmp_path, "--update", "--versions", "1.38", version="1.38")


def test_existing_checkout_picks_up_new_sparse_paths(origin, tmp_path, monkeypatch):
    import ingest.fetch

    monkeypatch.setattr(ingest.fetch, "SPARSE_PATHS", ["content/en/docs"])
    _, dest = run(origin, tmp_path, "--update", "--versions", "1.36")
    assert not (dest / "content/en/includes").exists()

    monkeypatch.undo()
    run(origin, tmp_path)
    assert (dest / "content/en/includes/task-tutorial-prereqs.md").exists()


def test_fetch_uses_pinned_commit_not_branch_tip(origin, tmp_path):
    lock, dest = run(origin, tmp_path, "--update", "--versions", "1.36")
    pinned = load_lock(lock)["sources"]["1.36"].commit
    lock_before = lock.read_text()

    git(origin, "checkout", "-q", "release-1.36")
    commit_files(origin, {"content/en/docs/new.md": "# New\n"}, "v2")
    run(origin, tmp_path)

    assert git(dest, "rev-parse", "HEAD") == pinned
    assert not (dest / "content/en/docs/new.md").exists()
    assert lock.read_text() == lock_before


def test_second_run_is_noop(origin, tmp_path, capsys):
    run(origin, tmp_path, "--update", "--versions", "1.36")
    capsys.readouterr()
    run(origin, tmp_path)
    assert "up to date" in capsys.readouterr().out


def test_update_moves_existing_checkout_to_new_tip(origin, tmp_path):
    lock, dest = run(origin, tmp_path, "--update", "--versions", "1.36")
    git(origin, "checkout", "-q", "release-1.36")
    new_tip = commit_files(origin, {"content/en/docs/new.md": "# New\n"}, "v2")

    run(origin, tmp_path, "--update")

    assert load_lock(lock)["sources"]["1.36"].commit == new_tip
    assert git(dest, "rev-parse", "HEAD") == new_tip
    assert (dest / "content/en/docs/new.md").exists()
    assert json.loads(lock.read_text())["repo"] == f"file://{origin}"
