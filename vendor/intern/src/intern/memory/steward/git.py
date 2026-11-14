from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from intern import names
from intern.memory.drive import MAIN, GitError, dirty_files, git, main_commit

AUTHOR = ("intern steward", "steward@intern.invalid")
PROPOSED = "proposed/"


def branch_exists(root: Path, branch: str) -> bool:
    try:
        git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    except GitError:
        return False
    return True


def _identity(root: Path) -> list[str]:
    try:
        git(root, "config", "user.email")
    except GitError:
        return ["-c", f"user.name={AUTHOR[0]}", "-c", f"user.email={AUTHOR[1]}"]
    return []


def commit_proposal(
    root: Path,
    branch: str,
    files: dict[str, str],
    message: str,
    trailers: dict[str, str],
    removed: list[str] | None = None,
) -> str:
    # Built in a private index on top of main, so a steward killed at any point leaves the working tree as it was.
    base = main_commit(root)
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git(root, "read-tree", base, env=env)
        names = list(files)
        blobs: list[str] = []
        if names:
            paths = [Path(tmp) / f"blob{i}" for i in range(len(names))]
            for path, name in zip(paths, names, strict=True):
                path.write_bytes(files[name].encode())
            stdin = "".join(f"{p}\n" for p in paths).encode()
            blobs = git(root, "hash-object", "-w", "--stdin-paths", stdin=stdin).decode().split()
        entries = "".join(f"100644 {blob}\t{name}\0" for blob, name in zip(blobs, names, strict=True))
        if entries:
            git(root, "update-index", "-z", "--index-info", stdin=entries.encode(), env=env)
        if removed:
            git(root, "update-index", "--force-remove", "--", *removed, env=env)
        tree = git(root, "write-tree", env=env).decode().strip()
    trailer = "\n".join(f"{k}: {v}" for k, v in trailers.items())
    commit = git(root, *_identity(root), "commit-tree", tree, "-p", base, "-m", message, "-m", trailer).decode().strip()
    # An empty old value: the branch must not exist yet.
    git(root, "update-ref", f"refs/heads/{branch}", commit, "")
    return commit


def publish(root: Path, branch: str) -> None:
    old = main_commit(root)
    new = git(root, "rev-parse", "--verify", f"refs/heads/{branch}^{{commit}}").decode().strip()
    try:
        git(root, "merge-base", "--is-ancestor", old, new)
    except GitError as e:
        raise GitError(f"{branch} does not fast-forward {MAIN}") from e
    on_main = _on_main(root)
    stray = dirty_files(root) if on_main else []
    git(root, "update-ref", f"refs/heads/{MAIN}", new, old)
    _reset(root, on_main, stray)
    git(root, "update-ref", "-d", f"refs/heads/{branch}", new)


def _on_main(root: Path) -> bool:
    try:
        return git(root, "symbolic-ref", "--quiet", "HEAD").decode().strip() == f"refs/heads/{MAIN}"
    except GitError:
        return False


def _reset(root: Path, on_main: bool, stray: list[str]) -> None:
    # The working tree is only a view of main; edits made in it outside the steward are discarded.
    if stray:
        print(f"{names.NAME}: discarding changes made outside the steward to {', '.join(stray)}", file=sys.stderr)
    if not on_main:
        git(root, "symbolic-ref", "HEAD", f"refs/heads/{MAIN}")
    git(root, "reset", "--quiet", "--hard", MAIN)


def recover(root: Path) -> None:
    # A merging steward that died left its proposal unpublished: publish it if main has not moved since, else drop it.
    # The tree goes back to main first: an older steward may have died with a proposed/ branch checked out.
    on_main = _on_main(root)
    _reset(root, on_main, dirty_files(root) if on_main else [])
    branches = git(root, "for-each-ref", "--format=%(refname:short)", f"refs/heads/{PROPOSED}").decode().split()
    for branch in branches:
        try:
            publish(root, branch)
        except GitError as e:
            print(f"{names.NAME}: dropping {branch}: {e}", file=sys.stderr)
            git(root, "branch", "--quiet", "-D", branch)
        else:
            print(f"published leftover {branch}")
