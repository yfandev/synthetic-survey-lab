from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import warnings
from collections.abc import Iterator
from importlib import resources
from pathlib import Path

from intern.memory.format import (
    ROOT_INDEX,
    SCHEMA_FILE,
    FormatError,
    Index,
    Page,
    build_indexes,
    index_file,
    page_file,
    parse_index,
    parse_page,
)
from intern.models.registry import ConfigError

MAIN = "main"
# Held shared while a session mounts the drive and exclusively while the steward moves its working tree.
TREE_LOCK = "nace-drive.lock"
# Held for a whole steward run, so stewards on one drive take turns.
STEWARD_LOCK = "nace-steward.lock"


class GitError(RuntimeError):
    pass


def git(root: Path, *args: str, stdin: bytes | None = None, env: dict[str, str] | None = None) -> bytes:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            input=stdin,
            capture_output=True,
            check=False,
            env={**os.environ, **env} if env else None,
        )
    except FileNotFoundError as e:
        raise GitError("git is not installed") from e
    if done.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {done.stderr.decode(errors='replace').strip()}")
    return done.stdout


def main_commit(root: Path) -> str:
    # Sessions and the steward read pages from main's commit, never from the working tree.
    top = Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != root.resolve():
        raise GitError(f"{root} is inside the repository {top}; a drive must be a repository's root")
    return git(root, "rev-parse", "--verify", f"refs/heads/{MAIN}^{{commit}}").decode().strip()


def tree_problem(root: Path) -> str:
    try:
        branch = git(root, "symbolic-ref", "--quiet", "--short", "HEAD").decode().strip()
    except GitError:
        branch = ""
    if branch != MAIN:
        return f"its working tree is on {repr(branch) if branch else 'a detached HEAD'}, not {MAIN!r}"
    if dirty := dirty_files(root):
        return f"its working tree has changes to {', '.join(dirty[:5])}" + (" and more" if len(dirty) > 5 else "")
    return ""


def dirty_files(root: Path) -> list[str]:
    out = git(root, "status", "--porcelain", "-z", "--untracked-files=no").decode(errors="replace")
    return [entry[3:] for entry in out.split("\0") if len(entry) > 3]


@contextlib.contextmanager
def locked(root: Path, name: str, shared: bool = False) -> Iterator[None]:
    try:
        handle = (root / ".git" / name).open("a")
    except OSError:
        # Not a repository's root, or one this user cannot write: there is nothing to coordinate with.
        yield
        return
    with handle:
        fcntl.flock(handle, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def create(root: Path) -> None:
    template = resources.files("intern.memory") / "template"
    (root / ".wiki").mkdir(parents=True, exist_ok=True)
    (root / ROOT_INDEX).write_text((template / "MEMORY.md").read_text())
    (root / SCHEMA_FILE).write_text((template / "schema.md").read_text())
    git(root, "init", "--quiet", "--initial-branch", MAIN)
    git(root, "add", "--all")
    git(
        root,
        "-c",
        "user.name=intern",
        "-c",
        "user.email=intern@localhost",
        "commit",
        "--quiet",
        "-m",
        "New memory drive",
    )


def tracked_files(root: Path, commit: str) -> dict[str, str]:
    names = [
        n for n in git(root, "ls-tree", "-r", "-z", "--name-only", commit).decode().split("\0") if n.endswith(".md")
    ]
    if not names:
        return {}
    out = git(root, "cat-file", "--batch", stdin="".join(f"{commit}:{n}\n" for n in names).encode())
    files: dict[str, str] = {}
    pos = 0
    for name in names:
        header_end = out.index(b"\n", pos)
        size = int(out[pos:header_end].split()[2])
        files[name] = out[header_end + 1 : header_end + 1 + size].decode(errors="replace")
        pos = header_end + 1 + size + 1
    return files


class Drive:
    def __init__(self, root: Path, commit: str, files: dict[str, str]) -> None:
        self.root = root
        self.commit = commit
        self.files = files
        self._built: dict[str, str] | None = None

    @classmethod
    def mount(cls, path: Path) -> Drive:
        root = path.expanduser().resolve()
        if not root.is_dir():
            raise ConfigError(f"memory drive {root} does not exist")
        try:
            with locked(root, TREE_LOCK, shared=True):
                commit = main_commit(root)
                drive = cls(root, commit, tracked_files(root, commit))
                problem = tree_problem(root)
        except GitError as e:
            raise ConfigError(f"memory drive {root} cannot be mounted: {e}") from e
        if problem:
            warnings.warn(
                f"memory drive {root}: {problem}; the session reads {MAIN} ({commit[:12]}), and the next steward "
                f"resets the tree to it",
                stacklevel=2,
            )
        return drive

    def resolve(self, name: str) -> Path:
        return self.root / name

    def page(self, path: str) -> Page | None:
        text = self.files.get(page_file(path))
        if text is None:
            return None
        try:
            return parse_page(path, text)
        except FormatError:
            return None

    def index(self, folder: str) -> Index | None:
        name = index_file(folder)
        for text in (self.files.get(name), self._indexes().get(name)):
            if text is None:
                continue
            try:
                return parse_index(name, text)
            except FormatError:
                continue
        return None

    def _indexes(self) -> dict[str, str]:
        if self._built is None:
            self._built = build_indexes(self.files)
        return self._built
