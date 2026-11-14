"""Small private-file publication boundary for retained web content."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import suppress
from pathlib import Path


class PrivateStorageError(OSError):
    """The configured store or existing content cannot be used safely."""


def _private_directory(directory: Path) -> Path:
    if os.name != "posix":
        raise PrivateStorageError("Private web storage currently requires POSIX owner permissions.")
    original = directory.expanduser().absolute()
    # Check the explicit target before resolving ordinary OS aliases such as /tmp.
    if original.is_symlink():
        raise PrivateStorageError("The web store directory must not be a symlink.")
    original.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = original.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PrivateStorageError("The web store must be a private directory owned by the current user (0700).")
    return original.resolve()


def _verify_existing(path: Path, expected: bytes) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_size != len(expected)
        ):
            raise PrivateStorageError("Existing web content must be a private regular file with the expected size.")
        # A concurrent publisher may still hold its staging hard link. That is
        # safe: bytes were flushed before publication; do not reject st_nlink=2.
        if source.read(len(expected) + 1) != expected:
            raise PrivateStorageError("Existing web content failed its integrity check.")


def publish_private_file(directory: Path, name: str, data: bytes) -> Path:
    """Publish private immutable bytes, or verify a bounded existing duplicate.

    The directory is an owner-controlled store, not an executor sandbox against
    other processes running as this same user. Existing permissions are refused,
    never silently changed. Publication does not follow an existing target link.
    """
    if name in {"", ".", ".."} or Path(name).name != name or "\\" in name:
        raise PrivateStorageError("Web content names must be single path components.")
    folder = _private_directory(directory)
    path = folder / name
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".pending-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            _verify_existing(path, data)
        descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return path
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
