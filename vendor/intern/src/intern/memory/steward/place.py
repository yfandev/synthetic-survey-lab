from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

PlaceKind = Literal["service", "project", "documents", "environment", "tool", "file"]

WORKSPACE_PAGE = "workspace"
FILE_SLUG_CHARS = 60
ENVIRONMENT_PAGE = "environment"
RESERVED = frozenset({ENVIRONMENT_PAGE, "general"})
# Files that mark the root of a code or package project, across ecosystems.
PROJECT_MARKERS = (
    ".git",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
)
_NAME_LINE = re.compile(r"""(?m)^\s*name\s*=\s*["']([^"']+)["']""")
_SETUP_NAME = re.compile(r"""\bname\s*=\s*["']([^"']+)["']""")
_GO_MODULE = re.compile(r"(?m)^module\s+(\S+)")


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-{}" else "-" for c in name.lower()).strip("-.") or WORKSPACE_PAGE


def page_for_host(host: str) -> str:
    for prefix in ("api.", "www."):
        host = host.removeprefix(prefix)
    return f"map/{_safe(host)}"


def page_for_tool(name: str) -> str:
    server = name.rsplit("__", 1)[0] if "__" in name else name.split("_", 1)[0]
    return f"map/{_safe(server.removeprefix('mcp__'))}"


def project_root(path: Path, cwd: Path) -> Path | None:
    folder = path if path.is_dir() else path.parent
    while folder == cwd or cwd in folder.parents:
        if any((folder / marker).exists() for marker in PROJECT_MARKERS):
            return folder
        folder = folder.parent
    return None


def _manifest_name(root: Path) -> str:
    for manifest, pattern in (
        ("pyproject.toml", _NAME_LINE),
        ("Cargo.toml", _NAME_LINE),
        ("setup.cfg", _NAME_LINE),
        ("setup.py", _SETUP_NAME),
        ("go.mod", _GO_MODULE),
    ):
        try:
            found = pattern.search((root / manifest).read_text(errors="replace"))
        except OSError:
            continue
        if found:
            return found[1].rsplit("/", 1)[-1]
    try:
        name = json.loads((root / "package.json").read_text(errors="replace")).get("name")
    except (OSError, ValueError, AttributeError):
        name = None
    return name.rsplit("/", 1)[-1] if isinstance(name, str) else ""


def project_name(root: Path, cwd: Path) -> str:
    # The workspace folder is named by whoever started the session, so only a manifest names it.
    name = _manifest_name(root)
    return name or (root.name if root != cwd else "")


def place_for_path(path: Path, cwd: Path) -> tuple[str, PlaceKind, Path]:
    root = project_root(path, cwd)
    if root is not None and (name := project_name(root, cwd)):
        return f"map/{_safe(name)}", "project", root
    parts = PurePosixPath(path.relative_to(cwd).as_posix()).parts
    if len(parts) > 1:
        name = _safe(parts[0])
        return f"map/{name}-files" if name in RESERVED else f"map/{name}", "documents", cwd / parts[0]
    return f"map/{WORKSPACE_PAGE}", "documents", cwd


def file_page(folder_page: str, relative: str) -> str:
    # One page per file, under its folder's page; a name like "history-2" would read as that folder's history.
    name = "-".join(re.findall(r"[a-z0-9]+", PurePosixPath(relative).name.lower()))[:FILE_SLUG_CHARS] or "file"
    return f"{folder_page}/{'file-' + name if name.startswith('history') else name}"


def is_folder_key(key: str) -> bool:
    return key.endswith("/")


def lesson_for(map_page: str) -> str:
    return "lessons/" + map_page.removeprefix("map/")


def name_of(page: str) -> str:
    return page.split("/", 1)[1] if "/" in page else page
