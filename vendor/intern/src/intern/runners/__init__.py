"""Runners by name: the built-in `local`, `docker` and `opensandbox`, or one a package registers.

A package registers its own under the `intern.runners` entry-point group (`e2b = "intern_e2b:E2BRunner"`), or
config names it as `module:Class`. See `intern.runners.base` for the interface and
docs/capabilities/runners.md for the contract.
"""

from __future__ import annotations

import functools
import importlib
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from intern.executor import Executor, LocalExecutor
from intern.runners.base import NoOptions, Runner
from intern.runners.docker import DockerRunner

__all__ = ["BUILTIN", "ENTRY_POINT_GROUP", "LocalRunner", "NoOptions", "Runner", "load_runner", "runner_names"]

ENTRY_POINT_GROUP = "intern.runners"
DEFAULT = "local"


class LocalRunner(Runner):
    """Commands run directly on this machine, as the user who started the harness. Session setup keeps the session's
    own executor for it; `create` serves callers that build one outside a session."""

    name = "local"
    title = "Local"
    note = "commands run directly on this machine"
    jobs = True
    isolated = False

    def create(self, options: BaseModel, *, workspace: Path, cwd: Path) -> Executor:
        return LocalExecutor()


BUILTIN: dict[str, type[Runner]] = {LocalRunner.name: LocalRunner, DockerRunner.name: DockerRunner}


@functools.cache
def registered() -> dict[str, Any]:
    """The entry points installed packages register, by runner name; built-ins excluded. Read once per process:
    what is installed does not change while it runs, and clients ask on every render."""
    points = {point.name: point for point in entry_points(group=ENTRY_POINT_GROUP)}
    return {name: points[name] for name in sorted(points) if name not in BUILTIN}


def runner_names(configured: str = DEFAULT) -> list[str]:
    """Every runner a session can pick, in order: the built-ins, each one a package registers (OpenSandbox ships
    that way), then the configured one when it is none of those (a `module:Class`)."""
    names = [*BUILTIN, *registered()]
    return [*names, configured] if configured not in names else names


def load_runner(spec: str) -> Runner:
    """A built-in runner, one a package registers under `intern.runners`, or any `module:Class`."""
    found: Any = BUILTIN.get(spec)
    if found is None and ":" not in spec:
        point = registered().get(spec)
        if point is None:
            known = ", ".join([*BUILTIN, *registered()])
            raise ValueError(f"Unknown runner {spec!r}; use one of {known} or module:Class")
        try:
            found = point.load()
        except Exception as error:
            raise ValueError(f"Runner {spec!r} failed to load ({type(error).__name__})") from None
    elif found is None:
        module_name, _, class_name = spec.partition(":")
        try:
            found = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"Cannot load runner {spec!r} ({type(error).__name__})") from None
    if not (isinstance(found, type) and issubclass(found, Runner)):
        raise ValueError(f"{spec!r} is not a Runner subclass")
    return found()
