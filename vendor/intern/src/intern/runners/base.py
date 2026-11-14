"""The runner plugin interface: where a session's commands run.

A *runner* is chosen by name (`[capabilities.runner] name`, the ACP `runner` option, `/runner`). It validates its
own options when the config loads and, at session setup, builds the session's *executor*: the object that runs each
command (`intern.executor.Executor`). Nothing starts then; an executor that owns a service (a sandbox) starts
it on first use and gives an `aclose()` the session calls when it closes.

An isolated runner keeps files and commands away from the host. Setup then swaps the host file tools for ones that
go through the executor, and the executor's `mount` says where the tools see the workspace. The guard, the web
reader and the file tools use that mount; none of them names a runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from intern.executor import Executor


class NoOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class Runner(ABC):
    """One place commands can run. Subclass it, then register it under the `intern.runners` entry-point group
    or name it as `module:Class`."""

    name: ClassVar[str]
    title: ClassVar[str] = ""
    """What `/runner` and ACP clients show (`Docker`, `OpenSandbox`); empty is its name."""
    note: ClassVar[str] = ""
    """One line on where commands run with it, shown beside its name."""
    Options: ClassVar[type[BaseModel]] = NoOptions
    """Its `[capabilities.runner.options]`. Use `extra="forbid"` so a misspelt option fails at load."""
    jobs: ClassVar[bool] = False
    """Opt in: True exactly when its executor has `start_job`, so the `process` tool can run there. Config load
    checks this flag (it builds no executor); setup checks that the executor agrees."""
    isolated: ClassVar[bool] = True
    """True when files and commands live away from the host. The executor must then give a `mount`, and setup
    replaces the host file tools with ones that go through it."""

    def problem(self, options: BaseModel) -> str | None:
        """Why this runner cannot run here with `options` (a missing extra, an unset key), else None. Cheap and
        starts nothing: clients show it beside the runner's name, and setup fails with it."""
        return None

    @abstractmethod
    def create(self, options: BaseModel, *, workspace: Path, cwd: Path) -> Executor:
        """Build the session's executor without starting anything. `workspace` is the configured workspace, already
        absolute; `cwd` the session's working directory, which the workspace must contain. Raise ValueError with a
        message free of secrets and option values when they do not fit."""
