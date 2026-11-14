"""Where prompts and rules come from: the packaged system prompt, SYSTEM2.md, AGENTS.md and SYSTEM1.md files.
Memory is not an InstructionSource: it is recalled per task, see `intern.memory`."""

from __future__ import annotations

import platform
from datetime import date
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

CONTEXT_FILE = "AGENTS.md"
SYSTEM_FILE = "SYSTEM2.md"  # System 2's own prompt, in place of BASE_PROMPT
OLD_SYSTEM_FILE = "SYSTEM.md"  # its previous name, still read when there is no SYSTEM2.md
SYSTEM1_FILE = "SYSTEM1.md"  # the user's guidance for System 1, read from the config home only
# How System 1 is shown it, after the session cut.
SYSTEM1_LABEL = "The user's SYSTEM1.md (their guidance for System 1, from their config home, not from any project):\n"
# Of System 1's max_state_tokens: the file sits on top of the cut, in the quarter the cut leaves for the ask's own
# text. Taken out of the cut instead, it cost 1.3 points of the probes at 24,576 tokens (p=0.008), all of them where
# the session fills the budget; on top of it, none.
SYSTEM1_SHARE = 0.05

BASE_PROMPT = """\
You are an agent that gets the user's request done: with tools when it needs them, with a direct answer when it \
does not.

When the request needs tools:
- Act through tools; do not describe what you would do. Take one step, look at the result, then decide the next.
- Read a file before you edit it. Change files with edit; use write only for new files or full rewrites.
- Keep tool output small: locate what you need before you read it, and read long files in pages.
- When a tool fails, follow the "Next:" line in its result.
- Formats stated in the task or docs (names, keys, nesting, order) are a contract: match them exactly.
- Before finishing, check your real output against each requirement in the request and spec, not just your tests.
- When the task is done, reply with a short final answer and no tool call. If you cannot finish, say what blocks you."""


class PinnedDoc(BaseModel):
    source: str
    content: str


class Instructions(BaseModel):
    system: str
    pinned: list[PinnedDoc]
    # SYSTEM1.md: what System 1 reads after the session in every ask about the run. Never a project's file: a
    # repository anyone can write must not speak for the user to the guard (Claude Code's auto mode reads its allow
    # rules from user settings only, for the same reason).
    system1: str = ""


class InstructionSource(Protocol):
    def load(self, cwd: Path) -> Instructions: ...


def environment_line(cwd: Path) -> str:
    return f"Working directory: {cwd}. OS: {platform.system()}. Today: {date.today().isoformat()}."


def _project_files(cwd: Path) -> list[Path]:
    """AGENTS.md files from the repository root (or filesystem root) down to `cwd`."""
    found: list[Path] = []
    for folder in [cwd, *cwd.parents]:
        candidate = folder / CONTEXT_FILE
        if candidate.is_file():
            found.append(candidate)
        if (folder / ".git").exists():
            break
    return list(reversed(found))


class FileInstructions:
    """Reads SYSTEM2.md and SYSTEM1.md from the config home and AGENTS.md from the home and the project.

    Args:
        home: The config home; `SYSTEM2.md` there (or `SYSTEM.md`, its old name) replaces the base prompt.
        context_files: False skips every file, for isolated runs such as evals.
    """

    def __init__(self, home: Path, *, context_files: bool = True) -> None:
        self.home = home
        self.context_files = context_files

    def load(self, cwd: Path) -> Instructions:
        base = BASE_PROMPT
        pinned: list[PinnedDoc] = []
        system1 = ""
        if self.context_files:
            if (guidance := self.home / SYSTEM1_FILE).is_file():
                system1 = guidance.read_text().strip()
            override = next((p for p in (self.home / SYSTEM_FILE, self.home / OLD_SYSTEM_FILE) if p.is_file()), None)
            if override:
                base = override.read_text().strip()
            seen: set[Path] = set()
            for path in [self.home / CONTEXT_FILE, *_project_files(cwd)]:
                if path.is_file() and path.resolve() not in seen:
                    seen.add(path.resolve())
                    pinned.append(PinnedDoc(source=str(path), content=path.read_text().strip()))
        return Instructions(system=f"{base}\n\n{environment_line(cwd)}", pinned=pinned, system1=system1)
