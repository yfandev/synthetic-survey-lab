"""Append-only JSONL tree of everything a session did.

Every entry points at its parent, so a session is a tree: appending after an older entry starts a
branch, and the active path is always root to leaf. The context the model sees is rebuilt from that
path on every turn; nothing in the journal is ever rewritten.
"""

from __future__ import annotations

import fcntl
import os
import uuid
import warnings
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from intern.events import (
    Compaction,
    Decision,
    Event,
    EventSink,
    Nudge,
    ProcessCheck,
    ProcessEvent,
    State,
    ToolGuard,
    now_ms,
)
from intern.messages import Message

JOURNAL_VERSION = 1


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class _EntryBase(BaseModel):
    id: str = Field(default_factory=new_id)
    parent_id: str | None = None
    timestamp: int = Field(default_factory=now_ms)


class SessionEntry(_EntryBase):
    kind: Literal["session"] = "session"
    version: int = JOURNAL_VERSION
    cwd: str


class SystemPromptEntry(_EntryBase):
    """The system prompt and tool list in force from this point; a change is a new entry, not an edit."""

    kind: Literal["system_prompt"] = "system_prompt"
    content: str
    tools: list[str]


class MessageEntry(_EntryBase):
    kind: Literal["message"] = "message"
    message: Message
    run_id: str | None = None


class ObservationEntry(_EntryBase):
    """The full output of a tool call whose result the model saw only in bounded form."""

    kind: Literal["observation"] = "observation"
    call_id: str
    content: str


class StateEntry(_EntryBase):
    """The State System 2 wrote for a compaction, over the turns the cut drops."""

    kind: Literal["state"] = "state"
    text: str
    covers_through_id: str = Field(description="Last history entry the writer saw; later ones are not in it.")
    tokens_before: int = Field(description="Tokens of history the previous State did not cover.")


class Kept(BaseModel):
    """A file read kept verbatim through a compaction because the current step works on it."""

    path: str
    content: str


class CompactionEntry(_EntryBase):
    kind: Literal["compaction"] = "compaction"
    summary: str
    first_kept_id: str | None = Field(description="Oldest history entry kept verbatim; None keeps nothing.")
    tokens_before: int
    method: Literal["summary", "drop", "state"] = Field(
        description="state: cut to the State (written first when it does not cover the cut); drop: a listing; "
        "summary: older journals."
    )
    kept: list[Kept] = Field(default_factory=list, description="Reads shown after the State until the next compaction.")


class ModelChangeEntry(_EntryBase):
    kind: Literal["model_change"] = "model_change"
    role: str
    model: str


class WindowCapEntry(_EntryBase):
    """The session's own cap on System 2's working window; None goes back to the process default (`--window`)."""

    kind: Literal["window_cap"] = "window_cap"
    cap: int | None


class CapabilitiesChangeEntry(_EntryBase):
    """The capability toggles a client set, every one of them, so a resumed session opens on them again."""

    kind: Literal["capabilities_change"] = "capabilities_change"
    toggles: dict[str, bool]
    runner: str | None = None
    """The runner the session runs on (`[capabilities.runner] name`); None in entries from before runners."""


class MemoryRecallEntry(_EntryBase):
    kind: Literal["memory_recall"] = "memory_recall"
    drive: str
    commit: str
    pages: list[str]
    paths: list[str]
    text: str = Field(description="The fenced memory block rendered into pinned; empty when nothing was recalled.")
    backend: str
    steps: int
    cost_usd: float = Field(default=0.0, description="System 1's cost of the recall; its decisions are not journaled.")


RecordedEvent = Annotated[
    Decision | ToolGuard | Nudge | State | Compaction | ProcessEvent | ProcessCheck, Field(discriminator="type")
]


class EventEntry(_EntryBase):
    """An event of a run that the conversation does not otherwise keep, where it happened: System 1's decisions
    for a turn (the supervisor's, the guard's, a compaction's choice of reads to keep), the guard's refusals, the
    engine's nudges, every State write and compaction with its turn, its trigger and how it went, and every command
    that started or ended (`ProcessEvent`). Kept so a
    resumed session can show what System 1 judged and what the loop did on it; never read into the context. A
    State's text and a compaction's summary are their own entries'; the copies here are empty."""

    kind: Literal["event"] = "event"
    event: RecordedEvent


Entry = Annotated[
    SessionEntry
    | SystemPromptEntry
    | MessageEntry
    | ObservationEntry
    | StateEntry
    | CompactionEntry
    | ModelChangeEntry
    | WindowCapEntry
    | CapabilitiesChangeEntry
    | MemoryRecallEntry
    | EventEntry,
    Field(discriminator="kind"),
]
_entry_adapter: TypeAdapter[Entry] = TypeAdapter(Entry)


class Journal:
    """An append-only entry tree, kept in memory and mirrored to a JSONL file when it has a path."""

    def __init__(self, cwd: str, path: Path | None = None) -> None:
        self.path = path
        self._entries: dict[str, Entry] = {}
        self._order: list[str] = []
        self.leaf_id: str | None = None
        self._disk_size = 0
        self._needs_separator = False
        self._write_failed = False
        if path is not None and path.exists() and path.stat().st_size > 0:
            self._load(path)
        if not self._order:
            self.append(SessionEntry(cwd=cwd))

    @property
    def session_id(self) -> str:
        return self._order[0]

    def _load(self, path: Path) -> None:
        # Coordinate recovery with appends: never mistake another writer's in-progress line for a tear.
        with path.open("r+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            data = stream.read()
            lines = data.splitlines(keepends=True)
            offset = 0
            for index, line in enumerate(lines):
                if line.strip():
                    try:
                        entry = _entry_adapter.validate_json(line)
                    except ValidationError as error:
                        # Only an unterminated final JSON fragment is recoverable. A complete invalid
                        # record or interior corruption must stay visible as an error.
                        torn = (
                            index == len(lines) - 1
                            and not line.endswith((b"\n", b"\r"))
                            and all(
                                item["type"] == "json_invalid"
                                and "EOF while parsing" in str(item.get("ctx", {}).get("error", ""))
                                for item in error.errors()
                            )
                        )
                        if not torn:
                            raise
                        backup = path.with_name(f"{path.name}.torn-{uuid.uuid4().hex}")
                        with backup.open("xb") as saved:
                            os.fchmod(saved.fileno(), 0o600)
                            saved.write(line)
                            saved.flush()
                            os.fsync(saved.fileno())
                        stream.seek(offset)
                        stream.truncate()
                        stream.flush()
                        os.fsync(stream.fileno())
                        data = data[:offset]
                        warnings.warn(
                            f"Recovered incomplete journal tail; original fragment saved to {backup}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        break
                    self._entries[entry.id] = entry
                    self._order.append(entry.id)
                offset += len(line)
            self._disk_size = len(data)
            self._needs_separator = bool(data) and not data.endswith(b"\n")
        self.leaf_id = self._order[-1] if self._order else None

    def append[E: _EntryBase](self, entry: E) -> E:
        """Attach `entry` under the current leaf and make it the new leaf."""
        entry.parent_id = self.leaf_id
        if self.path is not None:
            if self._write_failed:
                raise OSError("A prior journal write failed; reopen the session before continuing.")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = (("\n" if self._needs_separator else "") + entry.model_dump_json() + "\n").encode("utf-8")
            try:
                with self.path.open("a+b") as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    stream.seek(0, os.SEEK_END)
                    if stream.tell() != self._disk_size:
                        raise OSError("The journal changed in another writer; reopen the session before continuing.")
                    stream.write(row)
                    stream.flush()
                    os.fsync(stream.fileno())
                    self._disk_size = stream.tell()
            except OSError:
                self._write_failed = True
                raise
            self._needs_separator = False
        # A published in-memory entry must have made it through persistence first.
        self._entries[entry.id] = entry  # pyright: ignore[reportArgumentType]
        self._order.append(entry.id)
        self.leaf_id = entry.id
        return entry

    def last_models(self) -> dict[str, str]:
        """The models the active branch last switched to, by role, so a resumed session continues on them.
        Replies are not read: a reply names the model that wrote it, which may be a failover backup."""
        return {e.role: e.model for e in self.active_path() if isinstance(e, ModelChangeEntry)}

    def last_capabilities(self) -> dict[str, bool] | None:
        """The capability toggles the active branch last set; None when it never set any."""
        entry = next((e for e in reversed(self.active_path()) if isinstance(e, CapabilitiesChangeEntry)), None)
        return dict(entry.toggles) if entry is not None else None

    def last_runner(self) -> str | None:
        """The runner the active branch last ran on; None when it never recorded one."""
        entry = next((e for e in reversed(self.active_path()) if isinstance(e, CapabilitiesChangeEntry)), None)
        return entry.runner if entry is not None else None

    def branch(self, entry_id: str) -> None:
        """Move the leaf to `entry_id`; the next append starts a new branch there."""
        if entry_id not in self._entries:
            raise KeyError(f"no journal entry {entry_id!r}")
        self.leaf_id = entry_id

    def get(self, entry_id: str) -> Entry:
        return self._entries[entry_id]

    def entries(self) -> list[Entry]:
        """Every entry in append order, all branches included."""
        return [self._entries[i] for i in self._order]

    def active_path(self) -> list[Entry]:
        """Entries from the root to the current leaf."""
        return self.path_to(self.leaf_id)

    def path_to(self, entry_id: str | None) -> list[Entry]:
        """Entries from the root to `entry_id`."""
        if entry_id is not None and entry_id not in self._entries:
            raise KeyError(f"no journal entry {entry_id!r}")
        path: list[Entry] = []
        cursor = entry_id
        while cursor is not None:
            entry = self._entries[cursor]
            path.append(entry)
            cursor = entry.parent_id
        path.reverse()
        return path

    def observation(self, call_id: str) -> str | None:
        for entry in reversed(self.active_path()):
            if isinstance(entry, ObservationEntry) and entry.call_id == call_id:
                return entry.content
        return None


def recording(journal: Journal, emit: EventSink) -> EventSink:
    """`emit`, with the events an `EventEntry` keeps journaled first. A decision is kept when it was asked for a
    turn of the run; a tool's own asks (code search's) are part of its result."""

    def sink(event: Event) -> None:
        if isinstance(event, ToolGuard | Nudge | ProcessEvent | ProcessCheck) or (
            isinstance(event, Decision) and event.turn is not None
        ):
            journal.append(EventEntry(event=event))
        elif isinstance(event, State):
            journal.append(EventEntry(event=event.model_copy(update={"text": ""})))
        elif isinstance(event, Compaction):
            journal.append(EventEntry(event=event.model_copy(update={"summary": ""})))
        emit(event)

    return sink
