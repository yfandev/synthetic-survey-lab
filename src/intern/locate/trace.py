"""A search as it runs, for whoever draws it: every node System 1 judged (a folder, a group of files, a file, a part
of a file, the text of a part), its score, what the survivor rule did with it, and the phase the search is in.

`Cascade` reports to a `Tracer`; the tracer gathers the changes and sends them as `SearchTrace`s a few times a
second, at each phase and at the end, so a search of hundreds of asks is tens of traces, each with only the
nodes that changed (an upsert by `id`). The code-search extension sends them as `ExtensionEvent`s of kind
`search_trace` and its TUI view draws them (`extensions/code_search/view.py`). The tracer only watches: the
scores and the ranking are the same with it or without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from intern.decisions import DecisionClient

KIND = "search_trace"
"""The `kind` of the extension events that carry a trace."""
FLUSH_S = 0.1

Phase = Literal["descend", "parts", "read", "compare", "done"]
PHASES: tuple[Phase, ...] = ("descend", "parts", "read", "compare", "done")
NodeKind = Literal["root", "folder", "group", "file", "part"]
NodeState = Literal["asking", "scored", "kept", "pruned", "opened", "skipped", "reading", "read", "chosen"]


class SearchNode(BaseModel):
    """One node of the search. Its parent is always sent before it, or with it."""

    id: str
    parent: str | None = None
    kind: NodeKind
    label: str
    detail: str = Field(default="", description="`106 files`, `lines 40-80`, `function`.")
    state: NodeState = Field(
        description="`asking`: its card is with System 1; `scored`: answered; `kept` or `pruned`: by the survivor "
        "rule among its siblings; `opened`: its children are judged; `skipped`: kept, but a budget ran out "
        "first; `reading`/`read`: a part's text is with System 1 / was judged; `chosen`: in the result."
    )
    score: float | None = Field(default=None, description="System 1's probability for its card.")
    read: float | None = Field(default=None, description="A part whose text was read: the text's probability.")
    final: float | None = Field(default=None, description="A file's or part's score in the ranking.")
    cut: float | None = Field(
        default=None, description="An opened node: the survivor rule's bar for its children (best x gap)."
    )


class Vote(BaseModel):
    """One of the side-by-side choices over the top files: the order they were shown in, the probabilities."""

    order: list[str]
    probabilities: dict[str, float]


class Ranked(BaseModel):
    path: str
    score: float
    where: str = Field(default="", description="Its best place: a symbol or section and its lines.")


class SearchTrace(BaseModel):
    search: str = Field(description="One id per search: searches in one turn run side by side.")
    question: str = Field(description="The question as the tool was given it.")
    scope: str | None = None
    phase: Phase
    nodes: list[SearchNode] = Field(default_factory=list, description="Changed since the last trace.")
    asks: int = Field(default=0, description="States this search sent to System 1.")
    judged: int = Field(default=0, description="Cards and texts judged in them.")
    cost_usd: float = 0.0
    elapsed_ms: int = 0
    opened: int = 0
    opened_budget: int = 0
    files: int = Field(default=0, description="Files judged.")
    files_budget: int = 0
    file_cut: float | None = Field(default=None, description="The survivor rule's bar over all judged files.")
    votes: list[Vote] = Field(default_factory=list)
    ranking: list[Ranked] = Field(default_factory=list)
    finished: bool = False


TraceSink = Callable[[SearchTrace], None]


class Tracer:
    """Collects one search's changes and sends them, every FLUSH_S while it runs. With no sink it does nothing."""

    def __init__(self, sink: TraceSink | None, question: str, decisions: DecisionClient | None = None) -> None:
        self.sink = sink
        self.search = uuid.uuid4().hex[:8]
        self.question = question
        self.scope: str | None = None
        self.phase: Phase = "descend"
        self.nodes: dict[str, SearchNode] = {}
        self._changed: dict[str, None] = {}  # insertion-ordered: parents before children
        self.asks = self.judged = 0
        self.opened = self.opened_budget = self.files = self.files_budget = 0
        self.file_cut: float | None = None
        self.votes: list[Vote] = []
        self.ranking: list[Ranked] = []
        self._sent_votes = 0
        self._decisions = decisions
        self._spent = decisions.usage.cost_usd if decisions is not None else 0.0
        self._started = time.monotonic()
        self._ticker: asyncio.Task[None] | None = None

    @property
    def on(self) -> bool:
        return self.sink is not None

    def start(self, scope: str | None, opened_budget: int, files_budget: int) -> None:
        if self.sink is None:
            return
        self.scope, self.opened_budget, self.files_budget = scope, opened_budget, files_budget
        self._ticker = asyncio.ensure_future(self._tick())

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(FLUSH_S)
            if self._changed:
                self.flush()

    def node(self, id: str, **fields: object) -> None:
        """Create or change a node; a new one needs `kind`, `label` and `state` (and `parent`)."""
        if self.sink is None:
            return
        node = self.nodes.get(id)
        if node is None:
            try:
                node = SearchNode.model_validate({"id": id, **fields})
            except ValidationError:
                return  # a change to a node never reported: the trace goes without it, the search on
        else:
            node = node.model_copy(update=fields)
        self.nodes[id] = node
        self._changed[id] = None

    def asked(self, judged: int) -> None:
        self.asks += 1
        self.judged += judged

    def enter(self, phase: Phase) -> None:
        if self.sink is None or phase == self.phase:
            return
        self.flush()
        self.phase = phase
        self.flush()

    def vote(self, order: list[str], probabilities: dict[str, float]) -> None:
        if self.sink is not None:
            self.votes.append(Vote(order=order, probabilities=probabilities))

    def flush(self, finished: bool = False) -> None:
        if self.sink is None:
            return
        trace = self.snapshot(finished)
        try:
            self.sink(trace)
        except Exception:
            self.sink = None  # whoever draws the search cannot break it: no more traces, the search goes on

    def snapshot(self, finished: bool) -> SearchTrace:
        """The trace of what changed since the last one; a search that reports more (`UnionTracer`) adds to it."""
        changed = [self.nodes[i] for i in self._changed]
        self._changed.clear()
        votes, self._sent_votes = self.votes[self._sent_votes :], len(self.votes)
        cost = (self._decisions.usage.cost_usd - self._spent) if self._decisions is not None else 0.0
        return SearchTrace(
            search=self.search,
            question=self.question,
            scope=self.scope,
            phase=self.phase,
            nodes=changed,
            asks=self.asks,
            judged=self.judged,
            cost_usd=round(cost, 6),
            elapsed_ms=round((time.monotonic() - self._started) * 1000),
            opened=self.opened,
            opened_budget=self.opened_budget,
            files=self.files,
            files_budget=self.files_budget,
            file_cut=self.file_cut,
            votes=votes,
            ranking=self.ranking if finished else [],
            finished=finished,
        )

    async def finish(self) -> None:
        """The last trace, with the ranking; the ticker stops."""
        if self.sink is None:
            return
        if self._ticker is not None:
            self._ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ticker
        self.phase = "done"
        self.flush(finished=True)
