"""`decision_files`' search as it runs, for the workspace's view (`view.py`): the union's trace (`search.py`).

A union trace is a search trace (`locate/trace.py`) and more: the walk reports its folders, files and parts as it
does for `decision_code` (the same `SearchNode`s), and the union adds its own stages, the passages it judges (each with
its lane: the walk, the BM25 top 100 or an expansion term), the seeds, the terms with their weights and the seed
each came from, and the final order. `UnionTracer` batches it as `Tracer` does (an upsert of what changed, a few
times a second and at each stage), and the extension sends it as `search_trace` events, so whatever reads a search
trace reads this one. The tracer only watches: the ranking, the scores and the asks are the same without it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, Field, ValidationError

from intern.locate.trace import SearchTrace, Tracer, TraceSink
from intern.tools.images import IMAGE_SUFFIXES

if TYPE_CHECKING:
    from intern.decisions import DecisionClient
    from intern.extensions.workspace.search import Passage
    from intern.locate.index import Index

Stage = Literal["walk", "judge", "expand", "rejudge", "done"]
STAGES: tuple[Stage, ...] = ("walk", "judge", "expand", "rejudge", "done")
Lane = Literal["walk", "bm25", "term"]
HitState = Literal["candidate", "asking", "judged", "card"]


class Hit(BaseModel):
    """A passage the union ranks: a page, a section, a tab, a table, or a file without parts (an image)."""

    id: str = Field(description="`path#key`.")
    path: str
    key: str = ""
    title: str = Field(default="", description="The part's own: `page 4`, `tab Sales`, `Results`.")
    kind: str = Field(default="file", description="`page`, `pages`, `section`, `table`, `image`, `file`, ...")
    lanes: list[Lane] = Field(default_factory=list, description="What brought it: the walk, BM25, a term.")
    bm25: int | None = Field(default=None, description="Its place in the BM25 top 100, from 1.")
    prior: float | None = Field(default=None, description="The walk's prior: its card's score, by its file's.")
    term: str | None = Field(default=None, description="The expansion term that reached it.")
    state: HitState = Field(
        description="`candidate`: waiting; `asking`: with System 1; `judged`: its text scored as evidence; `card`: "
        "a walk chunk not read, ranked by its card."
    )
    score: float | None = None
    seed: bool = False
    rank: int | None = Field(default=None, description="Its place in the top 10, in diversity order, from 1.")
    scored: int | None = Field(default=None, description="Its place in the top 10 by score alone, from 1.")


class Term(BaseModel):
    term: str
    weight: float
    seed: str = Field(description="The hit id of the seed it came from (the one that weighed it most).")
    hits: list[str] = Field(default_factory=list, description="The new passages it reached first.")


class UnionTrace(SearchTrace):
    seq: int = Field(default=0, description="One more each trace, so a view can tell a stale or repeated one.")
    stage: Stage = "walk"
    hits: list[Hit] = Field(default_factory=list, description="Changed since the last trace.")
    terms: list[Term] = Field(default_factory=list, description="All of them, once they are chosen.")
    passages: int = Field(default=0, description="Passages in the BM25 table.")
    unread: int = Field(default=0, description="Walk chunks ranked below by their cards.")


UnionSink = Callable[[UnionTrace], None]


class UnionTracer(Tracer):
    """The walk's tracer, given to its `Cascade`, with the union's stages and passages on top."""

    def __init__(self, sink: UnionSink | None, question: str, decisions: DecisionClient | None = None) -> None:
        super().__init__(cast("TraceSink | None", sink), question, decisions)  # it only sends `UnionTrace`s
        self.stage: Stage = "walk"
        self.hits: dict[str, Hit] = {}
        self._changed_hits: dict[str, None] = {}
        self.terms: list[Term] = []
        self._terms_sent = 0
        self.passages = self.unread = 0
        self.seq = 0

    def begin(self, stage: Stage) -> None:
        if self.sink is None or stage == self.stage:
            return
        self.flush()
        self.stage = stage
        self.flush()

    def passage(self, index: Index, p: Passage, lane: Lane | None = None, **fields: object) -> None:
        """Create or change the hit for `p`; `lane` adds to the lanes that brought it."""
        if self.sink is None:
            return
        id = f"{p.path}#{p.key}"
        hit = self.hits.get(id)
        if hit is None:
            chunk = index.chunk(p.path, p.key) if p.key else None
            image = PurePosixPath(p.path).suffix.lower() in IMAGE_SUFFIXES
            try:
                hit = Hit.model_validate(
                    {
                        "id": id,
                        "path": p.path,
                        "key": p.key,
                        "title": chunk.title if chunk is not None else "",
                        "kind": chunk.kind if chunk is not None else "image" if image else "file",
                        "state": "candidate",
                        **fields,
                    }
                )
            except ValidationError:
                return
        else:
            hit = hit.model_copy(update=fields)
        if lane is not None and lane not in hit.lanes:
            hit = hit.model_copy(update={"lanes": [*hit.lanes, lane]})
        self.hits[id] = hit
        self._changed_hits[id] = None

    def snapshot(self, finished: bool) -> UnionTrace:
        base = super().snapshot(finished)
        self.seq += 1
        changed = [self.hits[i] for i in self._changed_hits]
        self._changed_hits.clear()
        terms = self.terms if len(self.terms) != self._terms_sent else []
        self._terms_sent = len(self.terms)
        return UnionTrace(
            **dict(base),
            seq=self.seq,
            stage=self.stage,
            hits=changed,
            terms=terms,
            passages=self.passages,
            unread=self.unread,
        )
