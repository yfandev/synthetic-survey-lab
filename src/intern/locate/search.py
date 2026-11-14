"""`decision_search`: the question walks the workspace tree with System 1, one card per state.

    tree nodes ──noul──▶ files ──noul──▶ chunks ──noul──▶ leaf text ──noul──▶ top files side by side ──choice

Best-first down the tree, from the smallest node holding the scope (the root when there is none; the folders above
a scope are never judged): each child of an opened node is judged alone (its card is the whole state), the children
within the gap of the best are kept, the kept inner nodes join the frontier and the kept leaf groups have their
files judged, until the file or folder budget is spent. Kept nodes under MIN_PROB wait in reserve, entered only if
the frontier empties before any file is judged; the result carries the walk (`Walk`) so an empty answer can say how
far it got. With `lexical`, the question's words steer the walk too (`lexical.py`): the leaf groups holding the best
BM25 passages have their files judged first and their parents join the frontier; they are judged like any other.
The kept files have every chunk judged (16 a state), the best chunks have their text read, and the top files are
shown side by side for one choice asked in three orders. Every ask goes through `DecisionClient.ask`, which retries
a failing System 1 and stops the run if it stays down; nothing stands in for an answer (a keyword heuristic scored
below grep on every real-size tree). The state is the question and one card, never the
run's context. Given a trace sink,
the cascade reports every node it judges as it goes (`trace.py`), for a view to draw; the scores are the same.

The shape follows the NDI locate prototype; the questions do not: they ask what the agent's
question asks for, not for "implementation code that must change", and nothing here ranks tests last.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from intern.decisions import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionClient,
    NoulAnswer,
    NoulQuestion,
    Question,
)
from intern.locate.index import DERIVED_DIR, pages_of
from intern.locate.lexical import Hit, Lexical
from intern.locate.trace import Ranked, Tracer, TraceSink

if TYPE_CHECKING:
    from intern.locate.index import FileEntry, Index
    from intern.locate.processors import Chunk
    from intern.locate.tree import Node

# The survivor rule: options scoring at least best * GAP (and MIN_PROB), at least KEEP_MIN of them, at most a cap.
MIN_PROB = 0.05
GAP = 0.25
KEEP_MIN = 2
KEEP_CHILDREN = 4
KEEP_FILES = 8
#: Budgets of the descent: files judged, nodes opened, nodes opened concurrently a wave.
FILE_BUDGET = 120
FOLDER_BUDGET = 40
WAVE = 6
#: Lexical steering: the BM25 passages looked at, the leaf groups their files steer the walk to, the files those
#: groups may add beyond FILE_BUDGET, and the weight of a hit by its rank (1 / (SEED_RANK + rank)).
SEED_HITS = 30
SEED_GROUPS = 4
SEED_FILES = 40
SEED_RANK = 10
#: Test files: they hold a question's words because they exercise the code, not because it lives there (the reason
#: CODE's questions ask for where a subject lives), so their hits steer nowhere.
_TESTS = re.compile(
    r"(^|/)(tests?|__tests__)/|(^|/)(test_[^/]*|[^/]*_test\.py|conftest\.py|[^/]*\.(test|spec)\.[jt]sx?)$"
)
#: Chunks judged per file (in order) and per state; chunks whose text is read, per file and in all.
CHUNKS_MAX = 160
CHUNKS_PER_STATE = 16
LEAF_PER_FILE = 8
LEAF_MAX = 20
LEAF_LINES = 80
LEAF_CHARS = 4000
#: The side-by-side choice: files compared, read chunks shown per file, characters of each.
COMPARE_FILES = 5
COMPARE_CHUNKS = 2
COMPARE_CHARS = 1200
QUESTION_CHARS = 7000
QUESTION_HEAD = 4500
#: The files a trace's ranking names, as many as the tool shows.
TRACE_RANKED = 8
ROOT = "n:."
"""The trace's root: the tree's root node, or the files themselves when there is no tree."""


class Questions(BaseModel):
    """What System 1 is asked at each level, in the words of the extension that owns the index: a code tree asks
    for the place where the question's subject is defined or would be changed (where 'would someone need to open
    this file' let the tests that exercise a feature crowd out the implementation); an office workspace asks for
    what the task needs (its material), which the code wording ranks below a single document about the subject.
    ``chunk`` takes the card's id as ``{id}``."""

    node: str
    file: str
    chunk: str
    leaf: str
    compare: str


NONE = "none"


class Place(BaseModel):
    path: str
    key: str | None = None
    kind: str = "file"
    title: str = ""
    unit: Literal["line", "page"] = "line"
    start: int = 0
    end: int = 0
    table: str | None = None
    score: float

    @property
    def where(self) -> str:
        if self.key is None:
            return self.path
        if self.table:
            return f"{self.path} {self.title}"
        if self.unit == "page":
            span = f"page {self.start}" if self.start == self.end else f"pages {self.start}-{self.end}"
            return f"{self.path} {span}" + (f"  {self.title}" if self.kind != "page" else "")
        return f"{self.path}:{self.start}-{self.end}  {self.title}"

    @property
    def hint(self) -> str:
        """How to open this place with the workspace tools."""
        if self.table:
            return f'sql query="SELECT * FROM {self.table} LIMIT 20"'
        if self.key is None:
            return f"read {self.path}"
        if self.unit == "page":
            span = str(self.start) if self.start == self.end else f"{self.start}-{self.end}"
            return f"read {self.path} pages={span}"
        return f"read {self.path} offset={self.start} limit={self.end - self.start + 1}"


class Left(BaseModel):
    """The best-scored folder or group the walk judged but did not enter."""

    label: str
    score: float


class Walk(BaseModel):
    """How far the descent got. A result with nothing to show reports it, so a search that stopped at its first
    level reads differently from one that judged its files and found nothing."""

    start: str = Field(description="Where the walk started: the tree's root, or the smallest node holding the scope.")
    in_scope: int = Field(default=0, description="Indexed files the walk could reach.")
    opened: int = Field(default=0, description="Folders and groups whose children were judged.")
    depth: int = Field(default=0, description="Levels below the start that were judged.")
    files: int = Field(default=0, description="Files judged.")
    left: Left | None = None
    forced: bool = Field(
        default=False, description="No folder reached MIN_PROB before a file was judged; the best were entered anyway."
    )
    seeded: int = Field(default=0, description="Leaf groups entered because the question's words matched their files.")


class SearchResult(BaseModel):
    question: str
    files: list[tuple[str, float]] = []
    places: list[Place] = []
    walk: Walk | None = None


def survivors(scored: list[tuple[str, float]], gap: float, keep_max: int) -> list[tuple[str, float]]:
    order = sorted(scored, key=lambda x: -x[1])
    if not order:
        return []
    kept = [x for x in order if x[1] >= max(MIN_PROB, order[0][1] * gap)]
    if len(kept) < KEEP_MIN:
        kept = order[:KEEP_MIN]
    return kept[:keep_max]


def _trim(question: str) -> str:
    """A long question keeps its head and tail in every state."""
    if len(question) <= QUESTION_CHARS:
        return question
    return question[:QUESTION_HEAD] + "\n[... cut ...]\n" + question[-(QUESTION_CHARS - QUESTION_HEAD) :]


class Cascade:
    def __init__(
        self,
        index: Index,
        root: Path,
        question: str,
        decisions: DecisionClient,
        asks: Questions,
        trace: TraceSink | None = None,
        lexical: bool = False,
    ) -> None:
        self.index = index
        self.root = root
        self.lexical = lexical
        self.question = _trim(question)
        self.decisions = decisions
        self.asks = asks
        self._pages: dict[str, list[str]] = {}
        self.file_p: dict[str, float] = {}
        self._file_tasks: dict[str, asyncio.Task[float]] = {}
        self.node_p: dict[str, tuple[Node, float]] = {}
        """Every folder or group judged, by key, with its score."""
        self.entered: set[str] = set()
        """The keys of the nodes opened or whose files were judged."""
        self.depth: dict[str, int] = {}
        """Levels below the walk's start, by node key."""
        self.walk: Walk | None = None
        self.seeded: set[str] = set()
        """Files judged because the question's words steered there; on top of FILE_BUDGET, not out of it."""
        self.tracer = Tracer(trace, question, decisions)
        """Watches the search for a view; does nothing without a sink."""

    # -- asking -----------------------------------------------------------------------------------------------

    def state(self, what: str, body: str) -> str:
        return f"Question:\n{self.question}\n\n{what}:\n{body}"

    async def noul(self, purpose: str, what: str, body: str, question: str) -> float:
        """One judgment on one card: the card is the whole state."""
        answers = await self.decisions.ask(purpose, self.state(what, body), {"q": NoulQuestion(instructions=question)})
        self.tracer.asked(1)
        return _noul(answers["q"])

    async def nouls(self, purpose: str, what: str, cards: list[tuple[str, str]], template: str) -> dict[str, float]:
        """One noul per card, CHUNKS_PER_STATE cards a state, all states in flight at once."""

        async def one(batch: list[tuple[str, str]]) -> dict[str, float]:
            body = "\n\n".join(f"[{i}] {card}" for i, card in batch)
            questions: dict[str, Question] = {i: NoulQuestion(instructions=template.format(id=i)) for i, _ in batch}
            answers = await self.decisions.ask(purpose, self.state(what, body), questions)
            self.tracer.asked(len(batch))
            return {i: _noul(answers[i]) for i, _ in batch}

        batches = [cards[i : i + CHUNKS_PER_STATE] for i in range(0, len(cards), CHUNKS_PER_STATE)]
        out: dict[str, float] = {}
        for part in await asyncio.gather(*(one(b) for b in batches)):
            out |= part
        return out

    async def choice(self, purpose: str, what: str, body: str, options: dict[str, str]) -> dict[str, float]:
        """Probabilities over ``options`` and NONE."""
        criteria = options | {NONE: "none of these"}
        answers = await self.decisions.ask(
            purpose,
            self.state(what, body),
            {"q": ChoiceQuestion(instructions=self.asks.compare, criteria=criteria)},
        )
        answer = answers["q"]
        self.tracer.asked(1)
        if not isinstance(answer, ChoiceAnswer):
            return {}
        return dict(answer.probabilities) or {answer.choice: 1.0}

    # -- the tree ---------------------------------------------------------------------------------------------

    def score_file(self, path: str, parent: str = ROOT) -> asyncio.Task[float]:
        """One state per file, once: the descent never judges a file twice."""
        if path not in self._file_tasks:
            entry = self.index.files[path]
            fid = _fid(path)
            self.tracer.node(fid, parent=parent, kind="file", label=path, detail=_size(entry), state="asking")

            async def run() -> float:
                p = await self.noul("workspace.files", "Candidate file", entry.card, self.asks.file)
                self.file_p[path] = p
                self.tracer.node(fid, score=p, state="scored")
                return p

            self._file_tasks[path] = asyncio.ensure_future(run())
            self.tracer.files = len(self._file_tasks)
        return self._file_tasks[path]

    async def score_node(self, node: Node, parent: str = ROOT) -> float:
        nid = _nid(node)
        kind = "folder" if node.kind == "inner" else "group"
        self.tracer.node(nid, parent=parent, kind=kind, label=node.label, detail=_files(node), state="asking")
        p = await self.noul("workspace.groups", "Candidate folder or group of files", node.card, self.asks.node)
        self.node_p[node.key] = (node, p)
        self.tracer.node(nid, score=p, state="scored")
        return p

    def kids(self, node: Node, within: set[str]) -> list[Node]:
        """The children of ``node`` holding a file the walk may reach."""
        return [self.index.nodes[k] for k in node.children if any(f in within for f in self.index.nodes[k].files)]

    def start(self, within: set[str]) -> Node | None:
        """Where the walk starts: the smallest node holding every file in scope. The scope already says the answer is
        under it, so the folders above it are not judged: one low score on an ancestor's broad card would end the
        search before anything in scope was seen. Unscoped, it is the root (or its only child)."""
        node = self.index.nodes.get(".")
        while node is not None and node.kind == "inner":
            kids = self.kids(node, within)
            if len(kids) != 1:
                break
            node = kids[0]
        return node

    async def open_node(self, node: Node, within: set[str]) -> list[tuple[Node, float]]:
        kids = self.kids(node, within)
        self.entered.add(node.key)
        for kid in kids:
            self.depth[kid.key] = self.depth.get(node.key, 0) + 1
        self.tracer.node(_nid(node), state="opened")
        scores = await asyncio.gather(*(self.score_node(k, _nid(node)) for k in kids))
        kept = survivors([(str(i), p) for i, p in enumerate(scores)], GAP, KEEP_CHILDREN)
        if self.tracer.on:
            self.tracer.node(_nid(node), cut=max(MIN_PROB, max(scores, default=0.0) * GAP))
            keep = {int(i) for i, _ in kept}
            for n, kid in enumerate(kids):
                self.tracer.node(_nid(kid), state="kept" if n in keep else "pruned")
        return [(kids[int(i)], p) for i, p in kept]

    def seeds(self, start: Node, within: set[str], hits: list[Hit]) -> list[Node]:
        """The leaf groups under ``start`` the question's words point to, best first: each hit adds
        1 / (SEED_RANK + its rank) to the group holding its file."""
        leaf_of = {f: n for n in self.index.nodes.values() if n.kind == "leaf" for f in n.files}
        weight: dict[str, float] = {}
        for hit in [h for h in hits if not _TESTS.search(h.path)][:SEED_HITS]:
            leaf = leaf_of.get(hit.path)
            if hit.path in within and leaf is not None and leaf.key != start.key:
                weight[leaf.key] = weight.get(leaf.key, 0.0) + 1.0 / (SEED_RANK + hit.rank)
        best = sorted(weight, key=lambda k: (-weight[k], k))[:SEED_GROUPS]
        return [self.index.nodes[k] for k in best]

    async def descend(self, within: set[str], hits: list[Hit] | None = None) -> None:
        """Best-first from the start node: a wave of the most likely open nodes at a time; a kept leaf has its files
        judged, a kept inner node joins the frontier, until the budgets are spent. A kept node under MIN_PROB waits in
        reserve: if the frontier empties before any file is judged, the best KEEP_MIN of the reserve are entered, so
        a question no card names still gets files judged rather than an empty answer."""
        root = self.index.nodes.get(".")
        label = root.label if root is not None else "."
        self.tracer.node(ROOT, kind="root", label=label, detail=f"{len(within)} files", state="opened")
        start = self.start(within)
        self.walk = Walk(start=start.label if start is not None else label, in_scope=len(within))
        if start is None:
            for path in sorted(within):
                self.score_file(path)
            await self._settle([])
            return
        here = ROOT if start is root else _nid(start)
        if start is not root:  # the scope's node, entered without a judgment
            kind = "folder" if start.kind == "inner" else "group"
            self.tracer.node(here, parent=ROOT, kind=kind, label=start.label, detail=_files(start), state="opened")
        if start.kind == "leaf":
            self.entered.add(start.key)
            for path in start.files:
                if path in within:
                    self.score_file(path, here)
        frontier = [(start, 1.0)] if start.kind == "inner" else []
        reserve: list[tuple[Node, float]] = []
        if hits and start.kind == "inner":
            await self.seed(start, here, within, hits, frontier)

        def enter(node: Node, p: float) -> None:
            if node.kind == "inner":
                frontier.append((node, p))
                return
            files = [f for f in node.files if f in within]
            if not self.walked() or self.walked() + len(files) <= FILE_BUDGET:
                self.entered.add(node.key)
                self.tracer.node(_nid(node), state="opened")
                for f in files:
                    self.score_file(f, _nid(node))
            else:
                self.tracer.node(_nid(node), state="skipped")

        opened = 0
        while frontier and self.walked() < FILE_BUDGET and opened < FOLDER_BUDGET:
            frontier.sort(key=lambda x: -x[1])
            wave, frontier[:] = frontier[:WAVE], frontier[WAVE:]
            opened += len(wave)
            self.tracer.opened = opened
            kept = [kp for kids in await asyncio.gather(*(self.open_node(n, within) for n, _ in wave)) for kp in kids]
            for node, p in sorted(kept, key=lambda x: -x[1]):
                if p < MIN_PROB:
                    if p > 0:  # a flat 0 is a no from System 1, not a low score: not worth more asks
                        reserve.append((node, p))
                    self.tracer.node(_nid(node), state="pruned")
                    continue
                enter(node, p)
            if not frontier and not self._file_tasks and reserve:
                reserve.sort(key=lambda x: -x[1])
                self.walk.forced = True
                for node, p in reserve[:KEEP_MIN]:
                    self.tracer.node(_nid(node), state="kept")
                    enter(node, p)
                del reserve[:KEEP_MIN]
        self.walk.opened = opened
        await self._settle(frontier)

    async def seed(
        self, start: Node, here: str, within: set[str], hits: list[Hit], frontier: list[tuple[Node, float]]
    ) -> None:
        """Steer the walk by the question's words: the seed groups' files are judged now, up to SEED_FILES on top of
        the walk's own FILE_BUDGET (steering adds to the walk, it never takes from it), and each group's parent is
        judged like any folder and joins the frontier at its score, so the neighbours of a seed whose folder fits the
        question are walked, and those of one that does not wait behind the walk's own nodes. Nothing here ranks: a
        seeded file is judged like any other, and a seed whose cards say no is pruned like any other."""
        up = {k: n.key for n in self.index.nodes.values() for k in n.children}
        queued = {start.key}
        for leaf in self.seeds(start, within, hits):
            files = [f for f in leaf.files if f in within]
            if self.seeded and len(self.seeded) + len(files) > SEED_FILES:
                break
            self.seeded.update(files)
            assert self.walk is not None
            self.walk.seeded += 1
            self.entered.add(leaf.key)
            self.tracer.node(
                _nid(leaf),
                parent=here,
                kind="group",
                label=leaf.label,
                detail=f"{_files(leaf)} · words",
                state="opened",
            )
            for f in files:
                self.score_file(f, _nid(leaf))
            parent = up.get(leaf.key)
            if parent is not None and parent not in queued:
                queued.add(parent)
        parents = [self.index.nodes[k] for k in queued if k != start.key]
        for parent, p in zip(parents, await asyncio.gather(*(self.score_node(n, here) for n in parents)), strict=True):
            if p >= MIN_PROB:  # its place in the frontier is System 1's judgment of its card, like any node's
                frontier.append((parent, p))
            else:
                self.tracer.node(_nid(parent), state="pruned")

    def walked(self) -> int:
        """Files the walk itself sent for judging: what FILE_BUDGET counts."""
        return len(self._file_tasks) - len(self.seeded)

    async def _settle(self, frontier: list[tuple[Node, float]]) -> None:
        """The descent's end: nodes kept but never opened are marked, the judged files awaited, the walk counted."""
        for node, _ in frontier:  # kept, but a budget ran out before they were opened
            self.tracer.node(_nid(node), state="skipped")
        await asyncio.gather(*self._file_tasks.values())
        assert self.walk is not None
        self.walk.files = len(self._file_tasks)
        self.walk.depth = max((self.depth.get(k, 0) for k in self.node_p), default=0)
        left = [(n, p) for k, (n, p) in self.node_p.items() if k not in self.entered]
        if left:
            node, p = max(left, key=lambda x: x[1])
            self.walk.left = Left(label=node.label, score=round(p, 3))

    # -- inside a file ------------------------------------------------------------------------------------------

    async def score_chunks(self, path: str) -> list[tuple[Chunk, float]]:
        chunks = self.index.files[path].chunks[:CHUNKS_MAX]
        if not chunks:
            return []
        ids = {f"c{n}": c for n, c in enumerate(chunks)}
        if self.tracer.on:
            unit = self.index.files[path].unit
            self.tracer.node(_fid(path), state="opened")
            for c in chunks:
                detail = f"{c.kind} · {unit}s {c.start}-{c.end}"
                self.tracer.node(
                    _cid(path, c), parent=_fid(path), kind="part", label=c.title, detail=detail, state="asking"
                )
        scores = await self.nouls(
            "workspace.chunks", f"Parts of {path}", [(i, c.card) for i, c in ids.items()], self.asks.chunk
        )
        for i, p in scores.items():
            self.tracer.node(_cid(path, ids[i]), score=p, state="scored")
        return [(ids[i], p) for i, p in scores.items()]

    async def score_leaf(self, path: str, chunk: Chunk) -> float:
        self.tracer.node(_cid(path, chunk), state="reading")
        p = await self.noul(
            "workspace.leaf", f"Text from {path}, {chunk.title}", self.text(path, chunk), self.asks.leaf
        )
        self.tracer.node(_cid(path, chunk), read=p, state="read")
        return p

    def text(self, path: str, chunk: Chunk | None) -> str:
        entry = self.index.files[path]
        if chunk is None:
            return entry.card
        if entry.unit == "page":
            pages = self.pages(entry)
            return "\n\n".join(pages[chunk.start - 1 : chunk.end])[:LEAF_CHARS]
        source = self.index.home / DERIVED_DIR / entry.derived if entry.derived else self.root / path
        try:
            lines = source.read_text(errors="replace").splitlines()[chunk.start - 1 : chunk.end]
        except OSError:
            return chunk.card
        body = "\n".join(lines[:LEAF_LINES]) + ("\n... (truncated)" if len(lines) > LEAF_LINES else "")
        return body[:LEAF_CHARS]

    def pages(self, entry: FileEntry) -> list[str]:
        if entry.path not in self._pages:
            try:
                text = (self.index.home / DERIVED_DIR / (entry.derived or "")).read_text()
            except OSError:
                text = ""
            self._pages[entry.path] = pages_of(text)
        return self._pages[entry.path]

    # -- the whole -------------------------------------------------------------------------------------------

    async def run(self, scope: str | None) -> SearchResult:
        within = {p for p in self.index.files if _in_scope(p, scope)}
        if not within:
            return SearchResult(question=self.question, walk=Walk(start=scope or ".", in_scope=0))
        self.tracer.start(scope, FOLDER_BUDGET, FILE_BUDGET)
        try:
            return await self._run(within, scope)
        finally:
            await self.tracer.finish()

    async def chunks(
        self, within: set[str], scope: str | None
    ) -> tuple[list[tuple[str, Chunk, float]], dict[tuple[str, str], float], list[tuple[str, Chunk, float]]]:
        """The walk's recall: the descent, the kept files' chunks judged by their cards (path, chunk, card score)
        best first by their prior (the card's score weighted by its file's), and the chunks the leaf step reads."""
        hits: list[Hit] = []
        if self.lexical:
            words = Lexical(self.index, self.root)
            await asyncio.to_thread(words.update)
            hits = await asyncio.to_thread(words.hits, self.question, scope, SEED_HITS * 3)  # room for tests skipped
        await self.descend(within, hits)
        kept = [p for p, _ in survivors(list(self.file_p.items()), GAP, KEEP_FILES)]
        if self.tracer.on:
            self.tracer.file_cut = max(MIN_PROB, max(self.file_p.values(), default=0.0) * GAP)
            for path in self.file_p:
                self.tracer.node(_fid(path), state="kept" if path in kept else "pruned")
            self.tracer.enter("parts")

        # chunks of the kept files, judged by their cards; the best have their text read
        per_file = await asyncio.gather(*(self.score_chunks(p) for p in kept))
        cands = [(p, c, sp) for p, lst in zip(kept, per_file, strict=True) for c, sp in lst]
        prior = {(p, c.key): sp * (0.5 + 0.5 * self.file_p.get(p, 0.0)) for p, c, sp in cands}
        cands.sort(key=lambda x: -prior[(x[0], x[1].key)])
        floor = max(MIN_PROB, prior[(cands[0][0], cands[0][1].key)] * GAP) if cands else 1.0
        pool = [x for x in cands if prior[(x[0], x[1].key)] >= floor]
        leaf_set: list[tuple[str, Chunk, float]] = []
        per: dict[str, int] = {}
        for x in pool:
            if per.get(x[0], 0) < LEAF_PER_FILE and len(leaf_set) < LEAF_MAX:
                leaf_set.append(x)
                per[x[0]] = per.get(x[0], 0) + 1
        leaf_set += [x for x in pool if x not in leaf_set][: LEAF_MAX - len(leaf_set)]
        return cands, prior, leaf_set

    async def _run(self, within: set[str], scope: str | None) -> SearchResult:
        result = SearchResult(question=self.question)
        cands, _, leaf_set = await self.chunks(within, scope)
        result.walk = self.walk
        self.tracer.enter("read")
        leaf_p = await asyncio.gather(*(self.score_leaf(p, c) for p, c, _ in leaf_set))
        leaf = {(p, c.key): lp for (p, c, _), lp in zip(leaf_set, leaf_p, strict=True)}

        def score(p: str, c: Chunk, sp: float) -> float:
            fp = self.file_p.get(p, 0.0)
            lp = leaf.get((p, c.key))
            if lp is not None:
                return 0.6 * lp + 0.25 * sp + 0.15 * fp
            return 0.5 * (0.25 * sp + 0.15 * fp)  # text not read: half the prior part, below read text

        final = sorted(((p, c, score(p, c, sp)) for p, c, sp in cands), key=lambda x: -x[2])
        best_chunk: dict[str, float] = {}
        for p, _, sc in final:
            best_chunk[p] = max(best_chunk.get(p, 0.0), sc)
        fscore = {p: 0.5 * v + 0.5 * best_chunk.get(p, 0.0) for p, v in self.file_p.items()}
        files = sorted(fscore, key=lambda p: -fscore[p])
        if self.tracer.on:
            for p, c, sc in final:
                self.tracer.node(_cid(p, c), final=round(sc, 3))
            for p, v in fscore.items():
                self.tracer.node(_fid(p), final=round(v, 3))
        if len(files) >= 2:
            self.tracer.enter("compare")
            files = await self.compare(files, final, leaf, fscore)

        places = [_place(p, c, self.index.files[p], sc) for p, c, sc in final if sc > 0]
        places += [
            _place(p, None, self.index.files[p], fscore[p]) for p in files if p not in best_chunk and fscore[p] > 0
        ]
        result.files = [(p, round(fscore[p], 3)) for p in files]
        result.places = sorted(places, key=lambda x: -x.score)
        if self.tracer.on:
            self._ranked(result)
        return result

    def _ranked(self, result: SearchResult) -> None:
        """The trace's ending: the top files and their best places, chosen."""
        best: dict[str, Place] = {}
        for place in result.places:
            best.setdefault(place.path, place)
        ranked = [(p, s) for p, s in result.files if s > 0][:TRACE_RANKED]
        for path, score in ranked:
            self.tracer.node(_fid(path), final=score)
        for path, score in survivors(ranked, GAP, TRACE_RANKED):  # the files the ranking stands behind
            self.tracer.node(_fid(path), state="chosen", final=score)
            place = best.get(path)
            if place is not None and place.key is not None:
                self.tracer.node(f"c:{path}#{place.key}", state="chosen")
        self.tracer.ranking = [Ranked(path=p, score=s, where=_where(best[p]) if p in best else "") for p, s in ranked]

    async def compare(self, files: list[str], final: list, leaf: dict, fscore: dict[str, float]) -> list[str]:
        """The top files side by side, each with the text of its best read chunks: one choice, asked in three
        orders so no position is favoured; a file's score becomes half its own, half the mean probability."""
        top = files[:COMPARE_FILES]
        blocks: list[tuple[str, str]] = []
        for p in top:
            read = [c for q, c, _ in final if q == p and (p, c.key) in leaf][:COMPARE_CHUNKS]
            body = "\n\n".join(f"# {c.title}\n{self.text(p, c)[:COMPARE_CHARS]}" for c in read)
            blocks.append((p, body or self.index.files[p].card[:COMPARE_CHARS]))

        async def ask(order: list[int]) -> dict[str, float]:
            shown = [blocks[j] for j in order]
            body = "\n\n".join(f"[f{i}] {p}\n{text}" for i, (p, text) in enumerate(shown, 1))
            ids = {f"f{i}": p for i, (p, _) in enumerate(shown, 1)}
            probs = await self.choice("workspace.compare", "Candidate files, side by side", body, ids)
            by_path = {ids[i]: v for i, v in probs.items() if i in ids}
            self.tracer.vote([p for p, _ in shown], by_path)
            return by_path

        n = len(blocks)
        orders = [list(range(n)), list(range(n))[::-1], [(j + 2) % n for j in range(n)]]
        runs = [r for r in await asyncio.gather(*(ask(o) for o in orders)) if r]
        if not runs:
            return files
        for p in top:
            fscore[p] = 0.5 * fscore[p] + 0.5 * sum(r.get(p, 0.0) for r in runs) / len(runs)
            self.tracer.node(_fid(p), final=round(fscore[p], 3))
        return sorted(top, key=lambda p: -fscore[p]) + files[len(top) :]  # re-ranked inside the top only


async def search(
    index: Index,
    root: Path,
    question: str,
    decisions: DecisionClient,
    asks: Questions,
    *,
    scope: str | None = None,
    trace: TraceSink | None = None,
    lexical: bool = False,
) -> SearchResult:
    """The places for `question`; with `trace`, the search reported to it as it runs (the result is the same); with
    `lexical`, the walk is also steered to where the question's words are (`lexical.py`)."""
    return await Cascade(index, root, question, decisions, asks, trace, lexical).run(scope)


def _nid(node: Node) -> str:
    return f"n:{node.key}"


def _fid(path: str) -> str:
    return f"f:{path}"


def _cid(path: str, chunk: Chunk) -> str:
    return f"c:{path}#{chunk.key}"


def _files(node: Node) -> str:
    return f"{len(node.files)} file" + ("s" if len(node.files) != 1 else "")


def _size(entry: FileEntry) -> str:
    return f"{entry.units} {entry.unit}" + ("s" if entry.units != 1 else "")


def _where(place: Place) -> str:
    if place.key is None:
        return ""
    unit = "page" if place.unit == "page" else "lines"
    return f"{unit} {place.start}-{place.end}  {place.title}"


def _noul(answer: Answer | None) -> float:
    return answer.noul if isinstance(answer, NoulAnswer) else 0.0


def _in_scope(path: str, scope: str | None) -> bool:
    if not scope:
        return True
    scope = scope.strip().removeprefix("./").rstrip("/")
    return not scope or path == scope or path.startswith(scope + "/")


def _place(path: str, chunk: Chunk | None, entry: FileEntry, score: float) -> Place:
    if chunk is None:
        return Place(path=path, unit=entry.unit, score=round(score, 3))
    table = next((t.name for t in entry.tables if chunk.kind == "table" and t.name == chunk.key), None)
    return Place(
        path=path,
        key=chunk.key,
        kind=chunk.kind,
        title=chunk.title,
        unit=entry.unit,
        start=chunk.start,
        end=chunk.end,
        table=table,
        score=round(score, 3),
    )
