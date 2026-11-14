"""What an extension needs to put `decision_search` on a directory: the index kept fresh (`Store`), the tools, and
the hooks (`Locate`, an Extension base). An extension chooses the processors that build its index, the questions
System 1 is asked, the home the index lives in (`.intern/` or a folder under it), whether the index is built on its
own at the start of a run (code: seconds, so yes) or when the agent asks for it with the index tool (documents:
OCR takes minutes and the agent chooses the folder), the tools it adds and the note the agent reads.

An index built at the start of a run is refreshed after any tool that may have changed files; one the agent built
is refreshed the same way. The search needs System 1: without a backend nothing is built, no note is pinned, and
`decision_search` says what to configure (a keyword stand-in scored below grep on every real-size tree). A build on a
terminal shows a progress bar on stderr. Every request carries a pinned note saying what the tools are for and
the root node's card (a map of the directory), or no note where the working directory holds nothing of the
extension's kind (`Locate.pin`), fixed at the start of the run so the prompt prefix stays cacheable,
placed before the task so the task's requirements remain the last thing the model reads. The note leaves the
choice of tool to the agent: told to search first, Qwen3.8 did so and reached the right place by its
second call instead of its sixth, then spent the saving on reading and iterating (+21% input tokens, same pass
rate); left to judge, it searched on office tasks and grepped on code, at no cost.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from intern import hooks as h
from intern import names
from intern.context import ContextView
from intern.decisions import DecisionClient
from intern.events import EventSink, ExtensionEvent, Indexing
from intern.locate.index import TooManyFiles
from intern.messages import ToolCall, ToolResult
from intern.tools.base import Effect, Tool, ToolContext, ToolOutput, fail

if TYPE_CHECKING:
    from intern.locate.index import Index, Processors
    from intern.locate.search import Place, Questions, SearchResult, Walk
    from intern.locate.trace import SearchTrace

#: Tools after which files may differ from the index.
CHANGING_TOOLS = frozenset({"write", "edit", "bash"})
PLACES_SHOWN = 10
FILES_SHOWN = 8
#: What a place's hint opens: the best place with READ_MARGIN lines above and below (the class line and imports
#: a fix needs; a symbol-tight slice had the agent paging up for them), at most READ_LIMIT lines; a file of up to
#: WHOLE_FILE lines whole. The file's other places are named on the entry, not opened.
WHOLE_FILE = 60
READ_MARGIN = 15
READ_LIMIT = 200
MORE_SHOWN = 3


class NoIndex(Exception):
    """No index at the store's root, and the store does not build one on its own."""


def moved_home(root: Path, folder: str, moved_from: str | None) -> Path:
    """``root / folder``, after moving an index kept under the folder's earlier name there."""
    home = root / folder
    if moved_from and not home.exists() and (old := root / moved_from).is_dir():
        old.rename(home)
    return home


class Store:
    """The indexes of the directories searched in a run, each kept fresh, with the processors that build them and
    the home each lives in (``folder`` under its root). ``auto``: build when missing, else only when `build` is
    called (the index tool)."""

    def __init__(
        self,
        processors: Processors,
        name: str,
        install: str,
        folder: str,
        auto: bool = True,
        moved_from: str | None = None,
    ) -> None:
        self.processors = processors
        self.name = name
        self.install = install
        self.folder = folder
        self.moved_from = moved_from
        """The home's earlier name: an index found there is moved to ``folder`` the first time its root is opened."""
        self.auto = auto
        self.root: Path | None = None
        """The root searched when the agent names no other: the working directory until the index tool builds one."""
        self.indexes: dict[Path, Index] = {}
        self.fresh: set[Path] = set()
        """Roots whose index was checked against their files since the last tool that may have changed files."""
        self.too_many: set[Path] = set()
        """Roots past MAX_FILES: not walked again (a home folder's walk takes seconds, before every run)."""
        self.decisions: DecisionClient | None = None
        self.emit: EventSink | None = None
        """The run's event sink: a build reports its progress there as `indexing` events (the stderr bar is for
        runs without one)."""
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def home(self, root: Path) -> Path:
        return moved_home(root, self.folder, self.moved_from)

    def mark_dirty(self) -> None:
        self.fresh.clear()

    async def ready(self, cwd: Path, root: Path | None = None) -> Index:
        """The index of ``root``, by default the store's root (``cwd`` until an index tool or a build names another),
        refreshed first when files changed. Raises ImportError without the extra, NoIndex when there is none and the
        store does not build on its own."""
        async with self._lock:
            self._loop = asyncio.get_running_loop()
            return await asyncio.to_thread(self._ensure, root or self.root or cwd, False)

    async def build(self, root: Path) -> Index:
        """Build or refresh the index of ``root``, which becomes the store's root."""
        async with self._lock:
            self._loop = asyncio.get_running_loop()
            return await asyncio.to_thread(self._ensure, root, True)

    def _progress(self, root: Path) -> tuple[Callable[[int, int], None], Callable[[bool], None]]:
        """How a build reports: `indexing` events on the run's loop when there is a sink (a few a second, and one
        with `finished` when it built), else the bar on stderr. Returns the callback and what to call when the build
        ends, with whether it built."""
        from intern.locate import index as wi

        emit, loop = self.emit, self._loop
        if emit is None or loop is None:
            bar = wi.Progress(f"{self.name}: indexing {root.name}")
            return bar, lambda built: bar.close()

        started = time.perf_counter()
        last = [0.0, 0, 0]  # when last sent, done, total

        def send(done: int, total: int, finished: bool) -> None:
            event = Indexing(
                name=self.name,
                root=str(root),
                done=done,
                total=total,
                finished=finished,
                duration_ms=round((time.perf_counter() - started) * 1000) if finished else None,
            )
            loop.call_soon_threadsafe(emit, event)

        def progress(done: int, total: int) -> None:
            now = time.perf_counter()
            last[1], last[2] = done, total
            if now - last[0] >= wi.REDRAW_S or done == 1:
                last[0] = now
                send(done, total, False)

        def ended(built: bool) -> None:
            if built:  # a failed build is not reported as one that finished (with 0 files, when it never began)
                send(last[1], last[2], True)

        return progress, ended

    def _ensure(self, root: Path, build: bool) -> Index:
        from intern.locate import index as wi

        root = root.resolve()
        if root in self.too_many:
            raise TooManyFiles(root)
        index = self.indexes.get(root)
        if index is None:
            index = wi.load(self.home(root))
            self.fresh.discard(root)
        if index is None and not (build or self.auto):
            raise NoIndex(root)
        if root not in self.fresh:
            if index is None or wi.stale(root, index, self.processors):
                report = lambda line: print(f"{self.name}: {line}", file=sys.stderr)  # noqa: E731
                progress, ended = self._progress(root)
                built = False
                try:
                    index = wi.build(
                        root, self.home(root), processors=self.processors, report=report, progress=progress
                    )
                    built = True
                except TooManyFiles:
                    self.too_many.add(root)
                    raise
                finally:
                    ended(built)
            self.fresh.add(root)
        assert index is not None
        self.indexes[root] = index
        if build:
            self.root = root
        return index

    def place(self, scope: str | None, cwd: Path) -> tuple[Path | None, str | None]:
        """The root to search and the scope within it, for a ``scope`` the agent gave relative to ``cwd`` or absolute.
        Under the store's root it is a scope there (root None: the default); elsewhere, another repository the agent
        was asked about, the root is the git work tree holding it (else the folder itself), indexed on first use and
        kept in its own home like any index. Raises FileNotFoundError for a path that does not exist, here or
        elsewhere: searched, it would read as a folder the index has nothing in."""
        if not scope:
            return None, None
        base = (self.root or cwd).resolve()
        target = (cwd / scope).resolve()
        if not target.exists():
            raise FileNotFoundError(scope)
        if target.is_relative_to(base):
            within = target.relative_to(base).as_posix()
            return None, None if within == "." else within
        folder = target if target.is_dir() else target.parent
        try:
            git = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"], cwd=folder, capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired):
            git = None
        root = Path(git.stdout.strip()).resolve() if git is not None and git.returncode == 0 else folder
        within = target.relative_to(root).as_posix()
        return root, None if within == "." else within


class SearchArgs(BaseModel):
    question: str = Field(description="What you are looking for, as a question or a description of the place.")
    scope: str | None = Field(
        default=None,
        description=(
            "Only search under this folder or file, e.g. src/app. It may be outside the working directory, e.g. "
            "another repository: that repository is indexed the first time (which can take a minute) and kept."
        ),
    )


class IndexArgs(BaseModel):
    path: str = Field(default=".", description="The folder to index, relative to the working directory.")


class IndexTool(Tool):
    """Builds or refreshes the store's index for a folder the agent names; the extension sets the name and says
    what the index is for."""

    name = "index"
    description = "Build or refresh the index of a folder, so it can be searched."
    Args = IndexArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"write"})

    def __init__(self, store: Store) -> None:
        self.store = store

    async def run(self, args: IndexArgs, ctx: ToolContext) -> ToolOutput:
        root = (ctx.cwd / args.path).resolve()
        if not root.is_dir():
            return fail(f"{args.path} is not a folder", "give a folder relative to the working directory.")
        try:
            index = await self.store.build(root)
        except ImportError as e:
            return fail(f"the index needs {e.name}", self.store.install)
        except TooManyFiles as e:
            return fail(f"cannot index {args.path}: {e}", "index a subfolder.")
        chunks = sum(len(e.chunks) for e in index.files.values())
        tables = sum(len(e.tables) for e in index.files.values())
        where = _relative(self.store.home(root), ctx.cwd)
        return ToolOutput(
            content=f"Indexed {args.path}: {len(index.files)} files, {chunks} chunks, {tables} tables -> {where}"
        )


class DecisionSearch(Tool):
    """The search tool. An extension subclasses it to give it its name and description: two extensions in one run
    must not share a tool name, as the later tool would replace the earlier."""

    name = "decision_search"
    description = (
        "Find the places in the working directory that answer a question, ranked, each with the call that opens "
        "it. Use it when you do not know where to look or the exact words to grep for; grep for exact words you know."
    )
    Args = SearchArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"read"})
    trace: ClassVar[bool] = False
    """Send each search as it runs to the run's sink, as `search_trace` extension events of the store's name
    (`locate/trace.py`), for the extension's view to draw. The result is the same either way."""
    lexical: ClassVar[bool] = False
    """Steer the walk to where the question's words are (`locate/lexical.py`, an FTS5 table beside the index)."""
    #: Characters of text shown under each of the first `len(inline)` entries (0: pointers only). NDI's locate shows
    #: the best page whole and the next ones' openings, so an answer on the first page costs no read.
    inline: ClassVar[tuple[int, ...]] = ()

    def __init__(self, store: Store, asks: Questions) -> None:
        self.store = store
        self.asks = asks

    def _trace_sink(self) -> Callable[[SearchTrace], None] | None:
        from intern.locate.trace import KIND

        emit, name = self.store.emit, self.store.name
        if not self.trace or emit is None:
            return None
        return lambda trace: emit(ExtensionEvent.of(name, KIND, trace))

    async def search(
        self, index: Index, root: Path, question: str, decisions: DecisionClient, scope: str | None
    ) -> SearchResult:
        """The ranking the tool shows: the walk down the index's tree (an extension may rank otherwise)."""
        from intern.locate.search import search

        return await search(
            index, root, question, decisions, self.asks, scope=scope, trace=self._trace_sink(), lexical=self.lexical
        )

    async def run(self, args: SearchArgs, ctx: ToolContext) -> ToolOutput:
        from intern.locate.search import GAP, MIN_PROB, survivors

        decisions = self.store.decisions
        if decisions is None:
            return fail(f"{self.name} cannot search before the run has started", "use grep and find meanwhile.")
        try:
            elsewhere, scope = await asyncio.to_thread(self.store.place, args.scope, ctx.cwd)
        except FileNotFoundError:
            return fail(f"{args.scope} does not exist", "give a folder or file that exists, or no scope.")
        try:
            index = await self.store.ready(ctx.cwd, elsewhere)
        except ImportError as e:
            return fail(f"the index needs {e.name}", f"{self.store.install}; meanwhile use grep and find.")
        except NoIndex as e:
            return fail(f"no index under {_relative(e.args[0], ctx.cwd)}", "build one with the index tool first.")
        except TooManyFiles as e:
            return fail(f"{self.name} cannot index this folder: {e}", "use grep and find.")
        root = elsewhere or self.store.root or ctx.cwd
        before = decisions.usage.model_copy()
        result = await self.search(index, root, args.question, decisions, scope)
        kept = survivors([(str(n), p.score) for n, p in enumerate(result.places) if p.score > 0], GAP, PLACES_SHOWN)
        places = [result.places[int(n)] for n, _ in kept]
        files = [(p, s) for p, s in result.files if s > 0][:FILES_SHOWN]
        if not places and not files:
            where = f"under {args.scope}" if args.scope else "in the working directory"
            walked = _walked(result.walk, decisions.usage.since(before).asks)
            return ToolOutput(content=f"Nothing {where} looks related. {walked} Try grep or find.")
        weak = (
            f"\n(No folder scored {MIN_PROB} or more for this question; the walk entered the best ones anyway, so "
            "these places are weak.)"
            if result.walk is not None and result.walk.forced
            else ""
        )
        prefix = _relative(root, ctx.cwd)  # the index's paths, as the agent's read sees them
        prefix = "" if prefix == "." else prefix + "/"
        lines = [f"Places for: {args.question}"]
        firsts = {}
        for place in places:
            firsts.setdefault((place.path, place.table), place)
        for n, ((where, score, hint), best) in enumerate(zip(_entries(places, index), firsts.values(), strict=True), 1):
            lines.append(f"{n}. {prefix}{where}  [{score:.2f}]\n   {hint.replace('read ', 'read ' + prefix, 1)}")
            if n <= len(self.inline) and (text := _excerpt(index, best, self.inline[n - 1])):
                lines.append(text)
        if files:
            lines.append("Files: " + ", ".join(f"{prefix}{p} {s:.2f}" for p, s in files))
        return ToolOutput(content="\n".join(lines) + weak)


def _walked(walk: Walk | None, asks: int) -> str:
    """How far an empty search got, so a walk that stopped at its first level reads differently from one that judged
    its files and found nothing."""
    if walk is None or not walk.in_scope:
        return "The index holds no file there (what git ignores is not indexed)."
    said = (
        f"The walk started at {walk.start}, opened {_count(walk.opened, 'folder or group', 'folders or groups')} "
        f"down {_count(walk.depth, 'level', 'levels')} and judged {_count(walk.files, 'file', 'files')}, in "
        f"{_count(asks, 'ask', 'asks')}"
    )
    if walk.left is not None:
        said += f"; the best folder it did not enter was {walk.left.label} ({walk.left.score:.2f})"
    return said + "."


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _excerpt(index: Index, place: Place, chars: int) -> str:
    """The text of ``place`` (its pages, or its lines) as the index derived it, or a table's rows and column profile,
    cut at ``chars``, indented."""
    from intern.locate.index import DERIVED_DIR, pages_of

    entry = index.files.get(place.path)
    if entry is not None and place.table:  # a table: its columns, as a page shows its text
        info = next((t for t in entry.tables if t.name == place.table), None)
        if info is None or not info.profile:
            return ""
        text = f"{info.rows:,} rows × {len(info.columns)} columns:\n{info.profile}"
        cut = text[:chars] + (f"\n[... {len(text) - chars} more characters: DESCRIBE it]" if len(text) > chars else "")
        return "\n".join("   | " + line for line in cut.splitlines())
    if entry is None or place.key is None or not entry.derived:
        return ""
    try:
        derived = (index.home / DERIVED_DIR / entry.derived).read_text(errors="replace")
    except OSError:
        return ""
    if entry.unit == "page":
        text = "\n".join(pages_of(derived)[place.start - 1 : place.end])
    else:
        text = "\n".join(derived.splitlines()[place.start - 1 : place.end])
    text = text.strip()
    if not text:
        return ""
    cut = text[:chars] + (f"\n[... {len(text) - chars} more characters: read it]" if len(text) > chars else "")
    return "\n".join("   | " + line for line in cut.splitlines())


def _relative(path: Path, cwd: Path) -> str:
    """``path`` as the agent names it from ``cwd`` (pure path arithmetic, no filesystem)."""
    try:
        return path.resolve().relative_to(cwd.resolve()).as_posix() or "."
    except ValueError:
        return str(path)


def _entries(places: list[Place], index: Index) -> list[tuple[str, float, str]]:
    """The places as one entry per file (or table): the best place, where, with the file's other places named;
    the score; one hint that opens the best place with margin, or a tiny file whole."""
    groups: dict[tuple[str, str | None], list[Place]] = {}
    for place in places:  # in rank order, so a group's first place is its best
        groups.setdefault((place.path, place.table), []).append(place)
    out: list[tuple[str, float, str]] = []
    for (path, table), group in groups.items():
        best, score = group[0], group[0].score
        if table or best.key is None:
            out.append((best.where, score, best.hint))
            continue
        others = [p for p in group[1:] if p.key is not None]
        more = f"; {len(others)} more here: " + ", ".join(_span(p) for p in others[:MORE_SHOWN]) if others else ""
        if best.unit == "page":
            out.append((f"{path}  {_span(best)}{more}", score, best.hint))
            continue
        entry = index.files.get(path)
        if entry is not None and entry.units <= WHOLE_FILE:
            hint = f"read {path}"
        else:
            offset = max(1, best.start - READ_MARGIN)
            hint = f"read {path} offset={offset} limit={min(best.end + READ_MARGIN - offset + 1, READ_LIMIT)}"
        out.append((f"{path}  {_span(best)}{more}", score, hint))
    return out


def _span(place: Place) -> str:
    if place.unit == "page":
        return f"page {place.start}" if place.start == place.end else f"pages {place.start}-{place.end}"
    return f"{place.title} ({place.start}-{place.end})"


class Locate(h.Extension):
    """An extension that keeps a directory indexed and searchable: subclasses set the name, the processors, the
    questions, the home folder, whether the index builds on its own (``auto``) or when the agent calls the index
    tool, the note, and what `tools()` returns beside `decision_search`."""

    name: ClassVar[str] = "locate"
    search_tool: ClassVar[type[DecisionSearch]] = DecisionSearch
    processors: ClassVar[Processors]
    asks: ClassVar[Questions]
    folder: ClassVar[str] = names.PROJECT_DIR
    moved_from: ClassVar[str | None] = None
    auto: ClassVar[bool] = True
    note: ClassVar[str] = ""
    #: The note when the working directory's index is there at the start (default: `note`).
    indexed_note: ClassVar[str] = ""
    install: ClassVar[str] = "install intern-agent[code-search]"

    def __init__(self) -> None:
        self.store = Store(type(self).processors, self.name, self.install, self.folder, self.auto, self.moved_from)
        self.pinned: str | None = None

    def tools(self) -> list[Tool]:
        return [self.search_tool(self.store, self.asks)]

    async def before_agent_start(self, ctx: h.RunContext) -> None:
        self.store.decisions = ctx.decisions
        self.store.emit = ctx.emit
        self.store.mark_dirty()
        self.pinned = await self.pin(Path(ctx.cwd))

    async def pin(self, cwd: Path) -> str | None:
        """The note this run pins: the note and the map of the working directory's index, `unindexed` when the agent
        builds the index, nothing past MAX_FILES. An extension may pin nothing where the folder is not its kind."""
        try:
            index = await self.store.ready(cwd)
        except TooManyFiles:
            return None  # no map to pin; the tool says why if called
        except NoIndex:
            return await asyncio.to_thread(self.unindexed, cwd)  # the agent builds it with the index tool
        root = index.nodes.get(".")
        note = self.indexed_note or self.note
        return f"{note}\n{root.card}" if root else note

    def unindexed(self, cwd: Path) -> str | None:
        """The note when the working directory has no index and the agent builds one (None: pin nothing)."""
        return self.note

    async def context(self, view: ContextView, ctx: h.RunContext) -> None:
        if self.pinned:
            # The user's request follows the pinned segment, in the history: what the model reads last of the two
            # stays the request, not the map.
            view.pinned.append(self.pinned)

    async def tool_result(self, call: ToolCall, result: ToolResult, ctx: h.RunContext) -> ToolResult | None:
        if call.name in CHANGING_TOOLS and result.status == "ok":
            self.store.mark_dirty()
        return None
