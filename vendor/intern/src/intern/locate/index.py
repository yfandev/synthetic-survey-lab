"""The index: every file of a directory as a card and chunks, in a search tree, kept in that directory's `.intern/`.

An index lives in a home directory an extension names: `.intern/` for the code-search extension, `.decision_workspace/`
for the workspace extension, so both can index the same directory.

    <home>/index.json         this model: files, chunks, the search tree and their cards
    <home>/derived/<path>.md  the text of a file that is not readable text, when a processor derives one
                              (page-marked for a PDF or workbook, line by line for a Word document)
    <home>/tables.duckdb      the workspace extension's tables, when its processors are in the build
    .intern/.gitignore          gets a line for the home, so the index stays out of git as tool caches do, and
                              nothing else of `.intern/` is touched

`build` is incremental: a file whose size and mtime are unchanged keeps its entry, the rest are processed again,
removed files leave, and the tree and keywords are recomputed over the whole (they are cheap). It never calls a model.
In a git work tree it indexes the files git sees, so what .gitignore keeps out of git is not searched either (in
audit-app 16k files, not the 478k a walk found under its ignored client workspaces); MAX_FILES bounds one index.

Files whose processor is marked ``parallel`` (PDFs: OCR takes seconds a page) are processed in worker processes, a
few at a time, so a build uses the machine and a native crash in a processor (OpenCV's resize under OCR segfaulted
on a scanned PDF) costs that file, not the build or the agent running it. What is processed is saved to
`<home>/index.partial.json` as it goes, so an interrupted build resumes where it stopped.
"""

from __future__ import annotations

import itertools
import multiprocessing
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from stat import S_ISREG
from typing import IO

from pydantic import BaseModel, Field, ValidationError

from intern import names
from intern.locate import cards, grouping, tree
from intern.locate.grouping import Vectorizer
from intern.locate.processors import (
    Chunk,
    Derivative,
    KeyedProcessor,
    MarkdownProcessor,
    Processor,
    PythonProcessor,
    TableInfo,
    Unit,
    WindowProcessor,
)
from intern.locate.tree import Node
from intern.tools.search import SKIP_DIRS

INDEX_FILE = "index.json"
#: The entries of a build still under way, saved every CHECKPOINT_S so an interrupted build resumes.
PARTIAL_FILE = "index.partial.json"
CHECKPOINT_S = 30.0
#: Cores per worker process of a build: a worker OCRs on PROCESSING_THREADS threads and holds about 2.5 GB while it
#: does, so a build leaves room for the rest of what the machine runs.
WORKER_CORES = 4
#: Threads one file's processing uses (OCR's inference), in a build's worker or in `read`. Past two, ONNX Runtime's
#: threads mostly wait: on a folder of scanned invoices 4 threads OCR'd 20-25% faster than 2 for 40% more CPU, and 14
#: threads 13% faster for five times the CPU, the same text either way.
PROCESSING_THREADS = 2
DERIVED_DIR = "derived"
VERSION = 5
#: Files larger than this are not indexed, unless their processor sets its own ``max_bytes`` (None: no limit, for a
#: table queried in place).
MAX_FILE_BYTES = 50 * 1024 * 1024
#: Files one index holds at most: past this a build stops with `TooManyFiles` rather than exhaust memory.
MAX_FILES = 100_000
GIT_TIMEOUT_S = 60
#: Office files, binary, so no text processor reads them: what a document index reads (PDF, Word, workbooks, Parquet).
DOCUMENT_SUFFIXES = frozenset({".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".parquet"})
#: Files `sample` lists at most: enough to tell a folder of documents from one of code, quick even in a home folder.
SAMPLE_FILES = 2000
#: Characters of a file's text its vector and keywords are built from.
HEAD_CHARS = 3000
BINARY_SNIFF_BYTES = 8192
#: The file tools (ruff, pytest, uv, pip) leave in a cache directory, so the walk skips it by that and not by name.
CACHE_TAG = "CACHEDIR.TAG"
#: The first line a cache directory tag must start with (https://bford.info/cachedir/).
CACHE_SIGNATURE = "Signature: 8a477f597d28d172789f06886806bc55"
PAGE_MARK = "--- Page {n} ---"
REDRAW_S = 0.05
_PAGE_MARK = re.compile(r"^--- Page \d+ ---\n?", re.MULTILINE)

Report = Callable[[str], None]
#: Called after each file of a build with (done, total).
ProgressFn = Callable[[int, int], None]


class TooManyFiles(Exception):
    """More than MAX_FILES files to index under a root (``args[0]``)."""

    def __str__(self) -> str:
        return f"more than {MAX_FILES:,} files to index under {self.args[0]}"


class FileEntry(BaseModel):
    path: str
    size: int
    mtime_ns: int
    processor: str
    unit: Unit
    units: int
    summary: str = ""
    #: What the file says it is (`Derivative.description`), for the root card's purposes only.
    description: str = ""
    note: str = ""
    #: Where the text lives when the source is not readable text, relative to `<home>/derived`.
    derived: str | None = None
    #: The text the file's vector and keywords are built from.
    head: str
    #: The card before its keywords line, which depends on the rest of the workspace.
    outline: str
    #: This file's line on its group's card.
    member: str
    card: str = ""
    chunks: list[Chunk] = []
    tables: list[TableInfo] = []
    #: What a Python file imports (`tree.import_names`), kept while the file is unchanged; None until a build reads it.
    imports: list[str] | None = None


class Index(BaseModel):
    version: int = VERSION
    #: The directory this index lives in (its derived text beside it); set on load and build, not saved.
    home: Path = Field(default=Path(), exclude=True)
    files: dict[str, FileEntry] = {}
    #: The search tree by node key; ``.`` is the root. Leaves are groups of files, inner nodes hold child keys.
    nodes: dict[str, Node] = {}

    @property
    def leaves(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == "leaf"]

    def chunk(self, path: str, key: str) -> Chunk | None:
        entry = self.files.get(path)
        return next((c for c in entry.chunks if c.key == key), None) if entry else None


#: What a build gets its processors from, given the indexed root and the index's home directory (the home may be
#: below the root, so a processor naming things by relative path takes the root); a processor may hold a resource
#: (a database connection) and close it when the build ends.
Processors = Callable[[Path, Path], list[Processor]]


def default_processors(root: Path, home: Path) -> list[Processor]:
    """The text processors in order of precedence: Python by its syntax tree, Markdown and reStructuredText by
    heading, keyed files by top-level key, anything else by windows of lines."""
    return [PythonProcessor(), MarkdownProcessor(), KeyedProcessor(), WindowProcessor()]


def pages_of(derived: str) -> list[str]:
    """The pages of a derived text, as `_entry` wrote them."""
    parts = _PAGE_MARK.split(derived)
    return [part.strip("\n") for part in parts[1:]]


def load(home: Path) -> Index | None:
    """The index saved in ``home``, or None when there is none or it is from another version."""
    return _load(home / INDEX_FILE, home)


def _load(path: Path, home: Path) -> Index | None:
    try:
        index = Index.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError):
        return None
    index.home = home
    return index if index.version == VERSION else None


def walk(root: Path) -> Iterator[tuple[str, os.stat_result]]:
    """Every regular file under ``root`` the index may hold, as (relative posix path, stat), in a sorted walk's
    order. In a git work tree these are the files git sees, tracked or untracked but not ignored: what .gitignore
    keeps out of git (build output, data, scratch, other checkouts) is neither indexed nor searched."""
    listed = _git_files(root)
    for rel in listed if listed else _walk_files(root):
        if rel.rsplit("/", 1)[-1].startswith("."):
            continue
        try:
            stat = (root / rel).stat()
        except OSError:
            continue
        # An empty file has nothing to index (placeholders by the hundred); a pipe or device is not a file to read.
        # How large a file may be is its processor's call (`_processor_for`).
        if S_ISREG(stat.st_mode) and stat.st_size > 0:
            yield rel, stat


def sample(root: Path, limit: int = SAMPLE_FILES) -> list[str]:
    """The first ``limit`` files `walk` finds under ``root``, in a git work tree or a plain folder: what it holds."""
    # shortcut: the first files of a sorted walk, not a spread across subfolders; a folder whose first subfolders
    # differ from the rest is misjudged. Sample each subfolder in turn if that shows up.
    return [rel for rel, _ in itertools.islice(walk(root), limit)]


def _git_files(root: Path) -> list[str] | None:
    """What git lists under ``root`` (relative posix paths, in `_walk_files`' order, without SKIP_DIRS), or None
    outside a work tree. Empty when ``root`` itself is ignored, and the caller then walks it."""
    try:
        done = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    skip = SKIP_DIRS | {names.PROJECT_DIR}
    rels = {os.fsdecode(p) for p in done.stdout.split(b"\0") if p}  # a conflicted file is listed once per stage
    rels = [rel for rel in rels if not skip.intersection(rel.split("/")[:-1])]
    # os.walk's order with sorted names: a folder's files, then each subfolder in turn.
    return sorted(rels, key=lambda rel: [(1, part) for part in rel.split("/")[:-1]] + [(0, rel.rsplit("/", 1)[-1])])


def _walk_files(root: Path) -> Iterator[str]:
    for folder, dirs, files in os.walk(root):
        if CACHE_TAG in files and folder != str(root):
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and d != names.PROJECT_DIR)
        for name in sorted(files):
            yield (Path(folder) / name).relative_to(root).as_posix()


def stale(root: Path, index: Index, processors: Processors = default_processors) -> bool:
    """Whether a file was added, removed or changed since ``index`` was built."""
    seen = set()
    available = processors(root, index.home)
    for rel, stat in walk(root):
        seen.add(rel)
        entry = index.files.get(rel)
        if entry is None:
            if _processor_for(root / rel, available, stat.st_size) is not None:
                return True
        elif (entry.size, entry.mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            return True
    return any(path not in seen for path in index.files)


def _processor_for(path: Path, available: list[Processor], size: int = 0) -> Processor | None:
    """The processor for the file at ``path``, if one takes it at ``size`` bytes."""
    suffix = path.suffix.lower()
    found: Processor | None = next((p for p in available if suffix in p.suffixes), None)
    if found is None:
        try:
            with path.open("rb") as f:
                found = None if b"\x00" in f.read(BINARY_SNIFF_BYTES) else WindowProcessor()
        except OSError:
            return None
    if found is None:
        return None
    cap = _max_bytes(found)
    return found if cap is None or size <= cap else None


def _max_bytes(processor: Processor) -> int | None:
    return getattr(processor, "max_bytes", MAX_FILE_BYTES)


def build(
    root: Path,
    home: Path | None = None,
    *,
    processors: Processors = default_processors,
    vectors: Vectorizer | None = None,
    report: Report = lambda line: print(line, file=sys.stderr),
    progress: ProgressFn | None = None,
) -> Index:
    """Build or refresh the index of ``root`` in ``home`` (default ``root/.intern``) and return it. ``vectors``
    places files for the tree's affinity; the default is the TF-IDF rows keywords come from."""
    root = root.resolve()
    home = (home or root / names.PROJECT_DIR).resolve()
    home.mkdir(parents=True, exist_ok=True)
    ignore(root, home)
    available = processors(root, home)
    try:
        index = _process(root, home, available, report, progress)
    finally:
        for processor in available:
            getattr(processor, "close", lambda: None)()
    _finish(root, index, vectors)
    _write(home / INDEX_FILE, index)
    (home / PARTIAL_FILE).unlink(missing_ok=True)
    report(
        f"{len(index.files)} files, {sum(len(e.chunks) for e in index.files.values())} chunks, "
        f"{sum(len(e.tables) for e in index.files.values())} tables, {len(index.leaves)} groups -> {home}"
    )
    return index


def ignore(root: Path, home: Path, name: str = INDEX_FILE) -> None:
    """A line in `.intern/.gitignore` for a file the home holds (``name`` when the home is `.intern/` itself, else the
    whole subfolder) and one for the ignore file itself: like .ruff_cache and .pytest_cache, the index keeps itself
    out of git (an agent's `git add -A` or a diff of the worktree would otherwise sweep it up) and out of `git status`,
    without touching anything else the user keeps in `.intern/`."""
    project = root / names.PROJECT_DIR
    if home == project:
        line = name
    elif home.is_relative_to(project):
        line = home.relative_to(project).as_posix() + "/"
    else:  # a home elsewhere is all cache: out of git, and (the tag) out of every walk outside a git work tree
        (home / ".gitignore").write_text("*\n")
        (home / CACHE_TAG).write_text(CACHE_SIGNATURE + "\n# The index of Intern; rebuilt as needed.\n")
        return
    ignore = project / ".gitignore"
    lines = ignore.read_text().splitlines() if ignore.exists() else []
    missing = [x for x in (".gitignore", line) if x not in lines]
    if missing:
        ignore.write_text("".join(f"{x}\n" for x in [*lines, *missing]))


class Progress:
    """A bar on a terminal, one line rewritten as the build goes:
    ``Indexing:  13%|█████                                   | 73/541 [00:00<00:02, 157 file/s]``.
    Off a terminal it writes nothing. Call with (done, total); ``close`` ends the line. Redrawn at most every
    REDRAW_S, and at the end."""

    def __init__(self, label: str = "Indexing", stream: IO[str] | None = None, width: int = 40) -> None:
        self.stream = stream or sys.stderr
        self.on = self.stream.isatty()
        self.label, self.width, self.t0, self.drawn = label, width, time.perf_counter(), False
        self.last = 0.0

    def __call__(self, done: int, total: int) -> None:
        if not self.on:
            return
        now = time.perf_counter()
        if done < total and now - self.last < REDRAW_S:
            return
        self.last = now
        elapsed = now - self.t0
        rate = done / elapsed if elapsed > 0 else 0.0
        left = (total - done) / rate if rate > 0 else 0.0
        share = done / total if total else 1.0
        bar = ("█" * round(share * self.width)).ljust(self.width)
        self.stream.write(
            f"\r{self.label}: {share:4.0%}|{bar}| {done}/{total} [{_clock(elapsed)}<{_clock(left)}, {rate:.0f} file/s]"
        )
        self.stream.flush()
        self.drawn = True

    def close(self) -> None:
        if self.drawn:
            self.stream.write("\n")
            self.stream.flush()


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


_Job = tuple[str, os.stat_result, Processor]
_Finish = Callable[[_Job, Derivative | BaseException], None]


def _process(root: Path, home: Path, available: list[Processor], report: Report, progress: ProgressFn | None) -> Index:
    """Every file's entry: kept from the prior index (or an interrupted build's) when unchanged, else processed again;
    removed files dropped."""
    prior = load(home) or Index()
    if (partial := _load(home / PARTIAL_FILE, home)) is not None:
        prior.files |= partial.files
    index = Index(home=home)
    files = list(itertools.islice(walk(root), MAX_FILES + 1))
    if len(files) > MAX_FILES:
        raise TooManyFiles(root)
    todo: list[_Job] = []
    for rel, stat in files:
        kept = prior.files.get(rel)
        if kept is not None and (kept.size, kept.mtime_ns) == (stat.st_size, stat.st_mtime_ns):
            index.files[rel] = kept
            continue
        processor = _processor_for(root / rel, available, stat.st_size)
        if processor is not None:
            todo.append((rel, stat, processor))
        elif (taker := _processor_for(root / rel, available)) is not None:
            report(f"skip {rel}: {stat.st_size // 2**20} MB, over the {(_max_bytes(taker) or 0) // 2**20} MB limit")
    missing: set[str] = set()
    done = len(files) - len(todo)
    saved = time.monotonic()

    def finish(job: _Job, result: Derivative | BaseException) -> None:
        nonlocal done, saved
        rel, stat, processor = job
        done += 1
        if isinstance(result, ImportError):
            if result.name not in missing:
                missing.add(result.name or "")
                report(f"{processor.name} files skipped: {result} (install intern-agent[workspace] or [ocr])")
        elif isinstance(result, BaseException):
            report(f"skip {rel}: {type(result).__name__}: {result}")
        else:
            index.files[rel] = _entry(rel, stat, processor.name, result, home)
        if progress is not None:
            progress(done, len(files))
        if time.monotonic() - saved >= CHECKPOINT_S:
            _write(home / PARTIAL_FILE, index)
            saved = time.monotonic()

    for job in todo:
        if not getattr(job[2], "parallel", False):
            finish(job, _derive(job[2], root / job[0], home))
    _in_workers([job for job in todo if getattr(job[2], "parallel", False)], root, home, finish)
    _drop(prior, index, home, available, report)
    order = {rel: n for n, (rel, _) in enumerate(files)}
    index.files = dict(sorted(index.files.items(), key=lambda kv: order.get(kv[0], len(order))))
    return index


def _derive(processor: Processor, path: Path, home: Path) -> Derivative | BaseException:
    try:
        return processor.process(path, home)
    except Exception as e:
        return e


def _in_workers(jobs: list[_Job], root: Path, home: Path, finish: _Finish) -> None:
    """Process ``jobs`` in worker processes, at most one in flight per worker. When a worker dies (a native crash),
    the files in flight are retried one at a time, so the one that kills its worker again is the one skipped."""
    if not jobs:
        return
    cores = os.cpu_count() or 1
    workers = max(1, min(len(jobs), cores // WORKER_CORES))
    suspects = _pool(jobs, root, home, finish, workers)
    for job in suspects:
        if _pool([job], root, home, finish, 1):
            finish(job, RuntimeError("its processing crashed the worker process twice"))


def _pool(
    jobs: list[_Job],
    root: Path,
    home: Path,
    finish: _Finish,
    workers: int,
) -> list[_Job]:
    """Run ``jobs`` on a pool of ``workers``, starting a new pool when one breaks; returns the jobs in flight when a
    pool broke (one of them crashed it)."""
    queue, crashed = list(reversed(jobs)), []
    context = multiprocessing.get_context("spawn")  # never fork: the build runs in a thread of the agent's process
    while queue:
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            flying: dict[Future[Derivative | BaseException], _Job] = {}
            try:
                while queue or flying:
                    while queue and len(flying) < workers:
                        flying[pool.submit(_derive, queue[-1][2], root / queue[-1][0], home)] = queue[-1]
                        queue.pop()
                    finished, _ = wait(flying, return_when=FIRST_COMPLETED)
                    for future in finished:
                        try:
                            result = future.result()
                        except BrokenProcessPool:
                            raise
                        except Exception as e:  # the result could not be sent back
                            result = e
                        finish(flying.pop(future), result)
            except BrokenProcessPool:
                crashed += flying.values()
    return crashed


def _write(path: Path, index: Index) -> None:
    """Replace ``path`` whole, so a reader never sees half an index."""
    tmp = path.with_suffix(".tmp")
    # Bytes straight from the serializer: model_dump_json's str and its encoding were 1.2 GB more at audit-app's size.
    tmp.write_bytes(index.__pydantic_serializer__.to_json(index))
    tmp.replace(path)


def _entry(rel: str, stat: os.stat_result, processor: str, derivative: Derivative, home: Path) -> FileEntry:
    derived = None
    if derivative.derived:
        derived = f"{rel}.md"
        target = home / DERIVED_DIR / derived
        target.parent.mkdir(parents=True, exist_ok=True)
        if derivative.unit == "page":
            text = "\n\n".join(f"{PAGE_MARK.format(n=n)}\n{page}" for n, page in enumerate(derivative.units, 1))
        else:
            text = "\n".join(derivative.units)
        target.write_text(text + "\n")
    for chunk in derivative.chunks:
        chunk.card = cards.chunk_card(rel, chunk, derivative)
    titles = " ".join(f"{c.title} {c.summary}" for c in derivative.chunks)
    body = derivative.head or " ".join(derivative.units[:50])
    # Content only: the path is on the card, and in the vector it made folder names the keywords of every folder.
    head = f"{derivative.summary} {titles} {body}"[:HEAD_CHARS]
    return FileEntry(
        path=rel,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        processor=processor,
        unit=derivative.unit,
        units=len(derivative.units),
        summary=derivative.summary,
        description=derivative.description,
        note=derivative.note,
        derived=derived,
        head=head,
        outline=cards.file_card(rel, derivative, []),
        member=cards.member_line(derivative),
        chunks=derivative.chunks,
        tables=derivative.tables,
    )


def _drop(prior: Index, index: Index, home: Path, available: list[Processor], report: Report) -> None:
    """Derived text of files that left the index, and whatever their processor keeps for them (tables)."""
    by_name = {p.name: p for p in available}
    for path, entry in prior.files.items():
        if path in index.files:
            continue
        if entry.derived:
            (home / DERIVED_DIR / entry.derived).unlink(missing_ok=True)
        remove = getattr(by_name.get(entry.processor), "remove", None)
        if remove is not None:
            try:
                remove(entry)
            except Exception as e:
                report(f"could not remove what was kept for {path}: {e}")


def _finish(root: Path, index: Index, vectors: Vectorizer | None) -> None:
    """Keywords and final cards for every file, then the tree and its cards."""
    heads = {path: entry.head for path, entry in index.files.items()}
    paths, matrix, vocab = grouping.tfidf(heads)
    rows = {path: matrix.getrow(i) for i, path in enumerate(paths)}
    for path, entry in index.files.items():
        keywords = grouping.top_terms(rows[path], vocab, cards.KEYWORDS)
        entry.card = entry.outline + (f"\n  keywords: {', '.join(keywords)}" if keywords else "")
    placed = vectors(heads) if vectors is not None else rows
    imports: dict[str, list[str]] = {}
    for path, entry in index.files.items():
        if path.endswith(".py"):
            if entry.imports is None:  # new or changed since the last build
                entry.imports = tree.import_names(root, path)
            imports[path] = entry.imports
    adj, centrality = tree.affinity(list(index.files), imports, placed)
    index.nodes = tree.build(list(index.files), adj)
    names = {
        path: [
            c.title.removeprefix("tab ").split(" → ")[0] if c.kind == "table" else c.key
            for c in entry.chunks
            if c.parent is None and c.kind not in {"page", "pages", "window"}
        ]
        for path, entry in index.files.items()
    }
    tree.write_cards(
        index.nodes,
        {p: e.member for p, e in index.files.items()},
        names,
        centrality,
        rows,
        vocab,
        {p: e.summary if e.processor == "python" else e.description for p, e in index.files.items()},
        {p: e.summary for p, e in index.files.items() if e.processor == "python"},
    )
