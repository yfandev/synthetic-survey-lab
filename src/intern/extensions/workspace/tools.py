"""The workspace extension's own tools: `sql` over the index's tables, and a `read` that knows documents. Both read
the extension's `Store` (see `intern.locate.store`, which also holds the search and index tools)."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeVar

from pydantic import BaseModel, Field

from intern import names
from intern.locate.index import DOCUMENT_SUFFIXES  # what `read` shows from derived text, not bytes
from intern.locate.store import NoIndex, Store
from intern.tools.base import Effect, Tool, ToolContext, ToolOutput, fail
from intern.tools.files import ReadArgs, ReadTool, page_lines
from intern.tools.shell import DEFAULT_TIMEOUT_S, MAX_TIMEOUT_S

if TYPE_CHECKING:
    import duckdb

    from intern.locate.index import Index
    from intern.locate.processors import TableInfo

T = TypeVar("T")

INSTALL = "install intern-agent[workspace] (and [ocr] for scanned pages)"
SQL_ROWS = 50
#: A failed query's error is shown up to this many characters (DuckDB's run to a few lines).
SQL_ERROR_CHARS = 1200
#: What `sql` runs: queries (SELECT covers DESCRIBE and SHOW), and the statements that build the session's scratch
#: tables and views; never COPY, SET, ATTACH, INSTALL or PRAGMA. A write can only reach the scratch database: the
#: workspace's tables are attached read-only.
QUERY_STATEMENTS = frozenset({"SELECT", "EXPLAIN"})
SCRATCH_STATEMENTS = frozenset({"CREATE", "DROP", "INSERT", "UPDATE", "DELETE", "ALTER"})
#: A CREATE that stores credentials (a persistent one is written to disk) is not scratch.
_SECRET = re.compile(r"^\s*create\s+(?:or\s+replace\s+)?(?:persistent\s+|temporary\s+|temp\s+)?secret\b", re.I)
#: A query's result is kept as a table (res_1, res_2, ...) for later queries if it has at most this many rows; a
#: larger one is not kept, rather than kept cut short, where a later query would count only part of it.
KEEP_ROWS = 1_000_000
#: Where, under the index's home, a session's sql spills what does not fit its memory: one folder per process.
SPILL_DIR = "sql-tmp"
#: A read of at most this many pages (by `pages=`) has its scanned pages OCR'd first (seconds a page).
OCR_READ_PAGES = 6
#: `DESCRIBE <table>` of a workspace table shows the index's profile of it and this many of its first rows.
DESCRIBE_ROWS = 3
_MEANT = re.compile(r'Did you mean "([^"]+)"')
#: A table's name as sql_export takes it (res_3, a scratch table, a workspace table; quoted or not).
_NAME = re.compile(r'"?[A-Za-z_][\w.]*"?')
#: What sql_export writes.
EXPORT_FORMATS = frozenset({".csv", ".parquet"})
_DESCRIBE = re.compile(r'^\s*desc(?:ribe)?\s+(?:workspace\.(?:main\.)?)?("?)([^\s";]+)\1\s*;?\s*$', re.I)
_RANGE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$")


class SqlArgs(BaseModel):
    query: str = Field(
        description="A DuckDB SQL query, or several separated by ';' (each one's result is shown). SHOW TABLES lists "
        "what there is; DESCRIBE <table> its columns, their values and ranges, and its first rows."
    )
    limit: int = Field(default=SQL_ROWS, ge=1, le=1000, description="Rows to show at most.")
    timeout_s: int = Field(
        default=DEFAULT_TIMEOUT_S, ge=1, le=MAX_TIMEOUT_S, description="Seconds before the query is stopped."
    )


class Sql(Tool):
    name = "sql"
    description = (
        "Query the workspace's tables with SQL (DuckDB): every spreadsheet tab, CSV and Parquet file in the "
        "working directory is a table, named after its path (reports/sales.xlsx tab Sales is reports__sales__sales). "
        "decision_files names the table that holds what you look for. It reads the files in place over every row, so "
        "do the work on a table here rather than load its file in python. Each query's result is kept as a table for "
        "the session (res_1, res_2, ...): page, filter or join it in a later query rather than run a heavy query "
        "again. For steps of your own, CREATE TABLE or VIEW (and INSERT, DROP) in the session's scratch database; "
        "the workspace's tables are read-only."
    )
    Args = SqlArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"read"})

    def __init__(self, store: Store) -> None:
        self.store = store
        # One query at a time: DuckDB already spreads one over every core, and two at once would each claim the
        # query's memory share.
        self._one = asyncio.Lock()
        self._session: SqlSession | None = None

    async def open(self, ctx: ToolContext) -> tuple[SqlSession, Index] | ToolOutput:
        """The session's DuckDB over the index, opened on first use, or why there is none."""
        try:
            index = await self.store.ready(ctx.cwd)
        except ImportError as e:
            return fail(f"the workspace tables need {e.name}", f"{INSTALL}.")
        except NoIndex:
            return fail("no files are indexed", "call index_files with the folder of documents first.")
        if not any(entry.tables for entry in index.files.values()):
            return fail("the workspace has no tables", "Tables come from .xlsx, .csv, .tsv and .parquet files.")
        root = (self.store.root or ctx.cwd).resolve()
        if self._session is None or self._session.root != root or self._session.home != index.home:
            if self._session is not None:
                self._session.close()
            self._session = SqlSession(index.home, root)
        return self._session, index

    async def timed(self, session: SqlSession, work: Callable[[], T], timeout_s: int) -> T | ToolOutput:
        """``work`` on the session, alone (one query at a time) and stopped after ``timeout_s``."""
        async with self._one:
            running = asyncio.ensure_future(asyncio.to_thread(work))
            try:
                return await asyncio.wait_for(asyncio.shield(running), timeout_s)
            except TimeoutError:
                session.con.interrupt()  # the thread's query raises InterruptException and the thread ends
                await asyncio.gather(running, return_exceptions=True)
                return fail(
                    f"the query ran over {timeout_s}s and was stopped",
                    "Narrow it (filter or aggregate first, avoid window functions and joins over every row), "
                    "build on a kept result, or raise timeout_s.",
                    argument="timeout_s",
                )

    async def run(self, args: SqlArgs, ctx: ToolContext) -> ToolOutput:
        opened = await self.open(ctx)
        if isinstance(opened, ToolOutput):
            return opened
        import duckdb

        session, index = opened
        infos = {t.name: t for entry in index.files.values() for t in entry.tables}
        tables = sorted(infos)
        try:
            result = await self.timed(session, lambda: session.run(args.query, args.limit, infos), args.timeout_s)
        except duckdb.Error as e:  # the query does not parse
            return _failed(e, tables)
        if isinstance(result, ToolOutput):
            return result
        if isinstance(result, str):
            return fail(
                f"sql runs queries and builds scratch tables, not {result}",
                "Use SELECT (with WITH, DESCRIBE, SHOW, EXPLAIN), or CREATE TABLE/VIEW, INSERT and DROP in the "
                "session's scratch database. To write a result to a file, use sql_export.",
                argument="query",
            )
        shown, error = result
        batch = len(shown) + (error is not None) > 1
        text = "\n\n".join(_show(one, args.limit, batch) for one in shown)
        if len(text) > ctx.max_output_chars:
            text = text[: ctx.max_output_chars] + "\n[cut: narrow the query or select fewer columns]"
        if error is not None:  # the statements before it ran, and their results are shown
            return _failed(error, tables, text, f"statement {len(shown) + 1}" if batch else "the query")
        return ToolOutput(content=text)


class SqlExportArgs(BaseModel):
    source: str = Field(description="What to write: a kept result (res_3), a scratch table, or a SELECT query.")
    path: str = Field(description="The file to write, in the working directory: .csv or .parquet.")
    timeout_s: int = Field(
        default=DEFAULT_TIMEOUT_S, ge=1, le=MAX_TIMEOUT_S, description="Seconds before the query is stopped."
    )


class SqlExport(Tool):
    name = "sql_export"
    description = (
        "Write a sql result to a CSV or Parquet file in the working directory, every row of it, without the rows "
        "passing through the conversation: for a list a deliverable needs, or a script's input. Give a kept result "
        "(res_3), a scratch table or a SELECT query. It never writes over an indexed document."
    )
    Args = SqlExportArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"write"})

    def __init__(self, sql: Sql) -> None:
        self.sql = sql
        #: What this session wrote: it may write those again once the index, refreshed, holds them as documents.
        self.wrote: set[Path] = set()

    async def run(self, args: SqlExportArgs, ctx: ToolContext) -> ToolOutput:
        path = ctx.resolve(args.path).resolve()
        if path.suffix.lower() not in EXPORT_FORMATS:
            return fail(f"{args.path} is not a .csv or .parquet file", "Give a path ending in .csv or .parquet.")
        store = self.sql.store
        private = {names.PROJECT_DIR, ".git", store.folder, store.moved_from}
        if not path.is_relative_to(ctx.cwd.resolve()) or private & set(path.parts):
            return fail(f"{args.path} is outside the working directory", "Give a path in it.", argument="path")
        opened = await self.sql.open(ctx)
        if isinstance(opened, ToolOutput):
            return opened
        import duckdb

        session, index = opened
        if path not in self.wrote and any(path == (session.root / rel).resolve() for rel in index.files):
            return fail(f"{args.path} is an indexed document", "Write to a new file.", argument="path")
        try:
            written = await self.sql.timed(session, lambda: session.export(args.source, path), args.timeout_s)
        except duckdb.Error as e:
            return _failed(e, sorted(t.name for entry in index.files.values() for t in entry.tables))
        if isinstance(written, ToolOutput):
            return written
        if isinstance(written, str):
            return fail(f"sql_export writes one SELECT's result, not {written}", "Give one SELECT query or a table.")
        self.wrote.add(path)
        return ToolOutput(content=f"Wrote {written:,} rows to {args.path}.")


@dataclass
class Shown:
    """One statement's result: its columns and first rows (``limit + 1`` at most), the table a SELECT's result was
    kept as (name, rows; the name is empty when there were too many rows to keep), and, for a workspace table's
    DESCRIBE, the index's profile shown above its first rows."""

    statement: str
    columns: list[str]
    rows: list[tuple[object, ...]]
    kept: tuple[str, int] | None = None
    profile: str = ""


def _show(one: Shown, limit: int, batch: bool) -> str:
    head = f"-- {' '.join(one.statement.split())[:120]}\n" if batch else ""
    if one.profile:
        head += one.profile + "\nfirst rows:\n"
    if not one.columns:
        return head + "(no result)"
    shown = one.rows[:limit]
    text = "| " + " | ".join(one.columns) + " |\n|" + "---|" * len(one.columns)
    text += "".join("\n| " + " | ".join(_cell(v) for v in row) + " |" for row in shown) or "\n(no rows)"
    if one.kept is not None:
        name, count = one.kept
        if name:
            text += f"\n[kept as {name}: {count:,} rows" + (f", {len(shown)} shown]" if count > len(shown) else "]")
        else:
            text += (
                f"\n[{len(shown)} rows shown of more than {KEEP_ROWS:,}, too many to keep: aggregate or filter in "
                "the query]"
            )
    elif len(one.rows) > limit:
        text += f"\n[{limit} rows shown; there are more. Narrow the query or raise limit.]"
    return head + text


def _failed(e: Exception, tables: list[str], before: str = "", which: str = "the query") -> ToolOutput:
    import duckdb

    # DuckDB's whole message: its "Did you mean", candidate columns and the caret under the fault are the fix.
    message = str(e).strip()
    message = message if len(message) <= SQL_ERROR_CHARS else message[:SQL_ERROR_CHARS] + " [...]"
    # An unknown table: list the workspace's, unless DuckDB already named the one meant (not a system table).
    meant = _MEANT.search(message)
    unknown_table = isinstance(e, duckdb.CatalogException) and not (
        meant and (meant.group(1).rsplit(".", 1)[-1] in tables or meant.group(1).startswith("res_"))
    )
    return fail(
        f"{which} failed: {message}",
        (f"Tables: {', '.join(tables)}. " if unknown_table else "") + "DESCRIBE <table> shows its columns.",
        argument="query",
        content=before,
    )


class SqlSession:
    """The `sql` tool's DuckDB for one session. It is in memory, so the results it keeps (res_1, res_2, ...) last as
    long as the session and leave no file behind; past the query memory share (`documents.query_memory`) DuckDB
    spills to ``<home>/sql-tmp/<pid>``, which it empties as it goes and this removes on close, and which a later
    session removes if this one's process was killed. The workspace's tables are attached read-only for each query
    and detached after, so an index build can rewrite them between queries. It reaches files under the indexed root
    (and the working directory, which the views' relative paths need) only, and cannot change that."""

    def __init__(self, home: Path, root: Path) -> None:
        import duckdb

        from intern.extensions.workspace import documents

        self.home, self.root, self.count = home, root, 0
        spills = home / SPILL_DIR
        _sweep(spills)
        self.spill = spills / str(os.getpid())
        self.spill.mkdir(parents=True, exist_ok=True)
        config: dict[str, str | bool | int | float | list[str]] = {"temp_directory": str(self.spill)}
        if memory := documents.query_memory():
            config["memory_limit"] = f"{memory // 2**20}MB"
        self.con = duckdb.connect(":memory:", config=config)
        lit = documents.literal
        self.con.execute(f"SET file_search_path = {lit(root)}")
        self.con.execute(f"SET allowed_directories = [{lit(f'{root}/')}, {lit(f'{self.spill}/')}, './']")
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")
        self._closer = weakref.finalize(self, _close, self.con, self.spill)

    def run(
        self, query: str, limit: int, profiles: Mapping[str, TableInfo]
    ) -> tuple[list[Shown], duckdb.Error | None] | str:
        """Each statement's result in turn, up to the first that fails (with its error), or the refused statement
        kinds, before any runs. A SELECT's result is kept as res_N; EXPLAIN's and a scratch statement's are not. A
        DESCRIBE of a workspace table shows the index's profile of it and its first rows."""
        import duckdb

        statements = self.con.extract_statements(query)
        kinds = {statement.type.name for statement in statements}
        refused = kinds - QUERY_STATEMENTS - SCRATCH_STATEMENTS
        if any(_SECRET.match(statement.query) for statement in statements):
            refused.add("CREATE SECRET")
        if refused:
            return ", ".join(sorted(refused))
        shown: list[Shown] = []
        with self._attached():
            for statement in statements:
                text = _body(statement.query)
                try:
                    shown.append(self._one(text, statement.type.name, limit, profiles))
                except duckdb.Error as e:
                    return shown, e
        return shown, None

    def export(self, source: str, path: Path) -> int | str:
        """Write ``source`` (a table's name, or one SELECT) to ``path`` as CSV or Parquet: the rows written, or the
        statement kinds refused."""
        from intern.extensions.workspace import documents

        if _NAME.fullmatch(source.strip()):
            body = f"SELECT * FROM {source.strip()}"
        else:
            statements = self.con.extract_statements(source)
            kinds = [statement.type.name for statement in statements]
            if kinds != ["SELECT"]:
                return ", ".join(sorted(set(kinds) - {"SELECT"})) or "several statements"
            body = _body(statements[0].query)
        form = "(FORMAT parquet)" if path.suffix.lower() == ".parquet" else "(FORMAT csv, HEADER)"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._attached():
            found = self.con.execute(f"COPY (SELECT * FROM ({body}\n)) TO {documents.literal(path)} {form}").fetchone()
        return int(found[0]) if found else 0

    @contextmanager
    def _attached(self) -> Iterator[None]:
        """The workspace's tables, attached read-only for one piece of work (an index build can rewrite them
        between two), after the session's own in its search path."""
        from intern.extensions.workspace import documents

        self.con.execute(f"ATTACH {documents.literal(self.home / documents.TABLES_DB)} AS workspace (READ_ONLY)")
        try:
            self.con.execute("SET search_path = 'memory.main,workspace.main'")
            yield
        finally:
            self.con.execute("USE memory")
            self.con.execute("DETACH workspace")

    def _one(self, text: str, kind: str, limit: int, profiles: Mapping[str, TableInfo]) -> Shown:
        described = _DESCRIBE.match(text)
        info = profiles.get(described.group(2)) if described else None
        if info is not None and info.profile and not self._own(info.name):
            cursor = self.con.execute(f'SELECT * FROM workspace.main."{info.name}" LIMIT {DESCRIBE_ROWS}')
            profile = f"table {info.name} ({info.source}): {info.rows:,} rows × {len(info.columns)} columns\n"
            return Shown(
                text, [str(d[0]) for d in cursor.description or []], cursor.fetchall(), None, profile + info.profile
            )
        if kind != "SELECT":
            cursor = self.con.execute(text)
            return Shown(text, [str(d[0]) for d in cursor.description or []], cursor.fetchmany(limit + 1))
        name = f"res_{self.count + 1}"
        # Closed on a line of its own, so a comment that ends the query does not swallow the parenthesis (and the
        # query's lines keep their numbers in DuckDB's errors).
        self.con.execute(f'CREATE TABLE memory.main."{name}" AS SELECT * FROM ({text}\n) LIMIT {KEEP_ROWS + 1}')
        found = self.con.execute(f'SELECT count(*) FROM "{name}"').fetchone()
        rows = int(found[0]) if found else 0
        cursor = self.con.execute(f'SELECT * FROM "{name}" LIMIT {limit + 1}')
        columns, first = [str(d[0]) for d in cursor.description or []], cursor.fetchall()
        if rows > KEEP_ROWS:
            self.con.execute(f'DROP TABLE memory.main."{name}"')
            return Shown(text, columns, first, ("", rows))
        self.count += 1
        return Shown(text, columns, first, (name, rows))

    def _own(self, name: str) -> bool:
        """Whether the session's scratch database has a table or view of this name (it comes first in the path)."""
        found = self.con.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE database_name = 'memory' AND table_name = $1 "
            "UNION ALL SELECT count(*) FROM duckdb_views() WHERE database_name = 'memory' AND view_name = $1",
            [name],
        ).fetchall()
        return any(n for (n,) in found)

    def close(self) -> None:
        self._closer()


def _body(statement: str) -> str:
    """A statement as DuckDB splits it off a query, without the semicolon that ends it and the comments after."""
    import duckdb

    raw = statement.encode()  # the tokens' positions are in bytes
    tokens = [at for at, kind in duckdb.tokenize(statement) if kind != duckdb.token_type.comment]
    if tokens and raw[tokens[-1] : tokens[-1] + 1] == b";":
        return raw[: tokens[-1]].decode().strip()
    return statement.strip()


def _close(con: duckdb.DuckDBPyConnection, spill: Path) -> None:
    con.close()
    shutil.rmtree(spill, ignore_errors=True)


def _sweep(spills: Path) -> None:
    """Remove the spill folders of processes that are gone (killed before they could)."""
    for folder in spills.glob("*") if spills.is_dir() else ():
        if folder.name.isdigit() and int(folder.name) != os.getpid():
            try:
                os.kill(int(folder.name), 0)
            except ProcessLookupError:
                shutil.rmtree(folder, ignore_errors=True)
            except OSError:
                pass  # alive under another user


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")[:200]


class WorkspaceReadArgs(ReadArgs):
    pages: str | None = Field(default=None, description="For a PDF or workbook: the pages to show, e.g. 3 or 3-5.")
    tab: str | None = Field(default=None, description="For a workbook: the tab to show, by name.")


class WorkspaceRead(ReadTool):
    description = (
        ReadTool.description + " PDFs, workbooks and Parquet files are shown as text, one page per PDF page or "
        "workbook tab (marked --- Page N ---); pick with pages or tab. Word documents are shown as Markdown."
    )
    Args = WorkspaceReadArgs

    def __init__(self, store: Store) -> None:
        self.store = store

    async def run(self, args: ReadArgs, ctx: ToolContext) -> ToolOutput:
        from intern.locate.index import DERIVED_DIR, pages_of

        path = ctx.resolve(args.path)
        if (
            not isinstance(args, WorkspaceReadArgs)
            or path.suffix.lower() not in DOCUMENT_SUFFIXES
            or not path.is_file()
        ):
            return await super().run(args, ctx)
        try:
            index = await self.store.ready(ctx.cwd)
        except ImportError as e:
            return fail(f"reading {path.suffix} files needs {e.name}", f"{INSTALL}.")
        except NoIndex:
            return fail(
                f"{args.path} has no derived text yet", "call index_files with its folder first.", argument="path"
            )
        try:
            rel = path.resolve().relative_to(self.store.root or ctx.cwd).as_posix()
        except ValueError:
            return fail(
                f"{args.path} is outside the indexed folder, so it has no derived text",
                "call index_files with its folder, or inspect it with bash.",
                argument="path",
            )
        entry = index.files.get(rel)
        if entry is None or not entry.derived:
            return fail(
                f"{args.path} could not be indexed (see the workspace messages)",
                "Inspect it with bash, or fix the file and try again.",
                argument="path",
            )
        derived = (index.home / DERIVED_DIR / entry.derived).read_text()
        if entry.unit == "line":  # a Word document: its Markdown, paged by offset and limit like any text
            return page_lines(args.path, derived.splitlines(), args, ctx.max_output_chars, resources=[])
        pages = pages_of(derived)
        chosen = list(range(1, len(pages) + 1))
        if args.tab is not None:
            tabs = {
                c.title.removeprefix("tab ").split(" → ")[0].lower(): c.start for c in entry.chunks if c.kind == "table"
            }
            if args.tab.lower() not in tabs:
                return fail(
                    f"{args.path} has no tab {args.tab!r}", f"Tabs: {', '.join(tabs) or 'none'}.", argument="tab"
                )
            chosen = [tabs[args.tab.lower()]]
        if args.pages is not None:
            chosen = [n for n in _pages(args.pages) if 1 <= n <= len(pages)]
            if not chosen:
                return fail(
                    f"{args.path} has {len(pages)} pages; {args.pages!r} is not among them",
                    f"Ask for pages between 1 and {len(pages)}.",
                    argument="pages",
                )
        if entry.processor == "pdf" and args.pages is not None and len(chosen) <= OCR_READ_PAGES:
            pages = await self._ocr_scans(path, index.home, entry.derived, pages, chosen)
        lines = [line for n in chosen for line in (f"--- Page {n} ---", *pages[n - 1].splitlines(), "")]
        return page_lines(args.path, lines, args, ctx.max_output_chars, resources=[])

    async def _ocr_scans(self, path: Path, home: Path, derived: str, pages: list[str], chosen: list[int]) -> list[str]:
        """``pages`` with the chosen scans OCR'd, unless the OCR lost much of the text layer (`OCR_KEEP`), and the
        derived text updated, so the next read (and the search's reading of the page) has the OCR."""
        from intern.extensions.workspace import documents
        from intern.locate.index import DERIVED_DIR, PAGE_MARK

        try:
            scans = await asyncio.to_thread(documents.unread_scans, path, home, chosen)
            found = await asyncio.to_thread(documents.ocr_pages, path, home, scans) if scans else {}
        except Exception:  # OCR is a bonus: without it the page shows its text layer
            return pages
        keep = documents.OCR_KEEP
        better = {n: text for n, text in found.items() if len(text.strip()) >= keep * len(pages[n - 1].strip())}
        if not better:
            return pages
        pages = [better.get(n, text) for n, text in enumerate(pages, 1)]
        text = "\n\n".join(f"{PAGE_MARK.format(n=n)}\n{page}" for n, page in enumerate(pages, 1))
        await asyncio.to_thread((home / DERIVED_DIR / derived).write_text, text + "\n")
        return pages


def _tab_name(title: str) -> str:
    return title.removeprefix("tab ").split(" → ")[0].lower()


def _pages(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if m := _RANGE.match(part):
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            out.extend(range(a, b + 1) if a <= b else [])
    return out
