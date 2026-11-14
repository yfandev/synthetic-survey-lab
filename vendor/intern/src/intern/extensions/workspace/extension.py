"""The `workspace` extension: a folder of office files as a tree System 1 can search.

Beside the locate engine's text processors it indexes PDFs (OCR for scanned pages with the `ocr` extra), Word
documents and workbooks, keeps their text under `.decision_workspace/derived/` and every spreadsheet tab, CSV and
Parquet file as a table in `.decision_workspace/tables.duckdb`. The agent builds the index where the documents are
(`index_files`, given a folder: OCR takes minutes, and in a repository with one folder of documents that folder is
the one to index; the working directory's own index, if it has one, is loaded at the start). Then `decision_files`
over the index, `sql` over the tables, and a `read` that shows documents as text (`pages=`, `tab=`). The questions
ask for what the task needs, its material: the code wording ranks a single document about the subject above a set of
raw-material files. Needs `intern-agent[workspace]`.

Without an index at the start, the note names the folders that hold documents, by count and whether each is indexed;
where the working directory holds none (a code repository) nothing is pinned.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from intern import hooks as h
from intern.decisions import ChoiceAnswer, ChoiceQuestion
from intern.extensions.workspace.documents import QUERY_MEMORY_SHARE, office_processors, query_memory
from intern.extensions.workspace.tools import INSTALL, Sql, SqlExport, WorkspaceRead
from intern.locate.index import DOCUMENT_SUFFIXES, SAMPLE_FILES, sample
from intern.locate.search import Questions
from intern.locate.store import DecisionSearch, IndexTool, Locate
from intern.messages import ToolCall
from intern.needs import Needs
from intern.tools.base import Tool

if TYPE_CHECKING:
    from intern.decisions import DecisionClient
    from intern.locate.index import Index
    from intern.locate.search import SearchResult

#: A table file is large when its data, uncompressed, is over this share of the machine's memory: loaded whole into
#: python it takes a few times that, more than the share one `sql` query may hold (`documents.QUERY_MEMORY_SHARE`).
LARGE_TABLE_SHARE = 1 / 32
#: Characters of a script a command runs that System 1 sees beside the command.
SCRIPT_CHARS = 6000
_SCRIPT = re.compile(r"[\w./-]+\.py\b")
#: A command that lists files by pattern may reach a table it never names.
_LISTS_FILES = re.compile(r"glob|rglob|os\.walk|listdir|iterdir|\*\*|\bfind\b")
#: Folders the note names that hold documents, the fullest first.
FOLDERS_SHOWN = 8
LOADS_QUESTION = ChoiceQuestion(
    instructions=(
        "Would running this command read all the rows of one or more of the large table files listed into a "
        "program's memory, even only some of their columns (for example pandas read_parquet or read_csv, pyarrow "
        "read_table, polars read_*), rather than reading only their schema, their size or a few rows?"
    ),
    criteria={
        "loads": "yes: it reads all the rows of at least one listed file into memory",
        "other": "no: it does not read a listed file, or reads only its schema, size or a few rows",
    },
)

OFFICE = Questions(
    node=(
        "Does this folder or group of files hold something the question needs: a file it is about, works on, or "
        "would change?"
    ),
    file=(
        "Is this file one the question needs: a file it is about, works on, or would change (not one that merely "
        "mentions, calls or tests that)?"
    ),
    chunk="Is the part shown as [{id}] something the question needs: what it is about, works on, or would change?",
    leaf="Is this text something the question needs: what it is about, works on, or would change?",
    compare=(
        "Which one of these files does the question need most: the one it is about, works on, or would change (not "
        "one that merely mentions, calls or tests that)?"
    ),
)


class IndexFiles(IndexTool):
    name = "index_files"
    description = (
        "Index a folder of documents (PDFs, Word, spreadsheets, CSV, Parquet and text) so decision_files, sql and read "
        "can use them. Give the folder that holds the documents; scanned pages are OCRed, so it can take minutes. A "
        "folder indexed before loads in seconds."
    )


class DecisionFiles(DecisionSearch):
    name = "decision_files"
    description = (
        "Find the places in a folder of documents (PDFs, Word, spreadsheets, CSV, Parquet, text) that answer a "
        "question (document sections and pages, spreadsheet tabs and tables), ranked, each with the call that opens it "
        "and the text of the first ones. Needs index_files on that folder first."
    )
    inline = (3000, 800, 800)
    trace = True

    async def search(
        self, index: Index, root: Path, question: str, decisions: DecisionClient, scope: str | None
    ) -> SearchResult:
        """The union (`search.py`): the walk's chunks and the BM25 top 100 judged together, one expansion round,
        drawn live by the workspace's view (`view.py`) from its trace (`trace.py`)."""
        from intern.extensions.workspace.search import union

        return await union(index, root, question, decisions, self.asks, scope=scope, trace=self._trace_sink())


class Workspace(Locate):
    name: ClassVar[str] = "workspace"
    tui_view: ClassVar[str | None] = "intern.extensions.workspace.view:DecisionFilesView"
    needs: ClassVar[Needs | None] = Needs(
        title="The workspace",
        extra="workspace",
        modules=("numpy", "networkx", "duckdb", "pymupdf", "openpyxl", "docx"),
    )
    search_tool = DecisionFiles
    processors = staticmethod(office_processors)
    asks: ClassVar[Questions] = OFFICE
    #: Not under `.intern/`: NDI keeps its own workspace state there, and the name says which index this is.
    folder: ClassVar[str] = ".decision_workspace"
    moved_from: ClassVar[str | None] = ".drex_workspace"
    auto: ClassVar[bool] = False
    install: ClassVar[str] = INSTALL
    note: ClassVar[str] = (
        "Documents (PDFs, Word, spreadsheets, CSV, Parquet) are searchable once their folder is indexed with "
        "`index_files`. Then `decision_files` finds the places that answer a question (sections and pages, tabs and "
        "tables) when you do not know where to look or the exact words to grep for, each with the call that opens "
        "it; `sql` queries every spreadsheet tab, CSV and Parquet file; `read` shows PDFs, Word documents and "
        "workbooks as text (`pages=`, `tab=`)."
    )

    indexed_note: ClassVar[str] = (
        "The documents in the working directory (PDFs, Word, spreadsheets, CSV, Parquet) are indexed already. To "
        "answer from them:\n"
        "1. Call `decision_files` with what you need and the whole question: it returns the places that hold it, best "
        "first, with the text of the best ones. When that text states the answer, answer.\n"
        "2. Read the pages it names, and the pages around them, with `read` (`pages=14` or `pages=14-16`; `tab=` "
        "for a workbook): it shows each page's best text, OCR included. Do not extract PDFs yourself (pdftotext, "
        "OCR, PDF libraries): you would get the same pages with worse text.\n"
        "3. For a part no page covers yet (another year, an entity, a line item), call `decision_files` again with "
        "just that part.\n"
        "4. Spreadsheet tabs, CSV and Parquet files are tables: work on them with `sql` only. It reads the files in "
        "place over every row; filter, join, total and check there, and build each next step on the result it kept "
        "(res_1, res_2, ...); `sql_export` writes a result's rows to a file. Do not read the table files in python "
        "(pandas, pyarrow, polars): a large one does not fit in memory, and such a command is refused. Do arithmetic "
        "on figures `sql` gave you with python3 -c, not in your head.\n"
        "Cite each figure's file and page."
    )

    def __init__(self) -> None:
        super().__init__()
        self._sizes: dict[str, int] = {}

    def unindexed(self, cwd: Path) -> str | None:
        """The note and the folders of the working directory that hold documents; nothing where it holds none."""
        files = sample(cwd)
        tops = Counter(
            rel.split("/", 1)[0] + "/" if "/" in rel else "./"
            for rel in files
            if Path(rel).suffix.lower() in DOCUMENT_SUFFIXES
        )
        if not tops:
            return None
        homes = [h for h in (self.folder, self.moved_from) if h]
        said = [
            f"{top} {n}" + (" (indexed)" if any((cwd / top / h).is_dir() for h in homes) else "")
            for top, n in tops.most_common(FOLDERS_SHOWN)
        ]
        more = f", and {len(tops) - FOLDERS_SHOWN} more folders" if len(tops) > FOLDERS_SHOWN else ""
        counted = f" (of its first {SAMPLE_FILES:,} files)" if len(files) == SAMPLE_FILES else ""
        return f"{self.note}\nDocuments in the working directory{counted}: {', '.join(said)}{more}."

    def tools(self) -> list[Tool]:
        sql = Sql(self.store)
        return [
            IndexFiles(self.store),
            DecisionFiles(self.store, self.asks),
            sql,
            WorkspaceRead(self.store),
            SqlExport(sql),
        ]

    async def tool_call(self, call: ToolCall, ctx: h.RunContext) -> h.ToolCallDecision | None:
        """Refuse a shell command that would load one of the workspace's large table files whole into a program's
        memory: `sql` reads it in place. The command (and a script it runs) is checked against the index's large
        tables first, so System 1 is asked only when it touches one, and judges whether it loads it."""
        command = call.arguments.get("command") if call.name == "bash" else None
        root = self.store.root
        index = self.store.indexes.get(root) if root is not None else None
        if not isinstance(command, str) or index is None or root is None or (share := query_memory()) is None:
            return None
        limit = share / QUERY_MEMORY_SHARE * LARGE_TABLE_SHARE  # of the machine's memory
        large = {
            rel: entry
            for rel, entry in index.files.items()
            if entry.processor == "table" and self._uncompressed(root, rel) > limit
        }
        if not large:
            return None
        text = command + "".join(_script(Path(ctx.cwd), name) for name in _SCRIPT.findall(command))
        listed = _LISTS_FILES.search(text)
        touched = {
            rel: e
            for rel, e in large.items()
            if Path(rel).name in text or Path(rel).stem in text or (listed and Path(rel).suffix in text)
        }
        if not touched:
            return None
        lines = [
            f"- {rel}: {e.tables[0].rows:,} rows, {self._uncompressed(root, rel) / 2**30:.1f} GB uncompressed"
            for rel, e in touched.items()
            if e.tables
        ]
        state = f"Command:\n{text[: len(command) + SCRIPT_CHARS]}\n\nLarge table files in the workspace:\n" + "\n".join(
            lines
        )
        answers = await ctx.decisions.ask(
            "workspace_table_load", state, {"loads": LOADS_QUESTION}, turn=ctx.turn, subject=command[:200]
        )
        answer = answers.get("loads")
        if not isinstance(answer, ChoiceAnswer) or answer.choice != "loads":
            return None
        names = ", ".join(f"{rel} as {e.tables[0].name}" for rel, e in touched.items() if e.tables)
        return h.Refuse(
            reason=(
                f"this would load a large table into python's memory ({', '.join(touched)}: "
                + "; ".join(line.split(": ", 1)[1] for line in lines)
                + "), more than a command may hold"
            ),
            next_action=(
                f"Query it in place with sql ({names}): filter, join and total there, and build the next step on the "
                "result it keeps (res_1, res_2, ...). Use python only on the figures sql returns."
            ),
        )

    def _uncompressed(self, root: Path, rel: str) -> int:
        """The table file's data size uncompressed: a Parquet file's from its metadata, a CSV's is its size."""
        if rel not in self._sizes:
            import duckdb

            path = root / rel
            size = 0
            try:
                size = path.stat().st_size
                if path.suffix.lower() == ".parquet":
                    with duckdb.connect() as con:
                        found = con.execute(
                            "SELECT sum(total_uncompressed_size) FROM parquet_metadata(?)", [str(path)]
                        ).fetchone()
                    size = int(found[0]) if found and found[0] is not None else size
            except (OSError, duckdb.Error):
                pass
            self._sizes[rel] = size
        return self._sizes[rel]


def _script(cwd: Path, name: str) -> str:
    """The text of a script the command runs, when it is a file under the working directory."""
    path = cwd / name
    try:
        return f"\n\n--- {name} ---\n" + path.read_text(errors="replace")[:SCRIPT_CHARS] if path.is_file() else ""
    except OSError:
        return ""
