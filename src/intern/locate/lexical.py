"""A lexical index of the index's chunks, so the walk can start where the question's words are.

Each chunk of every indexed file (a function, a method, a section, a key, a window) is one passage, titled by its
file's path and the chunk's title; a file without chunks is one passage. They live in an SQLite FTS5 table beside the
index (`<home>/lexical.sqlite`), brought up to date per file before a search, so no model runs and no embedding is
needed. Identifiers are split before FTS5 sees them (`RevenueAudit` gives revenue, audit; its `unicode61` tokenizer
already cuts at underscores and slashes), so a question's word meets the same word inside code and in paths.

This is the one BM25 for both searches; each index has its own table. `decision_code` uses the hits to steer the walk
(`search.Cascade.descend`), never to rank: what it ranks is what System 1 judged. `decision_files`' union
(`extensions/workspace/search.py`) indexes its own passages (`PassageFn`), takes the BM25 top 100 for System 1 to
judge and weighs its expansion terms by the document frequencies here. Each search keeps its own question -> query
words (`query` here; `match` there), as each was measured.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from collections.abc import Callable, Collection
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from intern.locate import grouping
from intern.locate.index import DERIVED_DIR, ignore, pages_of

if TYPE_CHECKING:
    from intern.locate.index import FileEntry, Index

LEXICAL_DB = "lexical.sqlite"
#: Bumped when what a passage holds changes: an older table is rebuilt.
SCHEMA = 1
#: Characters of a chunk's text a passage holds.
PASSAGE_CHARS = 8000
#: Distinct words of a question that go into its query.
QUERY_TERMS = 40
#: Where an identifier splits into words beyond what `unicode61` cuts: a capital after a lower-case letter or digit,
#: and the last capital of an acronym before a capitalised word (`HTTPServer` -> HTTP Server).
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TERM = re.compile(r"[^\W_]+")
_LOCKS: dict[Path, threading.Lock] = {}
#: A file's passages, (chunk key, title, text) each: `passages` for the code index; an extension may index others
#: (the workspace: leaf chunks, and each table by its columns and profile).
PassageFn = Callable[["Index", Path, "FileEntry"], list[tuple[str, str, str]]]


class Hit(BaseModel):
    """A passage the question's words match, best first."""

    path: str
    key: str | None
    rank: int


def words(text: str) -> str:
    """``text`` with its identifiers split, for FTS5 to index or match."""
    return _CAMEL.sub(" ", text)


def quote(term: str) -> str:
    """``term`` as an FTS5 string: its words in order, a double quote in it doubled."""
    return '"' + term.replace('"', '""') + '"'


def query(question: str) -> str:
    """The FTS5 expression for a question: any of its words that are not stop words (BM25 weighs the rarer ones)."""
    terms = [t for token in grouping.tokens(words(question)) for t in _TERM.findall(token) if len(t) > 2]
    return " OR ".join(quote(t) for t in list(dict.fromkeys(terms))[:QUERY_TERMS])


class Lexical:
    """The FTS5 table of the index's passages, under the index's home."""

    def __init__(self, index: Index, root: Path, passages: PassageFn | None = None) -> None:
        self.index, self.root = index, root
        self.path = index.home / LEXICAL_DB
        self.passages = passages
        self.total = 0
        """Passages in the table, as `open` found it: the N of `idf`."""
        self._df: dict[str, int] = {}

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=120, check_same_thread=False)

    def open(self) -> sqlite3.Connection:
        """The table brought up to date, and a connection to it with a vocabulary view for the term statistics."""
        self.update()
        con = self.connect()
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS passage_vocab USING fts5vocab(passage_fts, 'row')")
        self.total = con.execute("SELECT count(*) FROM passages").fetchone()[0]
        return con

    def update(self) -> None:
        """Add the passages of files new or changed since the table was written, drop those of files gone. The text
        is indexed, not kept (`contentless_delete`): the table holds where each passage is, the index the rest."""
        ignore(self.root, self.index.home, LEXICAL_DB + "*")  # the table and its journal stay out of git, as the index
        with _LOCKS.setdefault(self.path, threading.Lock()), self.connect() as con:
            if con.execute("PRAGMA user_version").fetchone()[0] != SCHEMA:
                con.executescript(
                    "DROP TABLE IF EXISTS passage_fts; DROP TABLE IF EXISTS passages;DROP TABLE IF EXISTS files;"
                )
            con.executescript(
                "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, stamp TEXT);"
                "CREATE TABLE IF NOT EXISTS passages(id INTEGER PRIMARY KEY, path TEXT, key TEXT);"
                "CREATE INDEX IF NOT EXISTS passages_path ON passages(path);"
                "CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(title, body, content='', "
                "contentless_delete=1, tokenize='unicode61 remove_diacritics 2');"
                f"PRAGMA user_version = {SCHEMA};"
            )
            known = dict(con.execute("SELECT path, stamp FROM files").fetchall())
            want = {p: f"{e.size}:{e.mtime_ns}:{e.processor}" for p, e in self.index.files.items()}
            gone = [p for p, stamp in known.items() if want.get(p) != stamp]
            new = [p for p, stamp in want.items() if known.get(p) != stamp]
            if not gone and not new:
                return
            con.execute("BEGIN IMMEDIATE")
            for p in gone:
                con.execute("DELETE FROM passage_fts WHERE rowid IN (SELECT id FROM passages WHERE path = ?)", (p,))
                con.execute("DELETE FROM passages WHERE path = ?", (p,))
                con.execute("DELETE FROM files WHERE path = ?", (p,))
            for p in new:
                for key, title, body in (self.passages or passages)(self.index, self.root, self.index.files[p]):
                    cur = con.execute("INSERT INTO passages(path, key) VALUES (?, ?)", (p, key))
                    con.execute(
                        "INSERT INTO passage_fts(rowid, title, body) VALUES (?, ?, ?)",
                        (cur.lastrowid, words(title), words(body)),
                    )
                con.execute("INSERT INTO files(path, stamp) VALUES (?, ?)", (p, want[p]))
            con.commit()

    def hits(self, question: str, scope: str | None, limit: int) -> list[Hit]:
        """The BM25 top ``limit`` passages for ``question`` under ``scope`` (a title weighs twice its text)."""
        expression = query(question)
        if not expression:
            return []
        with self.connect() as con:
            rows = self.search(con, expression, (), limit, scope)
        return [Hit(path=path, key=key or None, rank=n) for n, (_, path, key) in enumerate(rows)]

    def search(
        self, con: sqlite3.Connection, expression: str, seen: Collection[int], limit: int, scope: str | None
    ) -> list[tuple[int, str, str]]:
        """The BM25 top ``limit`` (id, path, key) for the FTS5 ``expression`` under the folder or file ``scope``,
        without the passages in ``seen`` (a title weighs twice its text). The scope is a path, not a pattern: `_`
        and `%` in it are characters, and a sibling it is a prefix of (`data` of `data_2024`) is outside it."""
        if not expression:
            return []
        return con.execute(
            "SELECT p.id, p.path, p.key FROM passage_fts JOIN passages p ON p.id = passage_fts.rowid "
            "WHERE passage_fts MATCH ? AND passage_fts.rowid NOT IN (SELECT value FROM json_each(?)) "
            "AND (? IS NULL OR p.path = ? OR substr(p.path, 1, length(?) + 1) = ? || '/') "
            "ORDER BY bm25(passage_fts, 2.0, 1.0) LIMIT ?",
            (expression, json.dumps(sorted(seen)), scope, scope, scope, scope, limit),
        ).fetchall()

    def df(self, con: sqlite3.Connection, term: str) -> int:
        """Passages holding ``term`` (a word, or words that must come together); needs `open`'s connection."""
        if term not in self._df:
            if " " in term:
                row = con.execute("SELECT count(*) FROM passage_fts WHERE passage_fts MATCH ?", (quote(term),))
            else:
                row = con.execute("SELECT doc FROM passage_vocab WHERE term = ?", (term,))
            found = row.fetchone()
            self._df[term] = int(found[0]) if found else 0
        return self._df[term]

    def idf(self, con: sqlite3.Connection, term: str) -> float:
        df = self.df(con, term)
        return math.log(1 + (self.total - df + 0.5) / (df + 0.5))


def passages(index: Index, root: Path, entry: FileEntry) -> list[tuple[str, str, str]]:
    """(key, title, text) for each chunk of a file, or the file itself when it has none. A chunk holds its
    sub-chunks' text too (a class its methods): the overlap only repeats a file among the hits, and without it a
    class's own docstring and fields would be in no passage."""
    try:
        if entry.derived:
            text = (index.home / DERIVED_DIR / entry.derived).read_text(errors="replace")
        else:
            text = (root / entry.path).read_text(errors="replace")
    except OSError:
        return []
    if not entry.chunks:
        return [("", entry.path, text[:PASSAGE_CHARS])]
    units = pages_of(text) if entry.unit == "page" else text.splitlines()
    sep = "\n\n" if entry.unit == "page" else "\n"
    return [
        (c.key, f"{entry.path} {c.title}", sep.join(units[c.start - 1 : c.end])[:PASSAGE_CHARS]) for c in entry.chunks
    ]
