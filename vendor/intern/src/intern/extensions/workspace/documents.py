"""Processors for professional files: PDFs, spreadsheets, tables and images.

Each imports its library on use so the harness runs without `intern-agent[workspace]`; a file whose library is
missing is skipped and the index says so once. Spreadsheets and tables are loaded into `.workspace/tables.duckdb`
for the `sql` tool; their text unit is a rendering of the head rows and a column profile.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import re
import statistics
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from intern.locate.processors import (
    Chunk,
    Derivative,
    KeyedProcessor,
    MarkdownProcessor,
    Processor,
    PythonProcessor,
    TableInfo,
    WindowProcessor,
    sections,
    windows,
)
from intern.tools.images import IMAGE_SUFFIXES

if TYPE_CHECKING:
    import duckdb

    from intern.locate.index import FileEntry

TABLES_DB = "tables.duckdb"
#: A PDF page with fewer characters of text layer than this is scanned, and OCR'd when the OCR extra is installed.
OCR_MIN_CHARS = 20
OCR_DPI = 200
OCR_DIR = "ocr"
#: A scan's OCR replaces its text layer when it has at least this share of the text layer's characters. The text
#: layer of a scan is an older OCR's: on scanned bulletins it reads 49 as `1*9` and drops whole columns.
OCR_KEEP = 0.8
#: The image must cover this share of the page for the page to count as scanned.
SCAN_COVER = 0.5
#: The steepest tilt of a scan's lines (in thousandths, dy/dx) that rows are levelled for.
TILT_MAX = 40
#: Pages of a PDF without an outline are offered in windows of this many, so a choice has few candidates.
PAGE_WINDOW = 10
#: Rows of a table its text rendering shows.
HEAD_ROWS = 20
#: Tables wider than this get column types only, not ranges and distinct counts.
PROFILE_COLUMNS = 64
SAMPLE_ROWS = 6
#: A column with at most this many distinct values lists them in the profile, as NDI's table profile does: a search
#: for a status, a type or a code then finds the table that holds it.
PROFILE_VALUES = 12
_UNSAFE = re.compile(r"\W+")
#: The share of the machine's memory one `sql` query may hold before DuckDB spills to disk (its default is 80%,
#: right for a database server, not for a tool inside an agent beside everything else the user runs).
QUERY_MEMORY_SHARE = 1 / 8
#: A column without a header cell, as DuckDB names it in a CSV and pandas in a file it saved.
_UNNAMED = re.compile(r"column\d+|Unnamed: \d+")
#: A CSV or Parquet file of at most this many rows whose columns come out mostly unnamed (a title above the header,
#: as in a sheet saved by pandas) is loaded like a workbook tab: its real header found, the rows above it a caption.
RESHAPE_ROWS = 100_000


class PageOcr:
    """OCR of a PDF's pages as text lines (`lines`), with every page's boxes kept in ``<home>/ocr/`` by the file's
    content: a rebuild after a processor change, a copy with new mtimes or a later `read` OCRs no page twice (one
    pass over 7.8k scanned pages takes an hour). The engine starts on the first page not kept; None from
    `make` when a page would need it and `intern-agent[ocr]` is not installed."""

    def __init__(self, path: Path, home: Path) -> None:
        self.path, self.home = path, home
        self._folder: Path | None = None
        self._engine: Any = None

    @property
    def folder(self) -> Path:
        if self._folder is None:
            digest = hashlib.sha1(self.path.read_bytes()).hexdigest()
            self._folder = self.home / OCR_DIR / f"{digest}-{OCR_DPI}"
        return self._folder

    def available(self) -> bool:
        try:
            import cv2  # noqa: F401
            import rapidocr_onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(self, page: Any) -> str | None:
        """The page's text, or None when it is not kept and there is no OCR to make it."""
        kept = self.folder / f"{page.number + 1}.json"
        if kept.exists():
            boxes = [tuple(b) for b in json.loads(kept.read_text())]
        elif not self.available():
            return None
        else:
            boxes = self._recognize(page)
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.write_text(json.dumps(boxes))
        # The recognizer is Chinese-English: a full stop in a figure comes back as the ideographic one.
        return lines(boxes).replace("。", ".")  # type: ignore[arg-type]

    def _recognize(self, page: Any) -> list[Box]:
        import numpy as np

        if self._engine is None:
            import cv2
            from rapidocr_onnxruntime import RapidOCR

            from intern.locate.index import PROCESSING_THREADS

            cv2.setNumThreads(1)  # inference gets the processing threads
            self._engine = RapidOCR(intra_op_num_threads=PROCESSING_THREADS, inter_op_num_threads=1)
        pix = page.get_pixmap(dpi=OCR_DPI)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
        # Padded white to multiples of 32, so the detector's resize is the identity: OpenCV's KleidiCV resize
        # segfaults on some near-identity sizes (1153x1539 -> 1152x1536).
        height, width = -(-pix.h // 32) * 32, -(-pix.w // 32) * 32
        padded = np.full((height, width, 3), 255, dtype=np.uint8)
        padded[: pix.h, : pix.w] = image if pix.n >= 3 else image[:, :, :1]
        found, _ = self._engine(padded)
        boxes: list[Box] = []
        for box, text, _ in found or []:
            xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
            boxes.append((min(xs), min(ys), max(xs), max(ys), str(text)))
        return boxes


Box = tuple[float, float, float, float, str]


def unread_scans(path: Path, home: Path, numbers: list[int]) -> list[int]:
    """Which of the page ``numbers`` are scans (an image over most of the page) not OCR'd yet."""
    import pymupdf

    kept = PageOcr(path, home).folder
    with pymupdf.open(path) as doc:
        return [
            n
            for n in numbers
            if n <= len(doc) and not (kept / f"{n}.json").exists() and _covered(doc[n - 1]) >= SCAN_COVER
        ]


def _covered(page: Any) -> float:
    area = page.rect.width * page.rect.height or 1.0
    return (
        max(
            (abs(i["bbox"][2] - i["bbox"][0]) * abs(i["bbox"][3] - i["bbox"][1]) for i in page.get_image_info()),
            default=0.0,
        )
        / area
    )


def ocr_pages(path: Path, home: Path, numbers: list[int]) -> dict[int, str]:
    """The OCR of pages ``numbers`` of the PDF at ``path``, done in a worker process, so that a native crash in OCR
    costs this call and not the session; kept in ``home`` like the build's. Empty without the OCR extra."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(_ocr_pages, path, home, numbers).result()


def _ocr_pages(path: Path, home: Path, numbers: list[int]) -> dict[int, str]:
    import pymupdf

    ocr = PageOcr(path, home)
    with pymupdf.open(path) as doc:
        found = {n: ocr(doc[n - 1]) for n in numbers}
    return {n: text for n, text in found.items() if text is not None}


def lines(boxes: list[Box]) -> str:
    """Words (or OCR boxes) as the lines of the page: top to bottom, each left to right, a wide gap as two spaces.
    A scanned table's text layer lists it column by column and OCR finds it cell by cell; here each row is one line
    (`1940  8,736  557  1,478`), which is what a reader and System 1 need of a table. Text set sideways (a landscape
    table on a portrait page) is left in the order it came, which follows its own lines."""
    if not boxes:
        return ""
    long = [b for b in boxes if len(b[4]) >= 3]
    if long and sum(b[3] - b[1] > b[2] - b[0] for b in long) > len(long) / 2:
        return "\n".join(b[4] for b in boxes)
    height = statistics.median(b[3] - b[1] for b in boxes)
    if slope := _tilt(boxes, height):  # a scan set askew: rows read level
        boxes = [(b[0], b[1] - slope * (b[0] + b[2]) / 2, b[2], b[3] - slope * (b[0] + b[2]) / 2, b[4]) for b in boxes]
    rows: list[list[Box]] = []
    top = bottom = 0.0
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        middle = (box[1] + box[3]) / 2
        if rows and top <= middle <= bottom:
            rows[-1].append(box)
        else:
            rows.append([box])
            margin = (box[3] - box[1]) * 0.15  # the band of a row: its first box, a little narrower
            top, bottom = box[1] + margin, box[3] - margin
    out = []
    for row in rows:
        row.sort(key=lambda b: b[0])
        text = row[0][4]
        for before, box in itertools.pairwise(row):
            text += ("  " if box[0] - before[2] > 0.6 * height else " ") + box[4]
        out.append(text)
    return "\n".join(out)


def _tilt(boxes: list[Box], height: float) -> float:
    """How far the page's lines fall per unit across (dy/dx), 0 when level: the slope that stacks the boxes' middles
    into the fewest bands (a projection profile), searched coarse then fine. A scan askew by 1° drops a line by a
    row's height across a page, which turns a wide table's rows into a mix of two."""

    def sharpness(slope: float) -> float:
        bands: dict[int, float] = {}
        for b in boxes:
            band = int(((b[1] + b[3]) / 2 - slope * (b[0] + b[2]) / 2) // (height / 3))
            bands[band] = bands.get(band, 0.0) + b[2] - b[0]
        return sum(w * w for w in bands.values())

    # shortcut: one slope a page; a curved scan splits some prose lines, its tables still
    # read better. Chain each row's boxes left to right if curved pages show up in tables.
    coarse = max((s / 1000 for s in range(-TILT_MAX, TILT_MAX + 1, 4)), key=sharpness)
    best = max((coarse + s / 1000 for s in range(-3, 4)), key=sharpness)
    return best if sharpness(best) > 1.1 * sharpness(0.0) else 0.0


class PdfProcessor:
    """Pages from the text layer (OCR when a page has none), each as its lines (`lines`), sections from the
    document's outline."""

    name: ClassVar[str] = "pdf"
    suffixes: ClassVar[frozenset[str]] = frozenset({".pdf"})
    parallel: ClassVar[bool] = True

    def process(self, path: Path, workspace: Path) -> Derivative:
        import pymupdf

        pages: list[str] = []
        scanned = recognized = 0
        ocr = PageOcr(path, workspace)
        with pymupdf.open(path) as doc:
            for page in doc:
                words = page.get_text("words")
                text = lines([(float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])) for w in words])
                if len(text.strip()) < OCR_MIN_CHARS:
                    scanned += 1
                    if (found := ocr(page)) is not None:
                        text, recognized = found, recognized + 1
                pages.append(text)
            toc = [(int(level), str(title).strip(), int(page)) for level, title, page in doc.get_toc() if page >= 1]
        # shortcut: a long section lists all its pages at one level; window them too when such PDFs show up.
        chunks = _sections(toc, len(pages)) if toc else _page_windows(len(pages))
        for n, text in enumerate(pages, 1):
            parent = _enclosing(chunks, n)
            first = next((line.strip() for line in text.splitlines() if line.strip()), "")
            chunks.append(
                Chunk(key=f"p{n}", kind="page", title=f"page {n}", summary=first[:120], start=n, end=n, parent=parent)
            )
        note = "text layer"
        if scanned:
            note += f"; {scanned} of {len(pages)} pages " + (
                "OCR'd" if recognized == scanned else "have no text (install intern-agent[ocr])"
            )
        summary = next((" ".join(text.split())[:160] for text in pages if text.strip()), "")
        return Derivative(unit="page", units=pages, chunks=chunks, summary=summary, note=note, derived=True)


def _sections(toc: list[tuple[int, str, int]], pages: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    seen: dict[str, int] = {}
    for n, (level, title, page) in enumerate(toc):
        following = next((p for lv, _, p in toc[n + 1 :] if lv <= level), pages + 1)
        while stack and stack[-1][0] >= level:
            stack.pop()
        key = title[:80]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            key = f"{key} ({seen[key]})"
        chunks.append(
            Chunk(
                key=key,
                kind="section",
                title=title,
                start=page,
                end=max(page, min(following - 1, pages)),
                parent=stack[-1][1] if stack else None,
            )
        )
        stack.append((level, key))
    return chunks


def _page_windows(pages: int) -> list[Chunk]:
    if pages <= PAGE_WINDOW:
        return []
    return [
        Chunk(
            key=f"p{s}-{min(s + PAGE_WINDOW - 1, pages)}",
            kind="pages",
            title=f"pages {s}-{min(s + PAGE_WINDOW - 1, pages)}",
            start=s,
            end=min(s + PAGE_WINDOW - 1, pages),
        )
        for s in range(1, pages + 1, PAGE_WINDOW)
    ]


def _enclosing(chunks: list[Chunk], page: int) -> str | None:
    """The key of the innermost section or window holding ``page``."""
    holding = [c for c in chunks if c.kind != "page" and c.start <= page <= c.end]
    return min(holding, key=lambda c: c.end - c.start).key if holding else None


def table_name(path: Path, root: Path, sheet: str = "") -> str:
    """A DuckDB name from the path under ``root`` (and the sheet), unique across the workspace."""
    rel = path.relative_to(root).with_suffix("").as_posix() if path.is_relative_to(root) else path.stem
    name = _UNSAFE.sub("_", rel.replace("/", "__") + (f"__{sheet}" if sheet else "")).strip("_").lower()
    return name if name and not name[0].isdigit() else f"t_{name}"


def literal(path: Path | str) -> str:
    """The path as a SQL string literal (table functions cannot take a prepared parameter)."""
    return "'" + str(path).replace("'", "''") + "'"


def office_processors(root: Path, home: Path) -> list[Processor]:
    """The workspace extension's processors: the text ones, then PDFs, Word documents, workbooks and tables (one
    DuckDB connection for the build, tables named by path under ``root``) and images."""
    tables = Tables(home, root)
    return [
        PythonProcessor(),
        MarkdownProcessor(),
        KeyedProcessor(),
        PdfProcessor(),
        DocxProcessor(),
        SheetProcessor(tables, root),
        TableProcessor(tables, root),
        ImageProcessor(),
        WindowProcessor(),
    ]


def connect(workspace: Path, root: Path | None = None, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """A connection to the workspace's `tables.duckdb`. A CSV or Parquet view names its file relative to the indexed
    ``root`` (so the folder can be moved, copied or cloned with its index), which the connection resolves from."""
    import duckdb

    if not read_only:
        workspace.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(workspace / TABLES_DB), read_only=read_only)
    if root is not None:
        con.execute(f"SET file_search_path = {literal(root)}")
    return con


class Tables:
    """One connection to `tables.duckdb` for a whole build. Opening the file replays its checkpoint, which grows
    with every table loaded, so a connection per spreadsheet made a workspace of hundreds of them quadratic (ten
    minutes and 8 GB for 200 files); shared, it is one load."""

    def __init__(self, workspace: Path, root: Path | None = None) -> None:
        self.workspace, self.root = workspace, root
        self._con: duckdb.DuckDBPyConnection | None = None

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = connect(self.workspace, self.root)
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> Tables:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _session(
    tables: Tables | None, workspace: Path, root: Path | None = None
) -> AbstractContextManager[duckdb.DuckDBPyConnection]:
    """The shared connection when a build gave one, else a connection of its own for this call."""
    return nullcontext(tables.con) if tables is not None else connect(workspace, root)


def _tabs(path: Path) -> list[tuple[str, list[list[Any]]]]:
    """Every tab of a workbook as (title, rows), trimmed; .xlsx/.xlsm through openpyxl, the old .xls through xlrd."""
    if path.suffix.lower() == ".xls":
        import xlrd

        book = xlrd.open_workbook(str(path))
        return [
            (sheet.name, _trimmed([[v if v != "" else None for v in sheet.row_values(r)] for r in range(sheet.nrows)]))
            for sheet in book.sheets()
        ]
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return [
            (sheet.title, _trimmed([list(r) for r in sheet.iter_rows(values_only=True)])) for sheet in book.worksheets
        ]
    finally:
        book.close()


#: Rows a header may be found in, and how full of strings a row must be to be one.
HEADER_ROWS = 5
HEADER_SHARE = 2 / 3


def _header_row(rows: list[list[Any]]) -> int:
    """The index of the header: the first of the top rows that is mostly filled with strings. Office sheets often
    open with a title row (one merged cell) above the real header; without this, columns are col2, col3..."""
    width = max(len(r) for r in rows)
    for i, row in enumerate(rows[:HEADER_ROWS]):
        filled = [v for v in row if v is not None]
        if len(filled) >= HEADER_SHARE * width and sum(isinstance(v, str) for v in filled) >= HEADER_SHARE * len(
            filled
        ):
            return i
    return 0


def _trimmed(rows: list[list[Any]]) -> list[list[Any]]:
    """Without empty rows and the empty columns to the right of the data: a sheet formatted out to column XFD
    comes from openpyxl as 16,384 cells a row."""
    rows = [r for r in rows if any(v is not None for v in r)]
    width = max((i + 1 for r in rows for i, v in enumerate(r) if v is not None), default=0)
    return [r[:width] for r in rows]


def profile(con: duckdb.DuckDBPyConnection, name: str, source: str) -> TableInfo:
    """The table's row count and one line per column: type, and for tables up to PROFILE_COLUMNS wide, the range,
    distinct count and nulls (SUMMARIZE costs a pass per column), and the values of a column with at most
    PROFILE_VALUES of them (one more pass, over those columns only)."""
    rows = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()
    described = con.execute(f'DESCRIBE "{name}"').fetchall()
    columns = [str(row[0]) for row in described]
    if len(columns) <= PROFILE_COLUMNS:
        lines = []
        summary = con.execute(f'SUMMARIZE "{name}"').fetchall()
        values = _few_values(
            con, name, [str(r[0]) for r in summary if r[4] is not None and int(r[4]) <= PROFILE_VALUES]
        )
        for column, kind, low, high, distinct, *_, nulls in summary:
            span = f" {low}..{high}" if low is not None and str(low) != str(high) else ""
            span = span or (f" = {low}" if low is not None else "")
            listed = f": {', '.join(values[column])}"[:300] if column in values else ""
            lines.append(f"{column} ({kind}){span[:80]}, ~{distinct} distinct, {nulls}% null{listed}")
    else:
        lines = [f"{column} ({kind})" for column, kind, *_ in described]
    return TableInfo(
        name=name, source=source, rows=int(rows[0]) if rows else 0, columns=columns, profile="\n".join(lines)
    )


def _few_values(con: duckdb.DuckDBPyConnection, name: str, columns: list[str]) -> dict[str, list[str]]:
    """Each of ``columns``'s distinct non-null values in order, for the columns that have at most PROFILE_VALUES (the
    distinct counts SUMMARIZE gives are estimates)."""
    if not columns:
        return {}
    quoted = ['"' + c.replace('"', '""') + '"' for c in columns]
    lists = ", ".join(f"list(DISTINCT {q} ORDER BY {q}) FILTER (WHERE {q} IS NOT NULL)" for q in quoted)
    found: tuple[list[Any] | None, ...] = con.execute(f'SELECT {lists} FROM "{name}"').fetchone() or ()
    return {
        column: [str(v)[:40] for v in got]
        for column, got in zip(columns, found, strict=False)
        if got is not None and len(got) <= PROFILE_VALUES
    }


def _render(con: duckdb.DuckDBPyConnection, info: TableInfo, title: str) -> str:
    head = con.execute(f'SELECT * FROM "{info.name}" LIMIT {HEAD_ROWS}').fetchall()
    out = [
        f"# {title}",
        f"table {info.name}: {info.rows} rows × {len(info.columns)} columns",
        "",
        "columns:",
        info.profile,
        "",
    ]
    out.append("| " + " | ".join(info.columns) + " |")
    out.append("|" + "---|" * len(info.columns))
    out.extend("| " + " | ".join(str(v)[:40] if v is not None else "" for v in row) + " |" for row in head)
    if info.rows > len(head):
        out.append(f"({info.rows - len(head)} more rows; query table {info.name} with sql)")
    return "\n".join(out)


def _shape(info: TableInfo) -> str:
    columns = ", ".join(info.columns[:12]) + (f", +{len(info.columns) - 12} more" if len(info.columns) > 12 else "")
    return f"{info.rows} rows × {len(info.columns)} columns: {columns}"


def _head(con: duckdb.DuckDBPyConnection, info: TableInfo) -> str:
    sample = con.execute(f'SELECT * FROM "{info.name}" LIMIT {SAMPLE_ROWS}').fetchall()
    return " ".join([*info.columns, *(str(v) for row in sample for v in row if v is not None)])


def _table_chunk(info: TableInfo, title: str, position: int, caption: str = "") -> Chunk:
    columns = ", ".join(info.columns[:12]) + (f", +{len(info.columns) - 12} more" if len(info.columns) > 12 else "")
    return Chunk(
        key=info.name,
        kind="table",
        title=f"{title} → table {info.name}",
        summary=(f"{caption}; " if caption else "") + f"{info.rows} rows: {columns}",
        start=position,
        end=position,
    )


class _UsesTables:
    """A processor that loads into `tables.duckdb`: it shares a build's connection, drops a removed file's tables
    and closes the connection when the build ends."""

    def __init__(self, tables: Tables | None = None, root: Path | None = None) -> None:
        self.tables = tables
        self.root = root

    def remove(self, entry: FileEntry) -> None:
        if self.tables is None:
            return
        for info in entry.tables:
            _drop(self.tables.con, info.name)

    def close(self) -> None:
        if self.tables is not None:
            self.tables.close()


class SheetProcessor(_UsesTables):
    """Every tab of a workbook as a DuckDB table (the first mostly-filled row of strings is its header) and a
    rendered page."""

    name: ClassVar[str] = "sheet"
    suffixes: ClassVar[frozenset[str]] = frozenset({".xlsx", ".xlsm", ".xls"})

    def process(self, path: Path, workspace: Path) -> Derivative:
        root = self.root or workspace.parent
        units: list[str] = []
        chunks: list[Chunk] = []
        tables: list[TableInfo] = []
        heads: list[str] = []
        with _session(self.tables, workspace) as con:
            for n, (title, rows) in enumerate(_tabs(path), 1):
                if not rows:
                    units.append(f"# {title}\n(empty tab)")
                    chunks.append(Chunk(key=title, kind="table", title=f"tab {title}", summary="empty", start=n, end=n))
                    continue
                name = table_name(path, root, title)
                staging = workspace / "derived" / path.relative_to(root).parent / f"{path.name}.{n}.csv"
                caption = _load_rows(con, name, rows, staging)
                info = profile(con, name, f"{path.relative_to(root).as_posix()}#{title}")
                tables.append(info)
                units.append(_render(con, info, f"tab {title}" + (f": {caption}" if caption else "")))
                chunks.append(_table_chunk(info, f"tab {title}", n, caption))
                heads.append(f"{title} {caption} {_head(con, info)}")
        summary = "; ".join(f"{t.source.rsplit('#', 1)[-1]}: {_shape(t)}" for t in tables)[:300]
        return Derivative(
            unit="page",
            units=units,
            chunks=chunks,
            tables=tables,
            summary=summary,
            note=f"{len(units)} tabs",
            head=" ".join(heads),
            derived=True,
        )


class TableProcessor(_UsesTables):
    """A CSV, TSV or Parquet file as a DuckDB view on the source and a rendered page. A view reads the file in place,
    so a table of any size is indexed (a year of journal lines is gigabytes)."""

    name: ClassVar[str] = "table"
    suffixes: ClassVar[frozenset[str]] = frozenset({".csv", ".tsv", ".parquet"})
    max_bytes: ClassVar[int | None] = None

    def process(self, path: Path, workspace: Path) -> Derivative:
        root = self.root or workspace.parent
        name = table_name(path, root)
        if any(path.with_suffix(s).exists() for s in self.suffixes - {path.suffix.lower()}):
            name = f"{name}_{path.suffix[1:].lower()}"  # a.csv beside a.parquet: each its own table
        source = path.relative_to(root) if path.is_relative_to(root) else path
        rel = path.relative_to(root).as_posix()
        caption = options = ""
        with _session(self.tables, workspace, root) as con:
            _drop(con, name)  # it may have been loaded as a table before the file changed
            if path.suffix == ".parquet":
                con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet({literal(source)})')
            else:
                options = _csv_view(con, name, source)
            info = profile(con, name, rel)
            if (
                sum(bool(_UNNAMED.fullmatch(c)) for c in info.columns) * 2 > len(info.columns)
                and info.rows <= RESHAPE_ROWS
            ):
                if path.suffix == ".parquet":  # the title is in the column names, the rest in the rows
                    named = [None if _UNNAMED.fullmatch(c) else c for c in info.columns]
                    raw = [named, *con.execute(f'SELECT * FROM "{name}"').fetchall()]
                else:  # every line as text: the sniffer may have taken the header line for a comment
                    raw = con.execute(
                        f"SELECT * FROM read_csv({literal(source)}, header=false, all_varchar=true{options})"
                    ).fetchall()
                rows = _trimmed([[None if _blank(v) else v for v in r] for r in raw])
                _drop(con, name)
                caption = _load_rows(con, name, rows, workspace / "derived" / Path(rel).parent / f"{path.name}.csv")
                info = profile(con, name, rel)
            page = _render(con, info, path.name + (f": {caption}" if caption else ""))
            head = f"{caption} {_head(con, info)}" if caption else _head(con, info)
        chunk = _table_chunk(info, path.name, 1, caption)
        return Derivative(
            unit="page", units=[page], chunks=[chunk], tables=[info], summary=_shape(info), head=head, derived=True
        )


def _csv_view(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> str:
    """A view on the CSV that reads it: UTF-8, else Latin-1 (an export from Excel on Windows or a Mac). Returns the
    read_csv options it needed."""
    import duckdb

    for options in ("", ", encoding='latin-1'"):
        try:
            con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_csv({literal(path)}{options})')
            con.execute(f'DESCRIBE "{name}"')
            return options
        except duckdb.InvalidInputException:
            if options:
                raise
    return ""


def query_memory() -> int | None:
    """QUERY_MEMORY_SHARE of the machine's physical memory in bytes; None where the platform does not say."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * QUERY_MEMORY_SHARE)
    except (AttributeError, ValueError, OSError):
        return None


def _drop(con: duckdb.DuckDBPyConnection, name: str) -> None:
    """Drop the table or view ``name``, whichever it is: DuckDB's DROP of one kind fails on the other."""
    kinds = con.execute(
        "SELECT 'VIEW' FROM duckdb_views() WHERE view_name = ? AND NOT internal "
        "UNION ALL SELECT 'TABLE' FROM duckdb_tables() WHERE table_name = ?",
        [name, name],
    ).fetchall()
    for (kind,) in kinds:
        con.execute(f'DROP {kind} "{name}"')


def _blank(value: object) -> bool:
    """An empty cell, as a sheet saved by pandas may write it: None, NaN, "" or the string "nan"."""
    if isinstance(value, float):
        return value != value
    return value is None or (isinstance(value, str) and value.strip() in ("", "nan"))


def _load_rows(con: duckdb.DuckDBPyConnection, name: str, rows: list[list[Any]], staging: Path) -> str:
    """The rows of a sheet as the table ``name``, its header found among the top rows (see `_header_row`); returns
    the rows above the header as a caption."""
    at = _header_row(rows)
    caption = " ".join(str(v).strip() for r in rows[:at] for v in r if v is not None)[:200]
    header = [" ".join(str(v).split()) if v is not None else f"col{i}" for i, v in enumerate(rows[at], 1)]
    staging.parent.mkdir(parents=True, exist_ok=True)
    with staging.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows[at + 1 :])
    # The dialect is given and strict mode off: the sniffer fails on a tiny tab with a newline in a header cell, and
    # office sheets have ragged rows.
    con.execute(
        f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM read_csv({literal(staging)}, header=true, '
        "delim=',', quote='\"', escape='\"', strict_mode=false)"
    )
    return caption


class DocxProcessor:
    """A Word document as Markdown lines (headings by style, tables as rows), sectioned like a Markdown file."""

    name: ClassVar[str] = "docx"
    suffixes: ClassVar[frozenset[str]] = frozenset({".docx"})

    def process(self, path: Path, workspace: Path) -> Derivative:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = docx.Document(str(path))
        lines: list[str] = []
        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                paragraph = Paragraph(child, document)
                text = paragraph.text.strip()
                if not text:
                    continue
                style = (paragraph.style.name if paragraph.style is not None else "") or ""
                if style == "Title":
                    text = f"# {text}"
                elif style.startswith("Heading") and (level := style.removeprefix("Heading").strip()).isdigit():
                    text = f"{'#' * min(int(level) + 1, 6)} {text}"
                lines.append(text)
            elif child.tag.endswith("}tbl"):
                for row in Table(child, document).rows:
                    lines.append("| " + " | ".join(" ".join(cell.text.split()) for cell in row.cells) + " |")
        derivative = sections(lines) or windows(lines)
        derivative.derived = True
        return derivative


class ImageProcessor:
    """An image is one unit naming its size; `read` attaches the pixels for a model that can see."""

    name: ClassVar[str] = "image"
    suffixes: ClassVar[frozenset[str]] = IMAGE_SUFFIXES

    def process(self, path: Path, workspace: Path) -> Derivative:
        from PIL import Image

        with Image.open(path) as image:
            line = f"{path.name}: {image.width}×{image.height} {image.format or path.suffix[1:]} image"
        return Derivative(unit="page", units=[line], summary=line)
