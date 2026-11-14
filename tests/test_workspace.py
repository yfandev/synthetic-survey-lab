"""The workspace extension: office documents in the index (PDF pages and sections, workbook tabs and tables, Word
documents), `sql`, the document-aware `read`, and `decision_files` over them once the agent has called `index_files`."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

pytest.importorskip("duckdb", reason="needs the workspace extra: uv sync --all-extras")

from intern.decisions import Answers, ChoiceAnswer, DecisionClient
from intern.executor import LocalExecutor
from intern.extensions.workspace import documents as docs
from intern.extensions.workspace.documents import lines, office_processors
from intern.extensions.workspace.extension import Workspace
from intern.locate import index as wi
from intern.tools.base import ToolContext

from .support import Needle


def build(root: Path) -> wi.Index:
    return wi.build(root, root / Workspace.folder, processors=office_processors, report=lambda line: None)


def test_the_index_lives_in_its_own_folder_and_never_indexes_itself(tmp_path: Path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Total due 1,250 EUR.")
    doc.save(tmp_path / "invoice.pdf")
    build(tmp_path)
    assert (tmp_path / ".decision_workspace" / "derived" / "invoice.pdf.md").exists()
    assert not (tmp_path / ".intern").exists()  # NDI keeps its own workspace state under .intern/
    again = build(tmp_path)  # not a git work tree: the walk sees the folder, and skips it as a cache
    assert set(again.files) == {"invoice.pdf"}, set(again.files)


def test_a_title_row_above_the_header_becomes_the_tab_caption(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "OKR"
    sheet.append(["Product team OKR tracking (2024)"])
    sheet.append(["quarter", "objective", "score"])
    sheet.append(["Q1", "core experience", 1.0])
    sheet.append(["Q2", "growth", 0.8])
    book.save(tmp_path / "okr.xlsx")
    entry = build(tmp_path).files["okr.xlsx"]
    assert entry.tables[0].columns == ["quarter", "objective", "score"] and entry.tables[0].rows == 2
    assert entry.chunks[0].summary.startswith("Product team OKR tracking (2024); 2 rows: quarter, objective, score")
    assert "OKR tracking" in entry.head


def test_documents_become_pages_sections_and_tables(tmp_path: Path):
    pymupdf = pytest.importorskip("pymupdf")
    openpyxl = pytest.importorskip("openpyxl")
    doc = pymupdf.open()
    for n in range(1, 4):
        doc.new_page().insert_text((72, 72), f"Page {n} of the 2023 report.", fontsize=11)
    doc.set_toc([[1, "Intro", 1], [1, "Body", 2], [2, "Detail", 3]])
    doc.save(tmp_path / "r.pdf")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Sales"
    for row in [["region", "amount"], ["EU", 10], ["US", 20]]:
        sheet.append(row)
    book.save(tmp_path / "s.xlsx")
    (tmp_path / "p.csv").write_text("name,age\nAnn,34\nBo,28\n")

    index = build(tmp_path)
    pdf = index.files["r.pdf"]
    assert pdf.unit == "page" and pdf.units == 3 and pdf.note == "text layer"
    assert [(c.key, c.kind, c.parent) for c in pdf.chunks] == [
        ("Intro", "section", None),
        ("Body", "section", None),
        ("Detail", "section", "Body"),
        ("p1", "page", "Intro"),
        ("p2", "page", "Body"),
        ("p3", "page", "Detail"),
    ]
    derived = (tmp_path / ".decision_workspace" / "derived" / "r.pdf.md").read_text()
    assert derived.startswith("--- Page 1 ---\nPage 1 of the 2023 report.") and "--- Page 3 ---" in derived
    assert "years: 2023" in pdf.card

    sheet_entry, csv_entry = index.files["s.xlsx"], index.files["p.csv"]
    assert [t.name for t in sheet_entry.tables] == ["s__sales"] and [t.name for t in csv_entry.tables] == ["p"]
    assert sheet_entry.summary == "Sales: 2 rows × 2 columns: region, amount"
    assert "amount (BIGINT) 10..20" in sheet_entry.chunks[0].card
    with docs.connect(tmp_path / ".decision_workspace", tmp_path, read_only=True) as con:
        assert con.execute("select sum(amount) from s__sales").fetchone() == (30,)
        assert con.execute("select count(*) from p where age > 30").fetchone() == (1,)

    (tmp_path / "p.csv").unlink()
    build(tmp_path)
    with docs.connect(tmp_path / ".decision_workspace", tmp_path, read_only=True) as con:
        assert con.execute("select count(*) from duckdb_views() where view_name = 'p'").fetchone() == (0,)
    assert not (tmp_path / ".decision_workspace" / "derived" / "p.csv.md").exists()


def test_a_table_of_any_size_and_a_latin1_export_with_a_title_row_are_indexed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(wi, "MAX_FILE_BYTES", 100)  # the journal stands in for a gigabyte one
    (tmp_path / "journal.csv").write_text("je,amount\n" + "".join(f"J{n},{n}\n" for n in range(200)))
    (tmp_path / "notes.txt").write_text("x" * 200)
    (tmp_path / "exceptions.csv").write_bytes("Overall\xa0;;\n#;Entry;Preparer\n1;JE01;lbauer\n".encode("latin-1"))
    lines_seen: list[str] = []
    index = wi.build(tmp_path, tmp_path / ".decision_workspace", processors=office_processors, report=lines_seen.append)
    assert set(index.files) == {"journal.csv", "exceptions.csv"}
    assert any(line.startswith("skip notes.txt: ") and "limit" in line for line in lines_seen)
    assert index.files["exceptions.csv"].tables[0].columns == ["#", "Entry", "Preparer"]
    assert not wi.stale(tmp_path, index, office_processors)
    with docs.connect(tmp_path / ".decision_workspace", tmp_path, read_only=True) as con:
        assert con.execute("select sum(amount) from journal").fetchone() == (19900,)
        assert con.execute("select preparer from exceptions").fetchone() == ("lbauer",)


def test_a_table_saved_from_a_sheet_with_a_title_above_its_header_gets_the_real_header(tmp_path: Path):
    duckdb = pytest.importorskip("duckdb")
    # As pandas writes a statement sheet: the title in the first column's name, "nan" for empty cells.
    rows = [
        ("Consolidated Balance Sheet ($ millions)", "nan", "nan"),
        ("nan", "nan", "nan"),
        ("Line item", "Dec 31, 2025", "Dec 31, 2024"),
        ("Cash and cash equivalents", "6296.5", "6781.0"),
        ("Inventories", "1200", "1100"),
    ]
    values = ", ".join(str(r) for r in rows)
    target = tmp_path / "balance_sheet.parquet"
    with duckdb.connect() as con:
        con.execute(
            f'COPY (SELECT col0 AS "Acme Corp", col1 AS "Unnamed: 1", col2 AS "Unnamed: 2" FROM (VALUES {values}) '
            f"t(col0, col1, col2)) TO '{target}' (FORMAT parquet)"
        )
    entry = build(tmp_path).files["balance_sheet.parquet"]
    assert entry.tables[0].columns == ["Line item", "Dec 31, 2025", "Dec 31, 2024"] and entry.tables[0].rows == 2
    assert entry.chunks[0].summary.startswith("Acme Corp Consolidated Balance Sheet ($ millions); 2 rows")
    with docs.connect(tmp_path / Workspace.folder, tmp_path, read_only=True) as con:
        assert con.execute('select sum("Dec 31, 2025") from balance_sheet').fetchone() == (7496.5,)


def test_a_column_with_few_values_lists_them_in_the_table_card(tmp_path: Path):
    rows = "".join(f"A{n},{'blocked' if n % 7 == 0 else 'active'},{n}\n" for n in range(200))
    (tmp_path / "accounts.csv").write_text("account,status,balance\n" + rows)
    card = build(tmp_path).files["accounts.csv"].chunks[0].card
    assert "status (VARCHAR) active..blocked, ~2 distinct, 0.00% null: active, blocked" in card
    assert "balance (BIGINT) 0..199" in card and "0, 1, 2" not in card  # 200 values are not listed


def test_files_that_differ_by_suffix_are_two_tables_and_a_removed_workbook_drops_its_tabs(tmp_path: Path):
    duckdb = pytest.importorskip("duckdb")
    openpyxl = pytest.importorskip("openpyxl")
    (tmp_path / "rollforward.csv").write_text("asset,cost\nA,1\n")
    with duckdb.connect() as con:
        con.execute(f"COPY (SELECT 'B' AS asset, 2 AS cost) TO '{tmp_path / 'rollforward.parquet'}' (FORMAT parquet)")
    book = openpyxl.Workbook()
    book.active.append(["k", "v"])
    book.active.append(["a", 1])
    book.save(tmp_path / "book.xlsx")
    index = build(tmp_path)
    names = {t.name for e in index.files.values() for t in e.tables}
    assert names == {"rollforward_csv", "rollforward_parquet", "book__sheet"}
    (tmp_path / "book.xlsx").unlink()
    build(tmp_path)
    with docs.connect(tmp_path / Workspace.folder, tmp_path, read_only=True) as con:
        assert con.execute(
            "select asset from rollforward_csv union all select asset from rollforward_parquet"
        ).fetchall() == [("A",), ("B",)]
        assert con.execute("select count(*) from duckdb_tables() where table_name = 'book__sheet'").fetchone() == (0,)


async def test_a_folder_copied_with_its_index_queries_its_own_csv_and_parquet(tmp_path: Path):
    duckdb = pytest.importorskip("duckdb")
    original = tmp_path / "original"
    (original / "ledger").mkdir(parents=True)
    (original / "ledger" / "je.csv").write_text("je,amount\nJ1,10\nJ2,32\n")
    with duckdb.connect() as con:
        con.execute(f"COPY (SELECT 'ACS' AS entity, 5 AS n) TO '{original / 'ledger' / 'tb.parquet'}' (FORMAT parquet)")
    build(original)
    copy = tmp_path / "copy"  # a clone, a move, a checkout elsewhere: the index travels in .intern/
    shutil.copytree(original, copy)
    shutil.rmtree(original)
    sql = Workspace().tools()[2]
    rows = await sql.run(
        sql.Args(query="select (select sum(amount) from ledger__je) + (select n from ledger__tb)"), ctx(copy)
    )
    assert rows.status == "ok" and rows.content.endswith("| 47 |\n[kept as res_1: 1 rows]"), rows.content


async def test_sql_reads_the_workspace_tables_and_nothing_else(tmp_path: Path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "je.csv").write_text("je,amount\nJ1,10\n")
    (outside / "secret.csv").write_text("key\nhunter2\n")
    build(root)
    sql = Workspace().tools()[2]
    assert (await sql.run(sql.Args(query="select sum(amount) from je"), ctx(root))).content.endswith(
        "| 10 |\n[kept as res_1: 1 rows]"
    )
    escapes = [
        f"COPY (select 1) TO '{outside / 'out.csv'}'",  # a write, from a tool declared read-only
        f"select * from read_csv('{outside / 'secret.csv'}')",  # a file the read tool may refuse
        "select * from read_csv('../outside/secret.csv')",
        "INSTALL httpfs",
        "SET enable_external_access = true",
        "select 1; COPY (select 1) TO 'out.csv'",
    ]
    for query in escapes:
        refused = await sql.run(sql.Args(query=query), ctx(root))
        assert refused.status == "error" and "hunter2" not in refused.content, query
    assert not (outside / "out.csv").exists() and not (root / "out.csv").exists()
    # A query holds a share of the machine's memory and spills past it, not DuckDB's default 80%.
    limit = (await sql.run(sql.Args(query="select current_setting('memory_limit')"), ctx(root))).content
    amount, unit = limit.splitlines()[2].strip("| ").split()
    share = docs.query_memory()
    assert share is not None and float(amount) * {"MiB": 2**20, "GiB": 2**30}[unit] <= share


async def test_a_query_result_is_kept_for_later_queries_and_leaves_nothing_behind(tmp_path: Path, monkeypatch):
    from intern.extensions.workspace import tools as wt

    (tmp_path / "je.csv").write_text("je,entity,amount\n" + "".join(f"J{n},E{n % 3},{n}\n" for n in range(30)))
    build(tmp_path)
    sql = Workspace().tools()[2]
    totals = await sql.run(
        sql.Args(query="select entity, sum(amount) s from je group by 1 order by 2 desc"), ctx(tmp_path)
    )
    assert totals.content.endswith("[kept as res_1: 3 rows]")
    reused = await sql.run(sql.Args(query="select entity from res_1 order by s desc limit 1"), ctx(tmp_path))
    assert "| E2 |" in reused.content and reused.content.endswith("[kept as res_2: 1 rows]")
    monkeypatch.setattr(wt, "KEEP_ROWS", 10)
    big = await sql.run(sql.Args(query="select * from je", limit=5), ctx(tmp_path))  # never kept cut short
    assert big.content.endswith("[5 rows shown of more than 10, too many to keep: aggregate or filter in the query]")
    assert (await sql.run(sql.Args(query="select count(*) from res_3"), ctx(tmp_path))).status == "error"
    # The index can be rebuilt between queries (the tables are attached only while one runs), and the results stay.
    (tmp_path / "more.csv").write_text("k\n1\n")
    build(tmp_path)
    assert "| 3 |" in (await sql.run(sql.Args(query="select count(*) from res_1"), ctx(tmp_path))).content
    # Its spill folder goes when the session closes; a killed session's is swept by the next.
    spills = tmp_path / Workspace.folder / wt.SPILL_DIR
    (spills / "999999999").mkdir()
    assert isinstance(sql, wt.Sql)
    session = sql._session
    assert session is not None and session.spill.is_dir()
    session.close()
    assert not session.spill.exists()
    wt.SqlSession(tmp_path / Workspace.folder, tmp_path).close()
    assert list(spills.iterdir()) == []


async def test_a_query_over_its_time_is_stopped_and_the_session_goes_on(tmp_path: Path):
    import time

    (tmp_path / "je.csv").write_text("je,amount\nJ1,10\n")
    build(tmp_path)
    sql = Workspace().tools()[2]
    endless = "select sum(a.range * b.range) from range(1000000000) a, range(1000) b"
    started = time.monotonic()
    stopped = await sql.run(sql.Args(query=endless, timeout_s=1), ctx(tmp_path))
    assert stopped.status == "error" and "ran over 1s and was stopped" in stopped.content
    assert time.monotonic() - started < 10
    assert (await sql.run(sql.Args(query="select sum(amount) from je"), ctx(tmp_path))).content.splitlines()[
        2
    ] == "| 10 |"


async def test_a_failed_query_shows_duckdbs_whole_error_and_lists_tables_only_when_it_cannot_name_one(tmp_path: Path):
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "accounts.csv").write_text("account_number,account_type\n101000,Asset\n")
    build(tmp_path)
    sql = Workspace().tools()[2]
    column = await sql.run(sql.Args(query="select acct from ledger__accounts"), ctx(tmp_path))
    assert 'Candidate bindings: "account_type", "account_number"' in column.content and "LINE 1:" in column.content
    assert "Tables:" not in column.content  # the columns are the fix, not the list of tables
    near = await sql.run(sql.Args(query="select * from ledger__account"), ctx(tmp_path))
    assert 'Did you mean "ledger__accounts"' in near.content and "Tables:" not in near.content
    far = await sql.run(sql.Args(query="select * from nope"), ctx(tmp_path))
    assert "Tables: ledger__accounts." in far.content


async def test_scratch_tables_and_views_live_in_the_session_and_the_workspace_stays_read_only(tmp_path: Path):
    (tmp_path / "je.csv").write_text("je,entity,amount\nJ1,E1,10\nJ2,E2,32\nJ3,E1,5\n")
    build(tmp_path)
    sql = Workspace().tools()[2]

    async def run(query: str) -> str:
        return (await sql.run(sql.Args(query=query), ctx(tmp_path))).content

    await run("create temp table by_entity as select entity, sum(amount) total from je group by 1")
    await run("create view big as select * from by_entity where total > 12")
    await run("insert into by_entity values ('E9', 1)")
    assert "| E2 | 32 |" in await run("select * from big order by entity")
    assert "| 5 |" in await run("select count(*) from (select * from by_entity union all select entity, 0 from big)")
    assert "read-only" in await run("drop table je") and (await run("insert into je values (1)")).startswith("Error")
    assert "not CREATE SECRET" in await run("create persistent secret s (type s3, key_id 'k', secret 'x')")
    assert "not COPY" in await run(f"COPY by_entity TO '{tmp_path / 'out.csv'}'")
    assert "| 3 |" in await run("select count(*) from je")  # the workspace's table is as it was
    await run("drop view big")
    assert "does not exist" in await run("select * from big")


async def test_each_statement_of_a_batch_shows_its_result_and_a_failed_one_keeps_those_before(tmp_path: Path):
    (tmp_path / "je.csv").write_text("je,entity,amount\nJ1,E1,10\nJ2,E2,32\nJ3,E1,5\n")
    build(tmp_path)
    sql = Workspace().tools()[2]
    batch = await sql.run(
        sql.Args(
            query="select count(*) n from je;\n-- then the large ones; on their own\n"
            "create table t as select * from je where amount > 6;\n"
            "select sum(amount) s from t -- the large ones"
        ),
        ctx(tmp_path),
    )
    assert batch.status == "ok", batch.content
    assert "| 3 |" in batch.content and "| 42 |" in batch.content  # every SELECT's rows, not only the last's
    assert "[kept as res_1: 1 rows]" in batch.content and "[kept as res_2: 1 rows]" in batch.content
    failed = await sql.run(sql.Args(query="select max(amount) from je; select nope from je"), ctx(tmp_path))
    assert failed.status == "error" and "| 32 |" in failed.content and '"nope" not found' in failed.content
    commented = await sql.run(sql.Args(query="select 7 as x -- a note after the query"), ctx(tmp_path))
    assert commented.status == "ok" and "| 7 |" in commented.content, commented.content
    wide = await sql.run(
        sql.Args(query="select 'Q4 — FY2025' as period; -- non-ASCII: é\nselect 8 as y"), ctx(tmp_path)
    )
    assert wide.status == "ok" and "| Q4 — FY2025 |" in wide.content and "| 8 |" in wide.content, wide.content


async def test_describe_of_a_workspace_table_shows_its_profile_and_first_rows(tmp_path: Path):
    rows = "".join(f"J{n},{'posted' if n % 2 else 'draft'},{n}\n" for n in range(30))
    (tmp_path / "je.csv").write_text("je,status,amount\n" + rows)
    build(tmp_path)
    sql = Workspace().tools()[2]

    async def run(query: str) -> str:
        return (await sql.run(sql.Args(query=query), ctx(tmp_path))).content

    shown = await run("DESCRIBE je")
    assert "30 rows × 3 columns" in shown and "draft, posted" in shown, shown  # the values, from the index
    assert "| J0 | draft | 0 |" in shown and "| J9 |" not in shown  # and its first rows
    assert "draft, posted" in await run('describe "je";')
    await run("create table mine as select 1 as x")
    assert "column_name" in await run("describe mine")  # a table of the session's own: DuckDB's DESCRIBE


async def test_the_memory_drive_guard_keeps_sql_export_out_of_the_drive(tmp_path: Path):
    from intern import hooks as h
    from intern.extensions.memory import Memory
    from intern.journal import Journal
    from intern.memory.drive import Drive
    from intern.messages import ToolCall
    from intern.models.profile import Profile

    (tmp_path / "drive").mkdir()
    memory = Memory(Drive(tmp_path / "drive", "c0", {}))
    run = h.RunContext(
        run_id="r", turn=1, cwd=str(tmp_path), profile=Profile(), working_window=32_768, chars_per_token=4.0,
        journal=Journal(str(tmp_path)), decisions=DecisionClient(LoadJudge(), lambda e: None), emit=lambda e: None,
        state=lambda: "task",
    )  # fmt: skip

    def export(path: str) -> ToolCall:
        return ToolCall(id="c", name="sql_export", arguments={"source": "res_1", "path": path})

    refused = await memory.tool_call(export("drive/totals.csv"), run)
    assert isinstance(refused, h.Refuse), refused
    assert "memory drive" in refused.reason
    assert await memory.tool_call(export("out/totals.csv"), run) is None


async def test_sql_export_writes_every_row_of_a_result_to_a_file_and_never_over_a_document(tmp_path: Path):
    duckdb = pytest.importorskip("duckdb")
    (tmp_path / "je.csv").write_text("je,amount\n" + "".join(f"J{n},{n}\n" for n in range(500)))
    build(tmp_path)
    ws = Workspace()
    tools = ws.tools()
    sql, export = tools[2], tools[4]
    assert export.name == "sql_export" and not export.read_only  # a write: the guard judges it
    await sql.run(sql.Args(query="select je, amount * 2 as twice from je where amount >= 100"), ctx(tmp_path))
    done = await export.run(export.Args(source="res_1", path="out/big.csv"), ctx(tmp_path))
    assert done.status == "ok" and "400 rows" in done.content, done.content
    lines = (tmp_path / "out" / "big.csv").read_text().splitlines()
    assert lines[0] == "je,twice" and len(lines) == 401  # every row, not the 50 a query shows
    query = await export.run(export.Args(source="select * from je where amount < 3", path="few.parquet"), ctx(tmp_path))
    assert query.status == "ok"
    with duckdb.connect() as con:
        assert con.execute(f"select count(*) from '{tmp_path / 'few.parquet'}'").fetchone() == (3,)
    before = (tmp_path / "je.csv").read_text()
    for source, path in [
        ("res_1", "je.csv"),  # an indexed document
        ("res_1", str(tmp_path.parent / "outside.csv")),
        ("res_1", "notes.txt"),
        ("res_1", ".decision_workspace/x.csv"),  # the index
        ("res_1", ".intern/x.csv"),
        ("select 1; drop table res_1", "x.csv"),
    ]:
        refused = await export.run(export.Args(source=source, path=path), ctx(tmp_path))
        assert refused.status != "ok", (source, path)
    assert (tmp_path / "je.csv").read_text() == before and not (tmp_path.parent / "outside.csv").exists()
    assert "| 400 |" in (await sql.run(sql.Args(query="select count(*) from res_1"), ctx(tmp_path))).content
    # A later change (a bash call) refreshes the index, which then holds the exports: the session's own may be
    # written again, the documents still not.
    ws.store.mark_dirty()
    again = await export.run(export.Args(source="select * from je where amount < 5", path="few.parquet"), ctx(tmp_path))
    assert again.status == "ok" and "5 rows" in again.content, again.content
    assert (await export.run(export.Args(source="res_1", path="je.csv"), ctx(tmp_path))).status != "ok"
    assert (tmp_path / "je.csv").read_text() == before


class LoadJudge:
    """System 1 for the large-table check: a command whose text reads a CSV with pandas loads it."""

    name = "load-judge"

    def __init__(self) -> None:
        self.states: list[str] = []

    async def decide(self, state: str, questions: dict) -> Answers:
        self.states.append(state)
        loads = "read_csv" in state
        return Answers(answers={"loads": ChoiceAnswer(choice="loads" if loads else "other")})


async def test_a_command_that_loads_a_large_table_in_python_is_refused_with_its_sql_table(tmp_path: Path, monkeypatch):
    from intern import hooks as h
    from intern.extensions.workspace import extension as we
    from intern.journal import Journal
    from intern.messages import ToolCall
    from intern.models.profile import Profile

    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "je.csv").write_text("je,amount\n" + "".join(f"J{n},{n}\n" for n in range(500)))
    (tmp_path / "ledger" / "rates.csv").write_text("ccy,rate\nEUR,1.1\n")
    (tmp_path / "load.py").write_text(
        "import pandas as pd\ndf = pd.read_csv('ledger/je.csv')\nprint(df.amount.sum())\n"
    )
    ws = Workspace()
    judge = LoadJudge()
    ws.store.decisions = DecisionClient(judge, lambda e: None)
    await ws.tools()[0].run(ws.tools()[0].Args(path="."), ctx(tmp_path))
    run = h.RunContext(
        run_id="r", turn=3, cwd=str(tmp_path), profile=Profile(), working_window=32_768, chars_per_token=4.0,
        journal=Journal(str(tmp_path)),
        decisions=ws.store.decisions, emit=lambda e: None, state=lambda: "task",
    )  # fmt: skip

    async def check(command: str) -> h.ToolCallDecision | None:
        return await ws.tool_call(ToolCall(id="c", name="bash", arguments={"command": command}), run)

    # At this machine's size neither file is large: nothing is asked, nothing refused.
    assert await check("python3 load.py") is None and judge.states == []
    # A file whose data is over the share is large (here every file with data in it).
    monkeypatch.setattr(we, "LARGE_TABLE_SHARE", 1e-12)
    refused = await check("python3 -c \"import pandas as pd; print(pd.read_csv('ledger/je.csv').amount.sum())\"")
    assert isinstance(refused, h.Refuse) and "ledger/je.csv" in refused.reason and "500 rows" in refused.reason
    assert "ledger/je.csv as ledger__je" in refused.next_action
    script = await check("cd . && python3 load.py")  # the script it runs is read too
    assert isinstance(script, h.Refuse) and "--- load.py ---" in judge.states[-1]
    asked = len(judge.states)
    assert await check("wc -l ledger/je.csv") is None and len(judge.states) == asked + 1  # judged: does not load
    assert await check("ls -la && git status") is None and len(judge.states) == asked + 1  # names no table: not asked
    assert await ws.tool_call(ToolCall(id="c", name="read", arguments={"path": "ledger/je.csv"}), run) is None


def ctx(root: Path) -> ToolContext:
    return ToolContext(cwd=root, executor=LocalExecutor(), abort=asyncio.Event(), max_output_chars=4000)


def documents(root: Path) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    openpyxl = pytest.importorskip("openpyxl")
    (root / "reports").mkdir()
    doc = pymupdf.open()
    for n in range(1, 5):
        doc.new_page().insert_text((72, 72), f"Page {n}. " + ("Total due 1,250 EUR." if n == 4 else "Revenue grew."))
    doc.set_toc([[1, "Intro", 1], [1, "Financials", 2], [1, "Appendix", 4]])
    doc.save(root / "reports" / "annual.pdf")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Sales"
    for row in [["region", "amount"], ["EU", 100.5], ["US", 200]]:
        sheet.append(row)
    targets = book.create_sheet("Targets")
    targets.append(["region", "target"])
    targets.append(["EU", 300])
    book.save(root / "reports" / "sales.xlsx")
    (root / "notes.md").write_text("# Notes\n\nAbout the report.\n")
    docx = pytest.importorskip("docx")
    resume = docx.Document()
    resume.add_heading("Li Yuchen", 0)
    resume.add_paragraph("Candidate for the logistics role.")
    resume.add_heading("Education", 1)
    resume.add_paragraph("Beijing University, 2020.")
    resume.add_heading("Skills", 1)
    table = resume.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "skill", "level"
    table.cell(1, 0).text, table.cell(1, 1).text = "forklift", "expert"
    resume.save(root / "resumes" / "li_yuchen.docx") if (root / "resumes").mkdir() is None else None


async def test_the_tools_open_documents_and_tables(tmp_path: Path):
    documents(tmp_path)
    ws = Workspace()
    ws.store.decisions = DecisionClient(
        Needle("Total due", "amount", "zebra", ("Beijing", "Education")), lambda e: None
    )
    index_files, find, sql, read, _ = ws.tools()
    assert [t.name for t in ws.tools()] == ["index_files", "decision_files", "sql", "read", "sql_export"]

    early = await find.run(find.Args(question="Where is the Total due on the invoice?"), ctx(tmp_path))
    assert early.status == "error" and "no index under ." in early.content and "index tool" in early.content
    assert (await sql.run(sql.Args(query="select 1"), ctx(tmp_path))).status == "error"  # nothing indexed yet
    built = await index_files.run(index_files.Args(path="."), ctx(tmp_path))
    assert built.content.startswith("Indexed .: ") and built.content.endswith("-> .decision_workspace")
    found = await find.run(find.Args(question="Where is the Total due on the invoice?"), ctx(tmp_path))
    first, hint, text = found.content.splitlines()[1:4]
    assert first.startswith("1. reports/annual.pdf  page 4") and hint == "   read reports/annual.pdf pages=4"
    assert text == "   | Page 4. Total due 1,250 EUR."  # the best place's text comes with it
    found = await find.run(find.Args(question="sales amount per region"), ctx(tmp_path))
    assert 'sql query="SELECT * FROM reports__sales__sales LIMIT 20"' in found.content
    assert "   | 2 rows × 2 columns:\n   | region (VARCHAR) EU..US, ~2 distinct, 0.00% null: EU, US" in found.content
    nothing = await find.run(find.Args(question="zebra"), ctx(tmp_path))
    assert nothing.content.startswith("Nothing in the working directory looks related")

    totals = "select region, sum(amount) as total from reports__sales__sales group by 1 order by 1"
    rows = await sql.run(sql.Args(query=totals), ctx(tmp_path))
    assert rows.content == "| region | total |\n|---|---|\n| EU | 100.5 |\n| US | 200.0 |\n[kept as res_1: 2 rows]"
    capped = await sql.run(sql.Args(query="select * from reports__sales__sales", limit=1), ctx(tmp_path))
    assert capped.content.endswith("[kept as res_2: 2 rows, 1 shown]")
    bad = await sql.run(sql.Args(query="select * from nope"), ctx(tmp_path))
    assert bad.status == "error" and "Tables: reports__sales__sales, reports__sales__targets" in bad.content
    assert (await sql.run(sql.Args(query="drop table reports__sales__sales"), ctx(tmp_path))).status == "error"

    page = await read.run(read.Args(path="reports/annual.pdf", pages="4"), ctx(tmp_path))
    assert page.content.startswith("--- Page 4 ---\nPage 4. Total due 1,250 EUR.")
    whole = await read.run(read.Args(path="reports/annual.pdf"), ctx(tmp_path))
    assert whole.content.count("--- Page ") == 4
    tab = await read.run(read.Args(path="reports/sales.xlsx", tab="targets"), ctx(tmp_path))
    assert tab.content.startswith("--- Page 2 ---\n# tab Targets\n")
    missing = await read.run(read.Args(path="reports/sales.xlsx", tab="Costs"), ctx(tmp_path))
    assert missing.status == "error" and "Tabs: sales, targets" in missing.content
    out_of_range = await read.run(read.Args(path="reports/annual.pdf", pages="9"), ctx(tmp_path))
    assert out_of_range.status == "error"
    plain = await read.run(read.Args(path="notes.md", limit=1), ctx(tmp_path))
    assert plain.content.startswith("# Notes\n[Lines 1-1 of 3.")
    assert (await read.run(read.Args(path="reports"), ctx(tmp_path))).content == "annual.pdf\nsales.xlsx"

    word = await read.run(read.Args(path="resumes/li_yuchen.docx"), ctx(tmp_path))
    assert word.content == "\n".join(
        [
            "# Li Yuchen",
            "Candidate for the logistics role.",
            "## Education",
            "Beijing University, 2020.",
            "## Skills",
            "| skill | level |",
            "| forklift | expert |",
        ]
    )
    found = await find.run(find.Args(question="who studied at Beijing University?"), ctx(tmp_path))
    first, hint = found.content.splitlines()[1:3]
    assert first.startswith("1. resumes/li_yuchen.docx  ") and "Education (3-4)" in first
    assert hint == "   read resumes/li_yuchen.docx"  # a short file is opened whole


def test_the_capability_setting_turns_the_workspace_on_and_a_missing_extra_fails_setup(monkeypatch: pytest.MonkeyPatch):
    import importlib.util

    from intern.capability_settings import CapabilitySettings
    from intern.models.registry import ConfigError
    from intern.session_setup import capability_extensions

    on = CapabilitySettings.model_validate({"workspace": {"enabled": True}})
    assert capability_extensions(CapabilitySettings()) == []
    assert [type(e) for e in capability_extensions(on)] == [Workspace]
    both = CapabilitySettings.model_validate({"code_search": {"enabled": True}, "workspace": {"enabled": True}})
    assert [type(e).name for e in capability_extensions(both)] == ["code-search", "workspace"]
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "duckdb" else find_spec(name))
    with pytest.raises(ConfigError, match=r"install intern-agent\[workspace\]"):
        capability_extensions(on)


def test_the_workspace_is_a_capability_toggle_and_says_when_its_extra_is_missing(monkeypatch: pytest.MonkeyPatch):
    import importlib.util

    from intern import capability_toggles as ct
    from intern.capability_settings import CapabilitySettings
    from intern.session_setup import capability_problem

    off = CapabilitySettings()
    assert ct.values(off, off, [])["cap.workspace"] is False  # /caps and ACP list it
    on = ct.flip(off, off, "cap.workspace", True, [])
    assert on.workspace.enabled and ct.values(on, off, [])["cap.workspace"] is True
    assert not ct.flip(on, off, "cap.workspace", False, []).workspace.enabled
    assert capability_problem("cap.workspace", off, {}) is None
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "duckdb" else find_spec(name))
    assert "intern-agent[workspace]" in (capability_problem("cap.workspace", off, {}) or "")


def test_a_scanned_table_listed_column_by_column_reads_row_by_row():
    # A scan's text layer (or OCR) gives a table cell by cell, a column at a time.
    cells = [(10, 100, 40, 110, "1939"), (10, 120, 40, 130, "1940"), (60, 101, 90, 111, "8,432"),
             (60, 121, 90, 131, "8,736"), (100, 99, 120, 109, "557"), (100, 119, 120, 129, "1,478"),
             (10, 80, 30, 90, "Year"), (60, 80, 70, 90, "Total"), (72, 80, 92, 90, "net")]  # fmt: skip
    assert lines([(float(a), float(b), float(c), float(d), t) for a, b, c, d, t in cells]).split("\n") == [
        "Year  Total net",
        "1939  8,432  557",
        "1940  8,736  1,478",
    ]
    # Scanned askew (2%): across 12 columns a row falls by a row's pitch, yet each row stays one line.
    tilted = [
        (float(x), y + 0.02 * x, x + 40.0, y + 0.02 * x + 10, f"{row}.{column}")
        for row, y in enumerate((100.0, 112.0, 124.0, 136.0))
        for column, x in enumerate(range(0, 1200, 100))
    ]
    assert [line.split("  ")[0] for line in lines(tilted).split("\n")] == ["0.0", "1.0", "2.0", "3.0"]
    sideways = [(10.0, 10.0, 20.0, 80.0, "Receipts"), (30.0, 10.0, 40.0, 90.0, "Expenditures")]
    assert lines(sideways) == "Receipts\nExpenditures"  # text set sideways keeps its own order


async def test_a_scanned_page_is_read_with_its_ocr_kept_for_the_next_read(tmp_path: Path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in ("Table 7.- Social Security Act 19S2 -l,l73.6 3,8Z7.5", "Table 8.- Receipts " + "1*9 6I» " * 40):
        scan = doc.new_page()  # a scan whose text layer is an older, worse OCR
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 80), False)
        pixmap.clear_with(200)
        scan.insert_image(scan.rect, pixmap=pixmap)
        scan.insert_textbox(pymupdf.Rect(36, 36, 560, 800), text, fontsize=8)
    doc.new_page().insert_text((72, 72), "Narrative. " * 60)
    doc.save(tmp_path / "bulletin.pdf")
    build(tmp_path)
    asked: list[list[int]] = []

    def ocr_pages(path: Path, home: Path, numbers: list[int]) -> dict[int, str]:  # in this process, not a worker
        asked.append(numbers)
        return docs._ocr_pages(path, home, numbers)

    boxes = [
        (10.0, 0.0, 140.0, 8.0, "Table 7.- Social Security Act"),
        (10.0, 10.0, 40.0, 20.0, "1952"),
        (50.0, 10.0, 90.0, 20.0, "-1,173.6"),
        (100.0, 10.0, 140.0, 20.0, "3,827。5"),
    ]
    lost = [(10.0, 0.0, 140.0, 8.0, "Table 8.- Receipts")]  # an OCR that lost most of the page
    monkeypatch.setattr(docs.PageOcr, "_recognize", lambda self, page: lost if page.number else boxes)
    monkeypatch.setattr(docs.PageOcr, "available", lambda self: True)
    monkeypatch.setattr(docs, "ocr_pages", ocr_pages)
    read = Workspace().tools()[3]
    shown = await read.run(read.Args(path="bulletin.pdf", pages="1-3"), ctx(tmp_path))
    assert "1952  -1,173.6  3,827.5" in shown.content and "19S2" not in shown.content
    assert "1*9" in shown.content and "Narrative." in shown.content  # the text layer stays when OCR lost it
    assert asked == [[1, 2]]  # the page with text was not sent
    again = await read.run(read.Args(path="bulletin.pdf", pages="1-2"), ctx(tmp_path))
    assert "3,827.5" in again.content and asked == [[1, 2]]  # kept in the derived text
    whole = await read.run(read.Args(path="bulletin.pdf"), ctx(tmp_path))
    assert "3,827.5" in whole.content


def test_ocr_boxes_are_kept_by_content_so_no_page_is_recognized_twice(tmp_path: Path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page()
    doc.save(tmp_path / "scan.pdf")
    (tmp_path / "copy").mkdir()
    shutil.copy(tmp_path / "scan.pdf", tmp_path / "copy" / "scan.pdf")  # another path and mtime, same content
    calls: list[str] = []

    def recognize(self: docs.PageOcr, page: object) -> list[docs.Box]:
        calls.append(self.path.parent.name)
        return [(10.0, 10.0, 50.0, 20.0, "Receipts"), (60.0, 11.0, 90.0, 21.0, "1,250。0")]

    monkeypatch.setattr(docs.PageOcr, "_recognize", recognize)
    monkeypatch.setattr(docs.PageOcr, "available", lambda self: True)
    home = tmp_path / "home"
    texts = []
    for path in (tmp_path / "scan.pdf", tmp_path / "copy" / "scan.pdf"):
        with pymupdf.open(path) as pdf:
            texts.append(docs.PageOcr(path, home)(pdf[0]))
    assert texts == ["Receipts  1,250.0", "Receipts  1,250.0"] and calls == [tmp_path.name]
    monkeypatch.setattr(docs.PageOcr, "available", lambda self: False)  # kept pages need no OCR installed
    with pymupdf.open(tmp_path / "scan.pdf") as pdf:
        assert docs.PageOcr(tmp_path / "scan.pdf", home)(pdf[0]) == "Receipts  1,250.0"
        assert docs.PageOcr(tmp_path / "scan.pdf", tmp_path / "elsewhere")(pdf[0]) is None


async def test_the_note_names_the_folders_that_hold_documents_and_a_code_folder_gets_none(make_harness):
    from intern.providers.fake import Turn

    async def head() -> str:
        harness = make_harness([Turn(text="done")], extensions=[Workspace()])
        await harness.run("what was the 2024 revenue")
        return harness.provider.requests[0].messages[1].content

    cwd = make_harness([]).cwd
    (cwd / "app.py").write_text("x = 1\n")
    assert "index_files" not in await head()  # a folder of code: nothing pinned
    (cwd / "2024").mkdir()
    (cwd / "2023" / Workspace.folder).mkdir(parents=True)  # indexed by an earlier session
    for folder, n in (("2024", 3), ("2023", 2)):
        for i in range(n):
            (cwd / folder / f"r{i}.pdf").write_bytes(b"%PDF")
    (cwd / "ledger.xlsx").write_bytes(b"PK")
    said = await head()
    assert "index_files" in said and "2024/ 3, 2023/ 2 (indexed), ./ 1" in said
