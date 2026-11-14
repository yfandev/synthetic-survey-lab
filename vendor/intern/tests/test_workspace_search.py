"""decision_files' union search (`extensions/workspace/search.py`): the walk's chunks judged on their text side by side,
a BM25 first stage that reaches what the walk pruned, one expansion round, a table found by its columns, the top ten
in diversity order. Scripted System 1s judge a card by its words and a passage by what it says (and by the evidence
in the state), as System 1 would; they prove the plumbing, not the ranking quality."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("networkx", reason="needs the workspace extra")

from intern.decisions import (
    Answer,
    Answers,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionClient,
    NoulAnswer,
    Question,
)
from intern.extensions.workspace import search as ws
from intern.extensions.workspace.documents import office_processors
from intern.extensions.workspace.extension import Workspace
from intern.extensions.workspace.trace import STAGES, UnionTrace
from intern.locate import index as wi
from intern.locate.search import search


def build(root: Path, files: dict[str, str]) -> wi.Index:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return wi.build(root, root / Workspace.folder, processors=office_processors, report=lambda line: None)


class Script:
    """A System 1 that judges a card (the walk's asks) by ``card`` and a passage (the union's) by ``passage``, given
    the passage and the state; every question gets an answer. Records each passage judgment's state and text."""

    name = "script"

    def __init__(self, card: Callable[[str], float], passage: Callable[[str, str], float]) -> None:
        self.card, self.passage = card, passage
        self.judged: list[tuple[str, str]] = []
        self.asks: list[str] = []

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        self.asks.append(state)
        body = state.split("\n\n", 1)[1]
        parts = re.split(r"^\[(\w+)\] ", body, flags=re.M)
        blocks = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
        answers: dict[str, Answer] = {}
        for name, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                best = max((c for c in q.criteria if c in blocks), key=lambda c: self.card(blocks[c]), default=None)
                best = best or next(iter(q.criteria))
                answers[name] = ChoiceAnswer(choice=best, probabilities={c: float(c == best) for c in q.criteria})
            elif "\nPassage [" in q.instructions:
                text = q.instructions.split("\nPassage [", 1)[1]
                self.judged.append((state, text))
                answers[name] = NoulAnswer(noul=self.passage(text, state))
            else:
                answers[name] = NoulAnswer(noul=self.card(blocks.get(name, body)))
        return Answers(answers=answers)


BULLETINS = {
    "bulletins/feb_1953.md": "# Treasury bulletin February 1953\n\nPublic debt table. Debt held: 266 billion.\n",
    "bulletins/feb_1954.md": (
        "# Treasury bulletin February 1954\n\nPublic debt table. Debt held: 271 billion. Compiled by Ada Quill.\n"
    ),
    "staff/quill.md": "# Staff\n\nAda Quill also kept the gold reserve ledger.\n",
    "staff/vaults.md": "# Vaults\n\nThe gold reserve is kept in vaults.\n",
}


def bulletin_card(text: str) -> float:
    """The walk prefers the wrong month: a card naming 1953 scores above one naming the public debt."""
    return 1.0 if "1953" in text else 0.6 if "Public debt" in text else 0.0


def bulletin_passage(text: str, state: str) -> float:
    """The debt for 1954 is the evidence; Ada Quill's line counts once a passage in the state names her."""
    if "Debt held" in text and "1954" in text:
        return 0.9
    return 0.8 if "Ada Quill" in text and "Ada Quill" in state else 0.05


async def test_the_walks_chunks_are_judged_on_their_text_side_by_side_and_expansion_reaches_what_it_pruned(
    tmp_path: Path,
):
    index = build(tmp_path, BULLETINS)
    question = "What was the public debt held in February 1954?"
    walked = await search(
        index,
        tmp_path,
        question,
        DecisionClient(Script(bulletin_card, bulletin_passage), lambda e: None),
        Workspace.asks,
    )
    assert walked.places[0].path == "bulletins/feb_1953.md"  # the walk alone: the wrong month first
    judge = Script(bulletin_card, bulletin_passage)
    result = await ws.union(index, tmp_path, question, DecisionClient(judge, lambda e: None), Workspace.asks)
    best = [p.path for p in result.places if p.score > 0.5]
    assert best[0] == "bulletins/feb_1954.md"  # its text, judged with the other month's beside it
    assert any("Debt held: 271" in text and "Debt held: 266" in state for state, text in judge.judged)
    assert "staff/quill.md" in best  # its card names neither: pruned by the walk, reached through the seed's name


async def test_the_bm25_stage_reaches_a_passage_the_walk_pruned_from_the_questions_own_words(tmp_path: Path):
    index = build(
        tmp_path,
        {
            "reports/summary.md": "# Summary\n\nQuarterly summary of operations.\n",
            "archive/minutes.md": "# Minutes\n\nThe board approved the Zorblax merger on 3 March.\n",
            "archive/agenda.md": "# Agenda\n\nItems for the next meeting.\n",
        },
    )
    judge = Script(
        lambda card: 1.0 if "Summary" in card else 0.0,  # the walk prunes the archive
        lambda text, state: 0.9 if "Zorblax merger" in text else 0.05,
    )
    question = "When did the board approve the Zorblax merger?"
    walked = await search(index, tmp_path, question, DecisionClient(judge, lambda e: None), Workspace.asks)
    assert "archive/minutes.md" not in [p.path for p in walked.places if p.score > 0.5]
    result = await ws.union(index, tmp_path, question, DecisionClient(judge, lambda e: None), Workspace.asks)
    assert result.places[0].path == "archive/minutes.md" and result.places[0].score > 0.5


async def test_the_trace_only_watches_the_ranking_scores_and_asks_are_the_same_without_it(tmp_path: Path):
    index = build(tmp_path, BULLETINS)
    question = "What was the public debt held in February 1954?"
    runs = []
    for sink in (None, lambda trace: None):
        judge = Script(bulletin_card, bulletin_passage)
        result = await ws.union(
            index, tmp_path, question, DecisionClient(judge, lambda e: None), Workspace.asks, trace=sink
        )
        runs.append((result.model_dump(), sorted(judge.asks), sorted(judge.judged)))
    assert runs[0] == runs[1]


async def test_the_trace_carries_every_stage_of_the_union(tmp_path: Path):
    index = build(tmp_path, BULLETINS)
    traces: list[UnionTrace] = []
    question = "What was the public debt held in February 1954?"
    judge = Script(bulletin_card, bulletin_passage)
    result = await ws.union(
        index, tmp_path, question, DecisionClient(judge, lambda e: None), Workspace.asks, trace=traces.append
    )
    hits = {h.id: h for t in traces for h in t.hits}
    nodes = {n.id: n for t in traces for n in t.nodes}
    terms = next(t.terms for t in traces if t.terms)
    assert {t.stage for t in traces} == set(STAGES) and traces[-1].finished
    assert {n.kind for n in nodes.values()} >= {"root", "file", "part"}  # the walk's tree, as decision_code's
    assert any(n.read is not None for n in nodes.values())  # its parts' text, judged by the union
    lanes = {lane for h in hits.values() for lane in h.lanes}
    assert lanes == {"walk", "bm25", "term"}
    assert all(h.bm25 for h in hits.values() if "bm25" in h.lanes)
    judged = [h for h in hits.values() if h.state == "judged"]
    assert judged and all(h.score is not None for h in judged)
    assert traces[-1].asks == len(judge.asks)
    seeds = {h.id for h in hits.values() if h.seed}
    assert seeds and {t.seed for t in terms} <= seeds
    reached = [i for t in terms for i in t.hits]
    assert reached and all(hits[i].term in {t.term for t in terms} and "term" in hits[i].lanes for i in reached)
    assert any(i.startswith("staff/quill.md#") for i in reached)  # pruned by the walk, reached through a term
    top = sorted((h for h in hits.values() if h.rank), key=lambda h: h.rank or 0)
    assert [h.rank for h in top] == list(range(1, len(top) + 1))
    assert [h.path for h in top] == [p.path for p in result.places[: len(top)]]
    assert all(h.lanes for h in top)


def test_tables_and_images_are_passages_too(tmp_path: Path):
    pytest.importorskip("duckdb", reason="needs the workspace extra")
    image = pytest.importorskip("PIL.Image", reason="needs the workspace extra")
    rows = "".join(f"C{n:02d},{n}\n" for n in range(40))  # ZAR comes after the rows a table's page shows
    (tmp_path / "scans").mkdir()
    image.new("RGB", (40, 20)).save(tmp_path / "scans" / "org_chart.png")
    index = build(tmp_path, {"ledger/rates.csv": f"currency,exchange_rate\n{rows}ZAR,99\n"})
    table = ws.Table(index, tmp_path)
    con = table.open()
    try:
        rate = table.search(con, ws.match("the ZAR exchange rate"), set(), 10, None)  # its profile's range names it
        chart = table.search(con, ws.match("the org chart"), set(), 10, None)
    finally:
        con.close()
    table_key = index.files["ledger/rates.csv"].tables[0].name  # its place opens with sql, not read
    assert [(path, key) for _, path, key in rate] == [("ledger/rates.csv", table_key)]
    assert [path for _, path, _ in chart] == ["scans/org_chart.png"]


def test_a_scope_holds_its_own_folder_only(tmp_path: Path):
    # `_` and `%` in a folder's name are characters, not wildcards; a letter's case counts.
    files = {
        f"{folder}/minutes.txt": "The board approved the merger.\n"
        for folder in ("data_2024", "dataX2024", "Data_2024")
    }
    index = build(tmp_path, files)
    table = ws.Table(index, tmp_path)
    con = table.open()
    try:
        hits = table.search(con, ws.match("board merger"), set(), 10, "data_2024")
    finally:
        con.close()
    assert [path for _, path, _ in hits] == ["data_2024/minutes.txt"]


class Slow(Script):
    """The script, judging passages after a delay past its client's limit for an ask (cards at once)."""

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        if any("\nPassage [" in getattr(q, "instructions", "") for q in questions.values()):
            await asyncio.sleep(0.2)
        return await super().decide(state, questions)


async def test_a_judgment_may_take_longer_than_the_clients_limit_for_an_ask(tmp_path: Path):
    index = build(tmp_path, BULLETINS)
    client = DecisionClient(Slow(bulletin_card, bulletin_passage), lambda e: None, timeout_s=0.1)
    result = await ws.union(index, tmp_path, "What was the public debt held in February 1954?", client, Workspace.asks)
    assert result.places[0].path == "bulletins/feb_1954.md" and result.places[0].score > 0.5


def test_the_top_ten_are_put_in_diversity_order():
    same = "treasury receipts expenditures fiscal"
    top = [
        ws.Passage(1, "a", "", "a", same, score=0.9),
        ws.Passage(2, "b", "", "b", same, score=0.88),
        ws.Passage(3, "c", "", "c", "customs duties tariff schedule", score=0.8),
    ]
    idf = {w: 1.0 for p in top for w in p.body.split()}
    assert [p.id for p in ws.diversity_order(top, idf)] == [1, 3, 2]  # the copy of the best goes after the other
