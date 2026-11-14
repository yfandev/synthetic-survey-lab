"""`decision_code`'s window: the code-search extension's own view, through real runs of the TUI on a real index."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("networkx", reason="needs the code-search extra: uv sync --all-extras")

from textual.screen import ModalScreen
from textual.widgets import Static

from intern.events import ExtensionEvent
from intern.extensions.code_search import CodeSearch
from intern.extensions.code_search.view import Control, SearchPanel, SearchState, short_label, tree, tree_rows
from intern.journal import EventEntry, Journal
from intern.locate.trace import SearchNode, SearchTrace
from intern.messages import JsonObject
from intern.providers.fake import Turn
from intern.tui.app import HarnessApp
from intern.tui.widgets import ToolBlock, Transcript

from .test_tui import SIZE, FakeDecision, make_opener, plain, send, transcript_text, wait_for

QUESTION = "where are dates parsed?"


def workdir(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "app").mkdir(parents=True)
    (work / "app" / "dates.py").write_text(
        textwrap.dedent('''
        """Dates from the wire."""

        def parse_date(text):
            """A date from ISO text."""
            return text.split("-")


        def format_date(parts):
            return "-".join(parts)
        ''')
    )
    (work / "app" / "server.py").write_text('"""The HTTP server."""\n\ndef serve(port):\n    return port\n')
    (work / "README.md").write_text("# Example\n\nA small app.\n\n## Dates\n\nParsed in app/dates.py.\n")


def code_call(question: str = QUESTION, **more: str) -> tuple[str, JsonObject]:
    arguments: JsonObject = {"question": question, **more}
    return ("decision_code", arguments)


async def run_prompt(app: HarnessApp, pilot, text: str) -> None:
    """Send `text` and wait for the run it starts to end."""
    before = app._run_task  # pyright: ignore[reportPrivateUsage]
    await send(app, pilot, text)
    await wait_for(lambda: (task := app._run_task) is not None and task is not before and task.done(), 30)  # pyright: ignore[reportPrivateUsage]
    await pilot.pause()


def panel_of(app: HarnessApp, index: int = 0) -> tuple[ToolBlock, SearchPanel]:
    block = list(app.query_one(Transcript).query(ToolBlock))[index]
    return block, block.query_one(SearchPanel)


def text_of(panel: SearchPanel, selector: str) -> str:
    return plain(panel.query_one(selector, Static))


def no_window(app: HarnessApp) -> bool:
    return len(app.screen_stack) == 1 and not isinstance(app.screen, ModalScreen)


async def test_a_decision_code_call_draws_its_search_live_right_under_the_call(tmp_path: Path):
    workdir(tmp_path)
    journal = tmp_path / "session.jsonl"
    script = [Turn(tool_calls=[code_call()]), Turn(text="In app/dates.py.")]
    app = HarnessApp(make_opener(tmp_path, script, system1=FakeDecision(), journal=journal, extensions=(CodeSearch,)))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await send(app, pilot, "find the date parser")
        await wait_for(lambda: len(app.query(SearchPanel)) == 1)  # under the call as soon as it starts
        assert no_window(app)
        await wait_for(lambda: app._run_task is not None and app._run_task.done(), 30)  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        assert no_window(app)  # never a window of its own: it all stays in the transcript
        block, panel = panel_of(app)
        assert list(block.children).index(panel) < list(block.children).index(block.out)  # between call and output
        state = panel.state
        assert state.question == QUESTION and state.finished and state.status == "ok"
        await wait_for(lambda: "✓ ranked in" in text_of(panel, ".decision-head"))
        head = text_of(panel, ".decision-head")
        assert "System 1 walks the index" in head and "● descend" in head and "● ranked" in head
        assert "RESULT" in text_of(panel, ".decision-board") and "dates.py" in text_of(panel, ".decision-board")
        lines = text_of(panel, ".decision-tree").splitlines()
        assert lines[0].startswith("◉ ") and "3 files" in lines[0]
        assert any("dates.py" in line and "0.20" in line for line in lines)
        assert any("def parse_date" in line for line in lines)
        assert "★" in "\n".join(lines) and lines[-1].startswith("┿ cut (best × 0.25)")  # the result, the legend
        assert "asks" in text_of(panel, ".decision-stats")
        assert "Places for: where are dates parsed?" in transcript_text(app)  # the tool's result, as before
        # Fold to one line and back.
        panel.query_one(".decision-fold", Control).on_click(_click())
        await pilot.pause()
        assert panel.has_class("-folded") and not panel.query_one(".decision-body").display
        assert text_of(panel, ".decision-head").startswith("⌕ search tree") and "▸ unfold" in text_of(
            panel, ".decision-fold"
        )
        panel.query_one(".decision-fold", Control).on_click(_click())
        await pilot.pause()
        assert not panel.has_class("-folded") and "System 1 walks the index" in text_of(panel, ".decision-head")
        # A replay plays the recorded search again, in place.
        # A click draws at once; the replay of a search that took milliseconds could end between two frames.
        panel.query_one(".decision-replay", Control).on_click(_click())
        assert panel.replay is not None and panel.shown is not state
        assert "⟲ replay ×2" in text_of(panel, ".decision-head")
        await wait_for(lambda: panel.replay is None)
        assert panel.shown is state and no_window(app)
        await app.action_quit()
    entries = Journal(cwd=str(tmp_path), path=journal).active_path()
    assert not [e for e in entries if isinstance(e, EventEntry) and isinstance(e.event, ExtensionEvent)]


async def test_searches_side_by_side_each_draw_under_their_own_call(tmp_path: Path):
    workdir(tmp_path)
    script = [
        Turn(tool_calls=[code_call(), code_call("where is the server started?", scope="app")]),
        Turn(text="Both found."),
        Turn(tool_calls=[code_call("where is the readme?")]),
        Turn(text="Found."),
    ]
    app = HarnessApp(make_opener(tmp_path, script, system1=FakeDecision(), extensions=(CodeSearch,)))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await run_prompt(app, pilot, "find both")
        (_, first), (_, second) = panel_of(app, 0), panel_of(app, 1)
        assert (first.state.question, second.state.question) == (QUESTION, "where is the server started?")
        assert second.state.scope == "app" and "under app" in text_of(second, ".decision-head")
        assert first.state.finished and second.state.finished and first.state.search != second.state.search
        assert all(n.label.startswith("app/") for n in second.state.nodes.values() if n.kind == "file")
        await run_prompt(app, pilot, "the readme")
        _, third = panel_of(app, 2)
        assert third.state.question == "where is the readme?" and third.state.finished
        assert no_window(app)
        await app.action_quit()


async def test_a_search_that_cannot_run_says_why_under_its_call(tmp_path: Path):
    workdir(tmp_path)
    script = [Turn(tool_calls=[code_call(scope="../gone")]), Turn(text="No luck.")]
    app = HarnessApp(make_opener(tmp_path, script, extensions=(CodeSearch,)))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await run_prompt(app, pilot, "find the date parser")
        _, panel = panel_of(app)
        await wait_for(lambda: "✗ error" in text_of(panel, ".decision-head"))
        assert panel.state.status == "error" and not panel.state.nodes
        drawn = text_of(panel, ".decision-tree")
        assert drawn.startswith("✗ decision_code did not search:") and "does not exist" in drawn
        await app.action_quit()


def test_the_tree_shows_each_node_under_its_parent_by_what_it_adds_best_first_with_the_rest_folded():
    state = SearchState(call_id="c1", question="q")
    nodes = [
        SearchNode(id="n:.", kind="root", label=".", detail="40 files", state="opened"),
        SearchNode(id="n:src", parent="n:.", kind="folder", label="src", state="opened", score=0.9, cut=0.22),
        SearchNode(id="n:docs", parent="n:.", kind="folder", label="docs", state="pruned", score=0.1),
        SearchNode(id="n:src/a", parent="n:src", kind="group", label="src · [(files 1/2)]", state="opened", score=0.8),
        SearchNode(id="f:src/x.py", parent="n:src/a", kind="file", label="src/x.py", state="chosen", score=0.7),
        *[
            SearchNode(id=f"c:{i}", parent="f:src/x.py", kind="part", label=f"def f{i}", state="scored", score=i / 10)
            for i in range(7)
        ],
    ]
    state.merge(SearchTrace(search="s", question="q", phase="parts", nodes=nodes), now=0.0)
    rows = tree_rows(state, show_pruned=True)
    labels = [short_label(state, n) if n else summary for _, n, summary in rows]
    assert labels[:4] == [".", "src/", "files 1/2", "src/x.py".removeprefix("src/")]
    assert labels[4:8] == ["def f6", "def f5", "def f4", "def f3"]  # the best parts first
    assert labels[8] == "… 3 more parts, best 0.20"
    assert labels[9] == "docs/"  # pruned, after the live ones
    hidden = [short_label(state, n) if n else summary for _, n, summary in tree_rows(state, show_pruned=False)]
    assert "docs/" not in hidden and "… 1 more pruned, best 0.10" in hidden
    content, hidden = tree(state, 100, now=10.0, frame=0, show_pruned=True)
    drawn = content.plain.splitlines()
    assert hidden == 0 and len(drawn) == len(rows)
    assert "┿" in drawn[2]  # the group's bar shows its parent's cut
    assert drawn[3].lstrip("│ ├└─").startswith("■ x.py") and "★" in drawn[3]
    capped, hidden = tree(state, 100, now=10.0, frame=0, show_pruned=True, max_rows=4)
    assert len(capped.plain.splitlines()) == 4 and hidden == len(rows) - 4  # the least likely rows, last


def _click():
    from textual.events import Click

    return Click(widget=None, x=0, y=0, delta_x=0, delta_y=0, button=1, shift=False, meta=False, ctrl=False)
