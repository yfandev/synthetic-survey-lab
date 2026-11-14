"""`decision_files`' panel: the workspace extension's own view, built from the union's trace, in a real run of the TUI
and replayed from a recorded search."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("networkx", reason="needs the workspace extra: uv sync --all-extras")

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from intern.decisions import DecisionClient
from intern.extensions.code_search.view import Control
from intern.extensions.workspace import search as ws
from intern.extensions.workspace.extension import Workspace
from intern.extensions.workspace.trace import UnionTrace
from intern.extensions.workspace.view import FilesPanel, FilesState
from intern.providers.fake import Turn
from intern.tui.app import HarnessApp
from intern.tui.theme import LIGHT_THEME
from intern.tui.views import ViewHost
from intern.tui.widgets import ToolBlock, Transcript

from .test_tui import SIZE, FakeDecision, make_opener, plain, send, wait_for
from .test_workspace_search import BULLETINS, Script, build, bulletin_card, bulletin_passage

QUESTION = "What was the public debt held in February 1954?"


async def recorded(tmp_path: Path) -> tuple[list[UnionTrace], list[str]]:
    """A scripted union search's traces, and the paths of the places it ranked."""
    index = build(tmp_path, BULLETINS)
    traces: list[UnionTrace] = []
    client = DecisionClient(Script(bulletin_card, bulletin_passage), lambda e: None)
    result = await ws.union(index, tmp_path, QUESTION, client, Workspace.asks, trace=traces.append)
    return traces, [p.path for p in result.places]


def merged(traces: list[UnionTrace]) -> FilesState:
    state = FilesState(call_id="c1", question=QUESTION)
    for trace in traces:
        state.merge(trace, now=0.0)
    return state


async def test_the_panels_state_gives_the_ranking_with_each_passages_lane_whatever_order_the_traces_come_in(
    tmp_path: Path,
):
    traces, places = await recorded(tmp_path)
    state = merged(traces)
    top = state.top()
    assert [h.path for h in top] == places[: len(top)] and state.finished and state.stage == "done"
    lanes = {h.path: h.lanes for h in top}
    assert lanes["staff/quill.md"] == ["term"] and state.missed(next(h for h in top if h.path == "staff/quill.md"))
    assert "walk" in lanes["bulletins/feb_1953.md"]
    # Late, repeated and shuffled traces leave the same picture: each passage and node at its newest.
    shuffled = traces + traces[::-1] + random.Random(4).sample(traces, len(traces))
    again = merged(shuffled)
    assert again.hits == state.hits and again.walk.nodes == state.walk.nodes
    assert [h.id for h in again.top()] == [h.id for h in top] and again.terms == state.terms
    assert again.stage == "done" and again.last == traces[-1] and len(again.history) == len(traces)


class Bare(App[Any]):
    def __init__(self, state: FilesState) -> None:
        super().__init__()
        self.register_theme(LIGHT_THEME)
        self.theme = LIGHT_THEME.name
        self.state = state

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield FilesPanel(ViewHost(self), self.state)


@pytest.mark.parametrize("width", [80, 200])
async def test_a_recorded_search_draws_folds_and_replays_at_any_width(tmp_path: Path, width: int):
    traces, places = await recorded(tmp_path)
    for upto in (len(traces) // 2, len(traces)):  # mid-search, then ranked
        state = merged(traces[:upto])
        if upto == len(traces):
            state.ended("ok", "")
        app = Bare(state)
        async with app.run_test(size=(width, 60)) as pilot:
            panel = app.query_one(FilesPanel)
            await pilot.pause()
            panel.now()
            head = plain(panel.query_one(".dxf-head", Static))
            podium = plain(panel.query_one(".dxf-podium", Static))
            assert "System 1 searches the workspace" in head
            if upto < len(traces):
                assert "✓ ranked" not in head
                continue
            assert "✓ ranked in" in head and "RESULT" in podium and places[0] in podium
            assert "HOP" in plain(panel.query_one(".dxf-flow", Static))
            panel.toggle_fold()
            await pilot.pause()
            assert plain(panel.query_one(".dxf-head", Static)).startswith("⌕ workspace search")
            assert not panel.query_one(".dxf-lanes").display
            panel.toggle_fold()
            panel.start_replay()
            assert panel.replay is not None and panel.shown is not state
            await wait_for(lambda panel=panel: panel.replay is None)
            assert panel.shown is state


async def test_a_decision_files_call_draws_its_search_under_the_call_in_the_tui(tmp_path: Path):
    work = tmp_path / "work"
    for rel, text in BULLETINS.items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(text)
    script = [
        Turn(tool_calls=[("index_files", {"path": "."})]),
        Turn(tool_calls=[("decision_files", {"question": QUESTION})]),
        Turn(text="271 billion."),
    ]
    app = HarnessApp(make_opener(tmp_path, script, system1=FakeDecision(), extensions=(Workspace,)))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await send(app, pilot, "what was the debt?")
        await wait_for(lambda: (task := app._run_task) is not None and task.done(), 30)  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        blocks = list(app.query_one(Transcript).query(ToolBlock))
        block = next(b for b in blocks if b.query(FilesPanel))
        panel = block.query_one(FilesPanel)
        assert list(block.children).index(panel) < list(block.children).index(block.out)  # between call and output
        assert panel.state.question == QUESTION and panel.state.finished and panel.state.status == "ok"
        assert panel.state.bm25() and panel.state.top()
        await wait_for(lambda: "✓ ranked in" in plain(panel.query_one(".dxf-head", Static)))
        assert isinstance(panel.query_one(".dxf-replay", Control), Control)
        await app.action_quit()
