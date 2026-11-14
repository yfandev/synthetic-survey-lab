"""The terminal UI over a scripted provider, driven through Textual's pilot."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("textual", reason="needs the tui extra")

from textual.app import App
from textual.content import Content
from textual.errors import NoWidget
from textual.events import Paste
from textual.widgets import OptionList, Static, TextArea

from intern.capability_settings import CapabilitySettings
from intern.cli import main as cli_main
from intern.config import Options, System2Unavailable
from intern.decisions import (
    Answer,
    Answers,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionBackend,
    DecisionClient,
    NoulAnswer,
    Question,
)
from intern.events import (
    Compaction,
    EventSink,
    Indexing,
    State,
    StateDelta,
    StateWriting,
    ToolCallEnd,
    ToolCallStart,
    ToolGuard,
)
from intern.executor import LocalExecutor
from intern.hooks import Extension, Hooks
from intern.instructions import FileInstructions
from intern.journal import Journal
from intern.loop import EngineDeps, ModelBinding
from intern.models.catalog import Catalog, ModelInfo
from intern.models.profile import EFFORT_LEVELS, Profile, Thresholds
from intern.models.registry import ConfigError
from intern.providers.fake import FakeProvider, Script, Turn
from intern.session import Session
from intern.session_cut import STATE_SHARE, session_cut
from intern.tools import load_packs
from intern.tui import app as app_module
from intern.tui import widgets
from intern.tui.app import (
    ALT_SCREEN_OFF,
    ALT_SCREEN_ON,
    PICKS,
    SHOW_CURSOR,
    HarnessApp,
    HarnessEvent,
    Models,
    StateScreen,
)
from intern.tui.pickers import Choice, ModelPicker, ranked
from intern.tui.state_panel import StatePanel
from intern.tui.widgets import (
    CompactionCard,
    Prompt,
    StatusBar,
    ToolBlock,
    Transcript,
)

from .support import Filled, Quiet, binding, is_state_write
from .test_tui_logic import SAMPLE_STATE, guard, run_start

SIZE = (160, 48)


class FakeDecision:
    """Answers every choice with its first option at 0.8 and every noul with 0.2, so no State write follows."""

    name = "decisions-api:fake-decision"

    def __init__(self) -> None:
        self.asks = 0

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        self.asks += 1
        answers: dict[str, Answer] = {}
        for name, question in questions.items():
            if isinstance(question, ChoiceQuestion):
                first = next(iter(question.criteria))
                probabilities = {c: (0.8 if c == first else 0.05) for c in question.criteria}
                answers[name] = ChoiceAnswer(choice=first, probabilities=probabilities)
            else:
                answers[name] = NoulAnswer(noul=0.2)
        return Answers(answers=answers, cost_usd=0.0001)


class Named(Quiet):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class Thinking(FakeProvider):
    """A provider whose API takes an effort for every model, the whole ladder, as the Anthropic provider says."""

    effort_levels = EFFORT_LEVELS


def make_opener(
    tmp_path: Path,
    script: Script,
    *,
    system1: DecisionBackend | None = None,
    journal: Path | None = None,
    configured_system2: str | None = "fake/model",
    takes_effort: bool = False,
    profile: Profile | None = None,
    providers: dict[str, FakeProvider] | None = None,
    extensions: tuple[type[Extension], ...] = (),
):
    """An opener over the scripted provider (`providers` gives some refs one of their own; `extensions`, made anew
    for each session, with their tools). With `configured_system2` None, opening without a pick fails the way
    Bootstrap does when nothing names a System 2."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    provider = (Thinking if takes_effort else FakeProvider)(script)

    async def open_session(emit: EventSink, models: Models) -> Session:
        async def resolve(ref: str) -> ModelBinding:
            if not ref.startswith("fake/"):
                raise ValueError(f"unknown provider in {ref!r}")
            bound = binding((providers or {}).get(ref, provider), profile=profile.model_copy() if profile else None)
            bound.ref = ref
            return bound

        system2 = models.system2 or configured_system2
        if system2 is None:
            raise System2Unavailable("no model given: pass --system2 provider/model", ref="", system1="drex-latest")
        made = [cls() for cls in extensions]
        deps = EngineDeps(
            system2=await resolve(system2),
            tools=load_packs(["base"]) + [tool for extension in made for tool in extension.tools()],
            hooks=Hooks(made, emit),
            journal=Journal(cwd=str(work), path=journal),
            instructions=FileInstructions(tmp_path / "home", context_files=False).load(work),
            decisions=DecisionClient(Filled(system1) if system1 else Quiet(), emit),
            executor=LocalExecutor(),
            cwd=work,
            spill_dir=tmp_path / "spill",
            emit=emit,
        )
        session = Session(
            deps,
            resolve_model=resolve,
            resolve_system1=Named,
            system1="drex-latest",
        )
        if models.system1:
            await session.set_model(models.system1, "system1")
        return session

    return open_session


async def wait_for(condition: Callable[[], bool], limit_s: float = 10) -> None:
    deadline = time.monotonic() + limit_s
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("the condition did not hold in time")
        await asyncio.sleep(0.02)


def plain(widget: object) -> str:
    """A widget's text, the non-breaking spaces inside a card's labelled levels read as spaces."""
    content = getattr(widget, "content", "")
    return (content.plain if isinstance(content, Content) else str(content)).replace(" ", " ")


def transcript_text(app: HarnessApp) -> str:
    lines: list[str] = []
    for widget in app.query_one(Transcript).query(Static):
        lines.append(plain(widget))
    return "\n".join(lines)


async def click_compaction(app: HarnessApp, pilot) -> None:
    """Click the latest compaction's card, once it is on screen."""
    card = app.query(CompactionCard).last()

    def on_screen() -> bool:
        try:
            widget, _ = app.screen.get_widget_at(*card.region.offset)
        except NoWidget:
            return False
        return widget is card or card in widget.ancestors

    await wait_for(on_screen)
    await pilot.click(card)
    await pilot.pause()


async def send(app: HarnessApp, pilot, text: str) -> None:
    app.query_one("#prompt", Prompt).text = text
    await pilot.press("enter")
    await pilot.pause()


async def test_a_prompt_runs_and_the_reply_tool_calls_and_system_1_rows_show(tmp_path: Path):
    script = [
        Turn(text="Let me look.", tool_calls=[("bash", {"command": "echo hi"})]),
        Turn(tool_calls=[("bash", {"command": "echo again"})]),
        Turn(tool_calls=[("bash", {"command": "echo third"})]),
        Turn(text="It printed **hi**."),
    ]
    decider = FakeDecision()
    app = HarnessApp(make_opener(tmp_path, script, system1=decider))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        status = app.query_one(StatusBar)
        assert (status.system2, status.system1) == ("fake/model", "drex-latest")
        await send(app, pilot, "say hi")
        await wait_for(lambda: app._run_task is not None and app._run_task.done())  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        text = transcript_text(app)
        assert "› say hi" in text
        assert "$ echo hi" in text and "$ echo third" in text and "Took 0." in text
        assert "hi" in plain(app.query_one(Transcript).query(ToolBlock).first().out)  # the output under the call
        assert "It printed" in "".join(m.source for m in app.query_one(Transcript).query(".reply"))  # pyright: ignore[reportAttributeAccessIssue]
        # System 1 was asked at the third turn (the first the supervisor reviews): its card is in the transcript at
        # the turn boundary, and the run is charted on its panel.
        assert decider.asks >= 1 and 3 in app.ledger.current.turns
        assert " turns · 1 prompt" in plain(app.query_one("#ledger-chart", Static))
        assert "s   advancing 0.80" in text
        assert "completed · 4 turns" in text
        assert not status.running
        await app.action_quit()


class Paced(FakeProvider):
    """Streams a word every few milliseconds, as a model does: the reply's Markdown is still being drawn when the
    next words arrive."""

    async def stream(self, request):
        async for item in super().stream(request):
            await asyncio.sleep(0.004)
            yield item


async def test_a_long_reply_streaming_in_keeps_its_end_on_screen(tmp_path: Path):
    reply = "Here they are:\n\n" + "\n".join(f"{i}. item {i} of the list" for i in range(1, 41))
    app = HarnessApp(make_opener(tmp_path, [], providers={"fake/model": Paced([Turn(text=reply)])}))
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for(lambda: app.session is not None)
        await send(app, pilot, "list forty things")
        await wait_for(lambda: app._run_task is not None and app._run_task.done())  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        transcript = app.query_one(Transcript)
        assert transcript.max_scroll_y > 0  # longer than the screen
        assert transcript.scroll_y == transcript.max_scroll_y  # the last item and the run's end in view, no PageDown
        await app.action_quit()


async def test_the_guards_verdicts_show_under_the_tool_calls_they_concern(tmp_path: Path):
    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")]))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        # System 1 judged a call and the guard refused it on that judgement.
        reason = "System 1 judged that it rewrites history that others share and that the user did not ask for it"
        events = [
            run_start("r1"),
            ToolCallStart(turn=2, call_id="c9", name="bash", arguments={"command": "git push --force origin main"}),
            guard(2, 'bash {"command": "git push --force origin main"}', "rewrites_shared_history", 0.02),
            ToolGuard(
                turn=2,
                call_id="c9",
                name="bash",
                layer="system1",
                rule="rewrites_shared_history",
                reason=reason,
                risk=0.96,
                asked=0.02,
            ),
            ToolCallEnd(
                turn=2,
                call_id="c9",
                name="bash",
                status="refused",
                content="the guard refused this call",
                duration_ms=1,
            ),
        ]
        for event in events:
            app.post_message(HarnessEvent(event))
        await pilot.pause()
        await pilot.pause()
        block = app.query_one(Transcript).query(ToolBlock).first()
        card = plain(block.query_one(".decision", Static))
        assert card.startswith("✗ System 1 refused   ")
        assert "rewrites shared history 0.96   risk 0.96   asked 0.02" in card
        assert reason in card and block.has_class("failed")
        # A fixed rule refused a call: no model was asked, and the card says which rule.
        events = [
            ToolCallStart(turn=3, call_id="c10", name="bash", arguments={"command": "rm -rf /"}),
            ToolGuard(turn=3, call_id="c10", name="bash", layer="rule", rule="delete_root", reason="deletes the root"),
        ]
        for event in events:
            app.post_message(HarnessEvent(event))
        await pilot.pause()
        await pilot.pause()
        cards = [plain(w) for w in app.query_one(Transcript).query(".decision")]
        assert any("✗ refused   rule · delete root" in c and "deletes the root" in c for c in cards), cards
        await app.action_quit()


async def test_capabilities_are_listed_and_a_switch_reopens_the_session_on_its_journal(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    opened: list[Models] = []
    inner = make_opener(tmp_path, [Turn(text="ok")], journal=journal)

    async def recording(emit: EventSink, picks: Models) -> Session:
        opened.append(Models(**vars(picks)))
        if picks.capabilities is not None and picks.capabilities.code_search.enabled:
            raise ConfigError("Code search needs the code-search extra: install intern-agent[code-search]")
        return await inner(emit, picks)

    settings = CapabilitySettings(tool_packs=["base", "web-read"])
    app = HarnessApp(recording, capabilities=settings, memory=tmp_path / "drive")
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        first = app.session
        await send(app, pilot, "/capa")  # a prefix that names one command
        assert isinstance(app.screen, ModelPicker)
        listed = {c.id: c.group for c in app.screen.choices}  # pyright: ignore[reportAttributeAccessIssue]
        assert listed["cap.base"] == "on" and listed["cap.web-read"] == "on" and listed["cap.web-search"] == "off"
        assert listed["cap.browser"] == "off" and listed["cap.code-search"] == "off" and listed["cap.memory"] == "on"
        assert listed["cap.workspace"] == "off"
        labels = {c.id: c.label for c in app.screen.choices}  # pyright: ignore[reportAttributeAccessIssue]
        assert labels["cap.base"] == "base tools" and labels["cap.code-search"] == "code search · System 1"
        assert labels["cap.memory"] == "memory drive"
        assert not any(row.startswith(("cap.docker", "cap.executor")) for row in labels)  # runners: /runner
        assert labels["cap.workspace"] == "workspace · System 1"
        groups = [c.group for c in app.screen.choices]  # pyright: ignore[reportAttributeAccessIssue]
        assert groups == sorted(groups, key=lambda g: g != "on")  # what is on first, so each group heads once
        notes = {c.id: c.note for c in app.screen.choices}  # pyright: ignore[reportAttributeAccessIssue]
        assert (
            notes["cap.base"].startswith("read, write, edit, bash, grep, find")
            and "decision_code" in notes["cap.code-search"]
            and "decision_files" in notes["cap.workspace"]
            and notes["cap.memory"].endswith(f"· {tmp_path / 'drive'}")
        )
        assert "used as is" not in plain(app.screen.query_one("#picker-hint", Static))  # no typed names here
        await pilot.press("escape")
        await pilot.pause()
        # A typed switch flips the capability and reopens the session on the same journal, keeping the screen.
        await send(app, pilot, "/capabilities browser")  # the long name is an alias
        await wait_for(lambda: app.session is not None and app.session is not first)
        await pilot.pause()
        picks = opened[-1]
        assert picks.capabilities is not None and picks.capabilities.browser.enabled
        assert picks.capabilities.tool_packs == ["base", "web-read"] and picks.session_file == journal
        text = transcript_text(app)
        assert "browser on; reopening the session…" in text
        assert "Session reopened · browser on · tools: read, write, edit, bash, grep, find" in text
        # A switch the session cannot open on: the reason is shown, and the session comes back as it was.
        await send(app, pilot, "/caps code-search")
        await wait_for(lambda: len(opened) == 4 and app.session is not None)
        await pilot.pause()
        assert opened[2].capabilities is not None and opened[2].capabilities.code_search.enabled  # the try
        restored = opened[3].capabilities
        assert restored is not None and not restored.code_search.enabled and restored.browser.enabled
        text = transcript_text(app)
        assert "Cannot open a session: Code search needs the code-search extra" in text
        assert "Back on the capabilities as they were." in text
        await send(app, pilot, "/caps web-read")
        await wait_for(lambda: len(opened) == 5)
        await pilot.pause()
        assert opened[-1].capabilities is not None and opened[-1].capabilities.tool_packs == ["base"]
        assert opened[-1].capabilities.browser.enabled  # the earlier switch stays
        await send(app, pilot, "/caps memory")
        await wait_for(lambda: len(opened) == 6)
        memory = opened[-1].capabilities.memory  # pyright: ignore[reportOptionalMemberAccess]
        assert not memory.enabled and "memory off" in transcript_text(app)
        await send(app, pilot, f"/capabilities memory {tmp_path / 'notes'}")
        await wait_for(lambda: len(opened) == 7)
        memory = opened[-1].capabilities.memory  # pyright: ignore[reportOptionalMemberAccess]
        assert memory.enabled and memory.drive == tmp_path / "notes"
        await send(app, pilot, "/caps nothing")
        await pilot.pause()
        assert "No capability called 'nothing'" in transcript_text(app) and len(opened) == 7
        await app.action_quit()


async def test_a_compactions_card_streams_system_2s_summary_and_a_click_shows_it_not_the_state_panel(tmp_path: Path):
    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")]))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        app.status.running = True
        for event in [run_start("r1"), StateWriting(turn=4, turns_covered=2)]:
            app.post_message(HarnessEvent(event))
        await wait_for(lambda: len(app.query(CompactionCard)) == 1)
        card = app.query_one(CompactionCard)
        # The write streams in after its card is up, as a model writes it: the card follows it on its own.
        for event in [StateDelta(turn=4, text="## Plan\n- Fix"), StateDelta(turn=4, text=" the parser\n")]:
            app.post_message(HarnessEvent(event))
        await wait_for(lambda: "Fix the parser" in plain(card.head))
        assert "System 2 is writing its summary" in plain(card.head)
        assert "writing the summary" in plain(app.query_one(StatusBar))
        landed = State(turn=4, landed=4, turns_covered=2, tokens_before=900, text=SAMPLE_STATE, duration_ms=1800)
        cut = Compaction(
            turn=4, method="state", tokens_before=900, tokens_after=400, turns_compacted=2, summary=SAMPLE_STATE
        )
        app.post_message(HarnessEvent(landed))
        app.post_message(HarnessEvent(cut))
        await wait_for(lambda: "Continues from System 2's summary" in plain(card.head))
        assert "writing" not in plain(app.query_one(StatusBar))
        # The summary is System 2's, not the State System 1 reads: a click on its card shows it, the panel never.
        assert "Fix the parser" not in transcript_text(app) and "Fix the parser" not in panel_text(app)
        await click_compaction(app, pilot)
        assert "Fix the parser (in progress)" in transcript_text(app) and "Fix the parser" not in panel_text(app)
        # A compaction stopped while System 2 writes (Esc, or an extension's veto) says so instead of spinning on.
        app.post_message(HarnessEvent(StateWriting(turn=6, turns_covered=1)))
        await wait_for(lambda: len(app.query(CompactionCard)) == 2)
        app.transcript.compactions_ended()
        await pilot.pause()
        assert "Compaction stopped · turn 6 · nothing was cut" in plain(app.query(CompactionCard).last().head)
        await app.action_quit()


async def test_an_index_build_shows_as_a_live_meter_then_settles(tmp_path: Path):
    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")]))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        app.status.running = True
        for event in (
            run_start("r1"),
            Indexing(name="code-search", root=str(tmp_path / "work"), done=1234, total=5712),
        ):
            app.post_message(HarnessEvent(event))
        await pilot.pause()
        await pilot.pause()
        text = transcript_text(app)
        assert "⌁ Indexing the working directory for decision_code" in text and "1,234 of 5,712 files" in text
        assert "decision_code indexing 1,234 of 5,712 files" in plain(app.query_one(StatusBar))
        done = Indexing(
            name="code-search", root=str(tmp_path / "work"), done=5712, total=5712, finished=True, duration_ms=6100
        )
        app.post_message(HarnessEvent(done))
        await pilot.pause()
        await pilot.pause()
        text = transcript_text(app)
        assert (
            "⌁ Indexed 5,712 files for decision_code in 6.1s" in text and "⌁ Indexing" not in text
        )  # one line, replaced
        assert "indexing" not in plain(app.query_one(StatusBar))
        await app.action_quit()


def state_screen(app: HarnessApp) -> StateScreen:
    screen = app.screen
    assert isinstance(screen, StateScreen)
    return screen


def panel_text(app: HarnessApp) -> str:
    return "\n".join(plain(w) for w in app.query_one(StatePanel).query(Static))


async def test_the_state_panel_draws_what_system_1_reads_and_state_opens_it_in_full(tmp_path: Path):
    script = [Turn(text="Let me look.", tool_calls=[("bash", {"command": "echo hi"})]), Turn(text="It printed hi.")]
    app = HarnessApp(make_opener(tmp_path, script))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        panel = app.query_one(StatePanel)
        await send(app, pilot, "say hi")
        await wait_for(lambda: panel.cut is not None and "It printed hi." in panel.cut.text)
        await pilot.pause()
        session, cut = app.session, panel.cut
        assert session is not None and cut is not None
        budget = int(session.deps.decisions.max_state_tokens * STATE_SHARE)
        assert cut.text == session_cut(session.deps.journal.active_path(), budget)  # what System 1 is given
        text = panel_text(app)
        assert "› say hi" in text and "✓ $ echo hi" in text and "It printed hi." in text and "3 whole" in text
        await send(app, pilot, "/state")
        assert json.loads(state_screen(app).query_one("#state-cut", TextArea).text) == json.loads(cut.text)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, StateScreen)
        await app.action_quit()


async def test_the_pickers_list_the_catalog_and_a_pick_switches_the_model(tmp_path: Path):
    listed = Catalog(
        system2=[ModelInfo(id="fake/big", name="Big"), ModelInfo(id="other/x", name="X")],
        system1=[
            ModelInfo(id="drex-latest", name="drex-latest"),
            ModelInfo(id="decisions:~typesafe/jev-latest", name="TypeSafe: Jev Latest"),
        ],
    )

    async def load() -> Catalog:
        return listed

    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")]), catalog=load)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None and app.catalog is not None)
        session = app.session
        assert session is not None
        await send(app, pilot, "/system2")
        picker = app.screen
        assert isinstance(picker, ModelPicker)
        options = picker.query_one(OptionList)
        ids = [options.get_option_at_index(i).id for i in range(options.option_count)]
        # The current model is listed first even when the catalog does not have it; group headers have no id.
        assert [i for i in ids if i] == ["fake/model", "fake/big", "other/x"]
        await pilot.press("b", "i", "g", "enter")
        await wait_for(lambda: session.deps.system2.ref == "fake/big")
        assert "System 2 → fake/big" in transcript_text(app)
        # System 1: console and decisions-API models in their groups; a pick rebuilds System 1.
        await send(app, pilot, "/system1")
        picker = app.screen
        assert isinstance(picker, ModelPicker)
        options = picker.query_one(OptionList)
        prompts = [plain_prompt(options.get_option_at_index(i).prompt) for i in range(options.option_count)]
        assert any(p.startswith("CONSOLE") for p in prompts) and any("DECISIONS API" in p for p in prompts)
        # Rows read as the id inside its group and, dim, the display name; the current one is marked.
        assert any(p.startswith("  ~typesafe/jev-latest") and "TypeSafe: Jev Latest" in p for p in prompts), prompts
        assert any(p.startswith("● drex-latest") and "current" in p for p in prompts), prompts
        await pilot.press("j", "e", "v", "enter")
        await wait_for(lambda: session.system1 == "decisions:~typesafe/jev-latest")
        assert isinstance(session.deps.decisions.backend, Named)
        # A direct set that cannot be resolved is refused and said; the model stays.
        await send(app, pilot, "/system2 nowhere/x")
        await wait_for(lambda: "Cannot set system2 to nowhere/x" in transcript_text(app))
        assert session.deps.system2.ref == "fake/big"
        # A typed name the picker does not list is accepted as typed.
        await send(app, pilot, "/system1")
        await pilot.press("d", "r", "e", "x", "-", "m", "a", "x", "-", "v", "1", ".", "1", "enter")
        await wait_for(lambda: session.system1 == "drex-max-v1.1")
        await app.action_quit()


def plain_prompt(prompt: object) -> str:
    return prompt.plain if isinstance(prompt, Content) else str(prompt)


def test_the_picker_ranks_prefix_then_substring_then_subsequence_and_keeps_catalog_order_on_ties():
    choices = [
        Choice("openrouter/qwen/qwen3.8-27b", "Qwen3.8 27B", "openrouter"),
        Choice("fireworks/qwen3-30b", "Qwen3 30B", "fireworks"),
        Choice("anthropic/claude-haiku-5-5", "Claude Haiku 5.5", "anthropic"),
    ]
    assert [c.id for c in ranked(choices, "")] == [c.id for c in choices]
    assert [c.id for c in ranked(choices, "fire")] == ["fireworks/qwen3-30b"]
    assert [c.id for c in ranked(choices, "qwen")] == ["openrouter/qwen/qwen3.8-27b", "fireworks/qwen3-30b"]
    assert [c.id for c in ranked(choices, "chk")] == ["anthropic/claude-haiku-5-5"]  # a subsequence
    assert ranked(choices, "zzz") == []


async def test_a_prompt_while_a_run_is_in_progress_steers_it(tmp_path: Path):
    """Enter during a run sends the text into the run, before its next model call."""
    started = asyncio.Event()

    def script(request) -> Turn:
        started.set()
        if any(getattr(m, "content", "") == "stop after this" for m in request.messages):
            return Turn(text="Stopping as asked.")
        return Turn(tool_calls=[("bash", {"command": "sleep 0.3"})])

    app = HarnessApp(make_opener(tmp_path, script))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        session = app.session
        assert session is not None
        await send(app, pilot, "keep going")
        await started.wait()
        assert session.running and app.query_one(StatusBar).running
        await send(app, pilot, "stop after this")
        assert "stop after this  queued" in transcript_text(app)
        await wait_for(lambda: app._run_task is not None and app._run_task.done())  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        assert "Stopping as asked." in "".join(m.source for m in app.query_one(Transcript).query(".reply"))  # pyright: ignore[reportAttributeAccessIssue]
        assert "completed" in transcript_text(app) and not app.query_one(StatusBar).running
        assert "› stop after this" in transcript_text(app) and "queued" not in transcript_text(app)  # it was taken
        await app.action_quit()


async def test_escape_aborts_the_run_in_progress(tmp_path: Path):
    started = asyncio.Event()

    def script(request) -> Turn:
        started.set()
        return Turn(tool_calls=[("bash", {"command": "sleep 5"})])

    app = HarnessApp(make_opener(tmp_path, script))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await send(app, pilot, "run forever")
        await started.wait()
        await pilot.press("escape")
        await wait_for(lambda: app._run_task is not None and app._run_task.done())  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        assert "Aborting the run" in transcript_text(app) and "aborted" in transcript_text(app)
        assert not app.query_one(StatusBar).running
        await app.action_quit()


async def test_compact_cuts_the_older_turns_to_the_state_now_and_says_when_there_is_nothing_to_cut(tmp_path: Path):
    replies = [
        Turn(tool_calls=[("bash", {"command": f"printf 'OUT1 {'x' * 4_000}'"})]),
        Turn(tool_calls=[("bash", {"command": f"printf 'OUT2 {'x' * 4_000}'"})]),
        Turn(text="printed both"),
    ]

    def script(request) -> Turn:
        if is_state_write(request):
            return Turn(text="## Plan\n- [x] print both outputs\n## Now\nreport")
        return replies.pop(0) if replies else Turn(text="done")

    # Far under compact_at, so no run compacts on its own; a small recent share, so /compact has turns to cut.
    profile = Profile(thresholds=Thresholds(keep_recent=0.05))
    journal = tmp_path / "session.jsonl"
    app = HarnessApp(make_opener(tmp_path, script, profile=profile, journal=journal))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await send(app, pilot, "/compact")
        await wait_for(lambda: "Nothing to compact" in transcript_text(app))
        await send(app, pilot, "print two outputs")
        await wait_for(lambda: app._run_task is not None and app._run_task.done())  # pyright: ignore[reportPrivateUsage]
        assert "⟲ Compaction" not in transcript_text(app)
        before = app.status.context
        await send(app, pilot, "/compact")
        await wait_for(lambda: "⟲ Compaction" in transcript_text(app))
        await pilot.pause()
        text = transcript_text(app)
        assert "Continues from System 2's summary" in text  # written first, then cut to
        assert app.session is not None and not app.session.running and not app.status.running
        assert 0 < app.status.context < before  # the masthead's context is the compacted one
        await click_compaction(app, pilot)
        assert "print both outputs" in transcript_text(app)  # what the agent continues from, a click away
        await app.action_quit()
    # Resumed, the card still has it: the journal's compaction entry keeps the summary its event leaves out.
    second = HarnessApp(make_opener(tmp_path, [Turn(text="ok")], profile=profile, journal=journal))
    async with second.run_test(size=SIZE) as pilot:
        await wait_for(lambda: second.session is not None and len(second.query(CompactionCard)) == 1)
        await pilot.pause()
        await click_compaction(second, pilot)
        assert "print both outputs" in transcript_text(second)
        await second.action_quit()


async def test_a_resumed_journal_is_replayed_with_its_state(tmp_path: Path):
    journal = tmp_path / "session.jsonl"
    first = HarnessApp(make_opener(tmp_path, [Turn(text="Four.")], journal=journal))
    async with first.run_test(size=SIZE) as pilot:
        await wait_for(lambda: first.session is not None)
        await send(first, pilot, "what is 2+2")
        await wait_for(lambda: first._run_task is not None and first._run_task.done())  # pyright: ignore[reportPrivateUsage]
        await first.action_quit()
    second = HarnessApp(make_opener(tmp_path, [Turn(text="Six.")], journal=journal))
    async with second.run_test(size=SIZE) as pilot:
        await wait_for(lambda: second.session is not None)
        await pilot.pause()
        text = transcript_text(second)
        assert "› what is 2+2" in text
        assert "Four." in "".join(m.source for m in second.query_one(Transcript).query(".reply"))  # pyright: ignore[reportAttributeAccessIssue]
        await second.action_quit()


async def test_a_resumed_journal_brings_back_system_1s_panel_cards_and_each_systems_cost_where_they_were(
    tmp_path: Path,
):
    journal = tmp_path / "session.jsonl"
    script = [
        Turn(text="Let me look.", tool_calls=[("bash", {"command": "echo hi"})]),
        Turn(tool_calls=[("bash", {"command": "echo again"})]),
        Turn(tool_calls=[("bash", {"command": "echo third"})]),
        Turn(text="It printed hi."),
    ]
    decider = FakeDecision()
    first = HarnessApp(make_opener(tmp_path, script, system1=decider, journal=journal))
    async with first.run_test(size=SIZE) as pilot:
        await wait_for(lambda: first.session is not None)
        await send(first, pilot, "say hi")
        await wait_for(lambda: first._run_task is not None and first._run_task.done())  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        live = plain(first.query_one("#ledger-chart", Static))
        status = first.query_one(StatusBar)
        costs = (status.cost2, status.cost1, status.unpriced)
        masthead = plain(status)
        await first.action_quit()
    assert " turns · 1 prompt" in live and 3 in first.ledger.current.turns, live
    # Each System 1 ask counted once, as it was asked and not again at the run's end; the fake System 2 reports
    # tokens but no price, so its cost is not known rather than $0.
    assert decider.asks >= 1 and costs == (0.0, pytest.approx(decider.asks * 0.0001), True)
    assert "price unknown" in masthead and f"${decider.asks * 0.0001:.4f}" in masthead
    second = HarnessApp(make_opener(tmp_path, [Turn(text="ok")], system1=FakeDecision(), journal=journal))
    async with second.run_test(size=SIZE) as pilot:
        await wait_for(lambda: second.session is not None)
        await pilot.pause()
        assert plain(second.query_one("#ledger-chart", Static)) == live  # the chart as it was
        status = second.query_one(StatusBar)
        assert (status.cost2, status.cost1, status.unpriced) == costs  # the costs as they were
        text = transcript_text(second)
        # The supervisor's card at turn 3's boundary: after the third call, before the reply.
        card = text.rindex("System 1 · ")
        assert text.index("$ echo third") < card and second.run_count == 1
        await second.action_quit()


async def test_a_long_resumed_journal_shows_its_last_turns_and_counts_all_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    journal = tmp_path / "session.jsonl"
    script = [
        Turn(text="Let me look.", tool_calls=[("bash", {"command": "echo hi"})]),
        Turn(tool_calls=[("bash", {"command": "echo again"})]),
        Turn(tool_calls=[("bash", {"command": "echo third"})]),
        Turn(text="It printed hi."),
    ]
    decider = FakeDecision()
    first = HarnessApp(make_opener(tmp_path, script, system1=decider, journal=journal))
    async with first.run_test(size=SIZE) as pilot:
        await wait_for(lambda: first.session is not None)
        await send(first, pilot, "say hi")
        await wait_for(lambda: first._run_task is not None and first._run_task.done())  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        status = first.query_one(StatusBar)
        costs = (status.cost2, status.cost1)
        await first.action_quit()

    monkeypatch.setattr(app_module, "RESUME_TURNS", 2)
    second = HarnessApp(make_opener(tmp_path, [Turn(text="ok")], system1=FakeDecision(), journal=journal))
    async with second.run_test(size=SIZE) as pilot:
        await wait_for(lambda: second.session is not None)
        await pilot.pause()
        text = transcript_text(second)
        # Only the last two turns are drawn: the third call and the reply, not the prompt or the first calls.
        assert "$ echo third" in text and "$ echo hi" not in text and "› say hi" not in text
        assert "It printed hi." in "".join(m.source for m in second.query_one(Transcript).query(".reply"))  # pyright: ignore[reportAttributeAccessIssue]
        # What is counted comes from all of it: both systems' costs, System 1's judgement.
        status = second.query_one(StatusBar)
        assert (status.cost2, status.cost1) == costs
        assert 3 in second.ledger.current.turns
        # System 1's State is rebuilt from the journal, all of it: the calls the transcript leaves out too.
        await wait_for(lambda: second.query_one(StatePanel).cut is not None)
        await pilot.pause()
        assert "› say hi" in panel_text(second) and "✓ $ echo hi" in panel_text(second)
        await second.action_quit()


async def test_reload_quits_with_what_to_reopen_on_and_a_new_app_resumes_the_conversation_on_it(tmp_path: Path):
    journal = tmp_path / "session.jsonl"
    started = asyncio.Event()

    def script(request) -> Turn:
        if not started.is_set():
            started.set()
            return Turn(tool_calls=[("bash", {"command": "sleep 5"})])
        return Turn(text="Four.")

    first = HarnessApp(make_opener(tmp_path, script, journal=journal))
    async with first.run_test(size=SIZE) as pilot:
        await wait_for(lambda: first.session is not None)
        await send(first, pilot, "what is 2+2")
        await started.wait()
        await send(first, pilot, "/reload")  # not during a run
        assert first.reload is None and "before reloading" in transcript_text(first)
        await pilot.press("escape")
        await wait_for(lambda: first._run_task is not None and first._run_task.done())  # pyright: ignore[reportPrivateUsage]
        first.pending.thinking = "high"
        await send(first, pilot, "/reload")
    assert first.reload is not None and not first.is_running
    assert (first.reload.session_file, first.reload.system2, first.reload.thinking) == (journal, "fake/model", "high")
    assert first.journal_path == journal
    # What `main` hands the restarted process: the picks, through the environment, as JSON.
    picks = PICKS.validate_json(PICKS.dump_json(first.reload))
    second = HarnessApp(make_opener(tmp_path, [Turn(text="Six.")], journal=journal), picks=picks)
    async with second.run_test(size=SIZE) as pilot:
        await wait_for(lambda: second.session is not None)
        await pilot.pause()
        assert "› what is 2+2" in transcript_text(second)
        await second.action_quit()


async def test_a_closed_terminal_window_quits_as_quit_does_and_closes_the_session(tmp_path: Path):
    """Closing the terminal window hangs the TUI up: it closes the session, which starts its steward, and exits,
    rather than dying with the session left open."""
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)  # without the app's handler the test fails, not pytest
    try:
        app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")]))
        async with app.run_test(size=SIZE):
            await wait_for(lambda: app.session is not None)
            session = app.session
            assert session is not None
            os.kill(os.getpid(), signal.SIGHUP)
            await wait_for(lambda: session.closed and not app.is_running)
    finally:
        signal.signal(signal.SIGHUP, previous)


async def test_ctrl_c_copies_what_is_selected_in_the_prompt_else_clears_it(tmp_path: Path):
    copied: list[str] = []
    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")]), copy=copied.append)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        prompt = app.query_one("#prompt", Prompt)
        await pilot.press(*"hello world")
        prompt.select_all()
        await pilot.press("ctrl+c")
        assert copied == ["hello world"] and prompt.text == "hello world"
        await pilot.press("end", "ctrl+c")
        assert prompt.text == "" and copied == ["hello world"]
        await pilot.press("ctrl+c")  # nothing to clear: the app's own Ctrl+C, which says how to quit
        assert app.is_running and app.session is not None
        await app.action_quit()


async def test_a_dropped_screenshot_is_kept_where_the_agent_can_still_read_it(tmp_path: Path, monkeypatch):
    kept = tmp_path / "kept"
    monkeypatch.setattr(widgets, "KEPT_IMAGES", kept)
    folder = tmp_path / "TemporaryItems" / "NSIRD_screencaptureui_A3fxCa"  # pytest's tmp_path is in the temp dir
    folder.mkdir(parents=True)
    shot = folder / "Screenshot 2026-11-09 at 3.35.05 PM.png"
    shot.write_bytes(b"\x89PNG fake")
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")]))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        prompt = app.query_one("#prompt", Prompt)
        await pilot.press(*"what is this? ")
        app.post_message(Paste(str(shot).replace(" ", "\\ ") + " "))  # as the terminal pastes a dropped file
        await pilot.pause()
        [copy] = kept.iterdir()
        assert prompt.text == f"what is this? {copy}" and copy.name.endswith("-Screenshot-2026-11-09-at-3.35.05-PM.png")
        shot.unlink()  # macOS empties the folder; the copy stays
        assert copy.read_bytes() == b"\x89PNG fake"
        # A file that is not an image in a temporary folder, or text that is not a path, lands as pasted.
        for pasted in (str(notes), "just some text"):
            prompt.clear()
            app.post_message(Paste(pasted))
            await pilot.pause()
            assert prompt.text == pasted
        # Dropped first, the path starts the prompt: it is sent, not taken for a command.
        prompt.clear()
        app.post_message(Paste(str(notes)))
        await pilot.press(*" what is this", "enter")
        await wait_for(lambda: app.run_count == 1)
        text = transcript_text(app)
        assert f"› {notes} what is this" in text and "Unknown command" not in text
        await app.action_quit()


async def test_an_image_the_prompt_names_shows_as_attached_with_whether_system_2_sees_it(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(widgets.tempfile, "gettempdir", lambda: str(tmp_path / "T"))  # the Desktop is not temporary
    logo = tmp_path / "Desktop" / "logo.jpeg"  # an image dropped from anywhere, not only a screenshot
    logo.parent.mkdir()
    logo.write_bytes(b"\xff\xd8 fake")
    for sees in (True, False):
        app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")], profile=Profile(vision=sees)))
        async with app.run_test(size=SIZE) as pilot:
            await wait_for(lambda app=app: app.session is not None)
            prompt = app.query_one("#prompt", Prompt)
            app.post_message(Paste(f"file://{logo}"))  # as some terminals drop it: the path is what lands
            await pilot.pause()
            assert prompt.text == str(logo)
            await pilot.press(*" what is this?")
            await pilot.pause()
            border = prompt.border_subtitle or ""
            assert "image" in border and "logo.jpeg" in border
            assert ("cannot see images" in border) is not sees
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert not prompt.border_subtitle
            await app.action_quit()


def test_a_reload_keeps_the_alternate_screen_up_from_one_process_to_the_next(tmp_path: Path, monkeypatch):
    written: list[str] = []

    class Recording:
        def __init__(self, app: object) -> None:
            pass

        def write(self, data: str) -> None:
            written.append(data)

    monkeypatch.setattr(App, "get_driver_class", lambda self: Recording)
    sequence = (ALT_SCREEN_ON, "frame", ALT_SCREEN_OFF, SHOW_CURSOR)
    leaving = HarnessApp(make_opener(tmp_path, []))
    leaving.reload = Models()
    driver = leaving.get_driver_class()(leaving)
    for data in sequence:
        driver.write(data)
    assert written == [ALT_SCREEN_ON, "frame"]  # the last frame stays up, the cursor hidden
    written.clear()
    starting = HarnessApp(make_opener(tmp_path, []), reloaded=True)
    driver = starting.get_driver_class()(starting)
    for data in sequence:
        driver.write(data)
    assert written == ["frame", ALT_SCREEN_OFF, SHOW_CURSOR]  # painted over the kept frame; left on quit


async def test_a_drag_copies_the_selection_and_a_click_keeps_the_prompt(tmp_path: Path):
    copied: list[str] = []
    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")]), copy=copied.append)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await send(app, pilot, "hello there")
        await wait_for(lambda: app.session is not None and app.run_count == 1 and not app.session.running)
        await pilot.pause()
        prompt = app.query_one("#prompt", Prompt)
        line = app.query_one(Transcript).query(".user").first(Static)
        # A click in the transcript does not take the keyboard from the prompt: typing and pasting still land.
        await pilot.click(line)
        assert app.focused is prompt
        # A drag selects, and the selection is copied as the mouse is released: no ctrl+c needed.
        await pilot.mouse_down(line, offset=(4, 0))  # past the bar, the padding and `› `
        await pilot.hover(line, offset=(9, 0))
        await pilot.mouse_up(line, offset=(9, 0))
        await pilot.pause()
        assert copied and copied[-1].strip() and copied[-1].strip() in "hello there"
        assert f"copied {len(copied[-1])} characters" in plain(app.query_one(StatusBar))
        assert app.focused is prompt
        await app.action_quit()


async def test_the_prompt_takes_several_lines_a_click_places_the_cursor_and_enter_sends_them_all(
    tmp_path: Path, monkeypatch
):
    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")]))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        prompt = app.query_one("#prompt", Prompt)
        app.post_message(Paste("first line\nsecond line"))  # as the terminal sends it: to the app, then the prompt
        await pilot.pause()
        assert prompt.text == "first line\nsecond line"  # a paste keeps its lines
        await pilot.press("ctrl+j", *"third", "shift+enter", *"fourth")
        assert prompt.text == "first line\nsecond line\nthird\nfourth" and prompt.outer_size.height == 6  # it grows
        # A terminal or multiplexer that does not report Shift sends a plain Enter: Shift down on the keyboard
        # makes it a new line all the same.
        monkeypatch.setattr(widgets, "shift_held", lambda: True)
        await pilot.press("enter")
        monkeypatch.setattr(widgets, "shift_held", lambda: False)
        assert prompt.text == "first line\nsecond line\nthird\nfourth\n" and app.run_count == 0
        await pilot.press("backspace")
        # The text starts inside the border and the padding: a click on the second row, seventh column.
        await pilot.click(prompt, offset=(2 + 6, 1 + 1))
        assert prompt.cursor_location == (1, 6)
        await pilot.press(*"X", "enter")
        await wait_for(lambda: app.run_count == 1 and not app.session.running)  # pyright: ignore[reportOptionalMemberAccess]
        await pilot.pause()
        sent = plain(app.query_one(Transcript).query(".user").first(Static))
        assert sent == "› first line\n  secondX line\n  third\n  fourth" and prompt.text == ""
        # A command being typed is completed in ghost text that → accepts.
        await pilot.press(*"/sys")
        assert prompt.suggestion == "tem2"
        await pilot.press("right")
        assert prompt.text == "/system2"
        await app.action_quit()


def test_a_binding_takes_an_effort_on_its_profile_s_word_or_its_provider_s():
    assert not binding(FakeProvider([])).takes_effort and binding(FakeProvider([])).effort_levels == ()
    assert binding(Thinking([])).takes_effort and binding(Thinking([])).effort_levels == EFFORT_LEVELS
    named = binding(FakeProvider([]), profile=Profile(reasoning_effort="low"))
    assert named.takes_effort and named.effort_levels == ("low", "medium", "high")  # the run every API takes


async def test_the_thinking_level_shows_cycles_on_shift_tab_is_set_in_the_picker_and_stays(tmp_path: Path):
    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")], takes_effort=True))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await pilot.pause()
        first = app.session
        assert first is not None
        status = app.query_one(StatusBar)
        assert "thinking default" in plain(status)  # the profile names none: the model's own
        await pilot.press("shift+tab")
        await pilot.pause()
        assert first.deps.system2.profile.reasoning_effort == "low" and "thinking low" in plain(status)
        assert "Thinking → low" in transcript_text(app)
        await send(app, pilot, "/thinking high")
        assert first.deps.system2.profile.reasoning_effort == "high" and "thinking high" in plain(status)
        await send(app, pilot, "/thinking")
        assert "Thinking is high; /thinking low|medium|high|xhigh|max sets it" in transcript_text(app)
        await send(app, pilot, "/thinking ultra")
        assert "/thinking takes low, medium, high, xhigh, max (now: high)" in transcript_text(app)
        await send(app, pilot, "/thinking max")
        assert first.deps.system2.profile.reasoning_effort == "max" and "thinking max" in plain(status)
        await pilot.press("shift+tab")  # round: past max comes low
        await pilot.pause()
        assert first.deps.system2.profile.reasoning_effort == "low"
        await send(app, pilot, "/thinking high")
        # ←/→ in the System 2 picker: the level goes with the pick; the same model keeps its session.
        await send(app, pilot, "/system2")
        assert isinstance(app.screen, ModelPicker)
        line = app.screen.query_one("#picker-thinking", Static)
        assert "◀ high ▶" in plain(line)
        await pilot.press("left")
        assert "◀ medium ▶" in plain(line)
        await pilot.press("enter")  # the current model is highlighted
        await wait_for(lambda: first.deps.system2.profile.reasoning_effort == "medium")
        await pilot.pause()
        assert app.session is first and "thinking medium" in plain(status)
        # The level stays across /clear and a typed model switch.
        await send(app, pilot, "/clear")
        await wait_for(lambda: app.session is not None and app.session is not first)
        await pilot.pause()
        assert app.session is not None and app.session.deps.system2.profile.reasoning_effort == "medium"
        await send(app, pilot, "/system2 fake/other")
        await wait_for(lambda: app.session is not None and app.session.deps.system2.ref == "fake/other")
        await pilot.pause()
        assert app.session.deps.system2.profile.reasoning_effort == "medium"
        assert "System 2 → fake/other (working window 32.8k tokens) · thinking medium" in transcript_text(app)
        await app.action_quit()


async def test_a_profile_named_effort_offers_the_common_three_and_a_kept_level_waits_for_a_model_that_takes_it(
    tmp_path: Path,
):
    """A provider that declares no levels offers low, medium, high on the profile's word. A level chosen on a
    model with more is not forced on one with fewer: that one keeps its own, the switch notice says so, and the
    choice is back on the next model that takes it."""
    opener = make_opener(
        tmp_path,
        [Turn(text="ok")],
        takes_effort=True,
        profile=Profile(reasoning_effort="medium"),
        providers={"fake/plain": FakeProvider([Turn(text="ok")])},
    )
    app = HarnessApp(opener)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await pilot.pause()
        assert app.session is not None and app.session.deps.system2.effort_levels == EFFORT_LEVELS
        await send(app, pilot, "/thinking max")
        assert "thinking max" in plain(app.query_one(StatusBar))
        await send(app, pilot, "/system2 fake/plain")
        await wait_for(lambda: app.session is not None and app.session.deps.system2.ref == "fake/plain")
        await pilot.pause()
        assert app.session is not None
        bound = app.session.deps.system2
        assert bound.effort_levels == ("low", "medium", "high") and bound.profile.reasoning_effort == "medium"
        assert "· thinking medium (max is not a level it takes)" in transcript_text(app)
        assert "thinking medium" in plain(app.query_one(StatusBar))
        await send(app, pilot, "/thinking xhigh")
        assert "/thinking takes low, medium, high (now: medium)" in transcript_text(app)
        await send(app, pilot, "/system2 fake/model")
        await wait_for(lambda: app.session is not None and app.session.deps.system2.ref == "fake/model")
        await pilot.pause()
        assert app.session is not None and app.session.deps.system2.profile.reasoning_effort == "max"
        await app.action_quit()


async def test_a_model_that_takes_no_effort_says_so(tmp_path: Path):
    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")]))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await pilot.pause()
        assert "thinking" not in plain(app.query_one(StatusBar))
        await send(app, pilot, "/thinking low")
        assert "fake/model takes no thinking level" in transcript_text(app)
        await pilot.press("shift+tab")
        await pilot.pause()
        assert transcript_text(app).count("takes no thinking level") == 2
        await app.action_quit()


async def test_clear_starts_over_on_a_fresh_journal_with_the_models_kept(tmp_path: Path):
    journal, fresh = tmp_path / "journal.jsonl", tmp_path / "fresh.jsonl"
    opened: list[Models] = []
    openers = {
        False: make_opener(tmp_path, [Turn(text="ok")], system1=FakeDecision(), journal=journal),
        True: make_opener(tmp_path, [Turn(text="again")], system1=FakeDecision(), journal=fresh),
    }

    async def recording(emit: EventSink, picks: Models) -> Session:
        opened.append(Models(**vars(picks)))
        return await openers[picks.fresh](emit, picks)

    app = HarnessApp(recording)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        first = app.session
        assert first is not None
        await send(app, pilot, "/system2 fake/other")  # a pick made in place, which a fresh journal cannot replay
        await wait_for(lambda: first.deps.system2.ref == "fake/other")
        await send(app, pilot, "hello")
        await wait_for(lambda: app.run_count == 1 and not first.running)
        await pilot.pause()
        assert "› hello" in transcript_text(app)
        await send(app, pilot, "/clear")
        await wait_for(lambda: len(opened) == 2 and app.session is not None and app.session is not first)
        await pilot.pause()
        picks = opened[-1]
        assert picks.fresh and picks.session_file is None and picks.system2 == "fake/other"
        assert app.session is not None and app.session.deps.system2.ref == "fake/other"
        text = transcript_text(app)
        assert "› hello" not in text
        assert "Session cleared · the conversation so far is kept in " in text and journal.name in text
        assert "pick the models" not in text  # what to do is the masthead's, never the transcript's
        assert app.run_count == 0 and app.query_one(StatusBar).context == 0
        assert app.query_one(StatePanel).cut is None
        assert "› hello" not in panel_text(app)
        assert not plain(app.query_one("#ledger-chart", Static))
        await send(app, pilot, "hello again")
        await wait_for(lambda: app.run_count == 1 and app.session is not None and not app.session.running)
        await pilot.pause()
        assert "again" in transcript_text(app)
        assert "hello again" in fresh.read_text() and "hello again" not in journal.read_text()
        await app.action_quit()


def test_the_bare_command_opens_the_tui_on_a_terminal_and_p_or_a_pipe_keeps_it_headless(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[tuple[Options, str | None]] = []
    monkeypatch.setattr("intern.tui.app.main", lambda options, env=None, prompt=None: calls.append((options, prompt)))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert cli_main(["--system2", "fake/m", "--cwd", str(tmp_path), "fix", "it"]) == 0
    assert cli_main(["--cwd", str(tmp_path)]) == 0
    assert [prompt for _, prompt in calls] == ["fix it", None] and calls[0][0].system2 == "fake/m"
    # -p, --json, or no terminal: the headless path, which here fails on the unknown provider before any model call.
    for argv in (["-p", "--system2", "fake/m", "task"], ["--json", "--system2", "fake/m", "task"]):
        assert cli_main([*argv, "--cwd", str(tmp_path)]) == 2
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert cli_main(["--system2", "fake/m", "--cwd", str(tmp_path), "task"]) == 2
    assert len(calls) == 2


WORKER_IN_APP = """
import asyncio, multiprocessing, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from textual.app import App
from intern.config import Options
from intern.tui import app

class Spawning(app.HarnessApp):
    def run(self):  # a plain Textual app stands in, so stderr is Textual's capture while a worker process starts
        async def inside():
            async with App().run_test():
                with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
                    return pool.submit(abs, -3).result()
        print("worker answered", asyncio.run(inside()))

if __name__ == "__main__":  # a spawned worker imports this file
    app.HarnessApp = Spawning
    app.main(Options(cwd=Path(sys.argv[1]), system2="fake/m", persist=False), env={"INTERN_HOME": sys.argv[1]})
"""


def test_a_worker_process_starts_while_the_tui_runs(tmp_path: Path):
    # Index builds and OCR run in spawned worker processes. The first spawn in a process starts multiprocessing's
    # resource tracker with sys.stderr's descriptor: -1 under Textual's capture, and the spawn failed ("bad value(s)
    # in fds_to_keep", index_files on a folder of PDFs). A fresh interpreter, so no tracker runs yet.
    (tmp_path / "worker_in_app.py").write_text(WORKER_IN_APP)  # a file: Textual reads an app class's source
    done = subprocess.run(
        [sys.executable, str(tmp_path / "worker_in_app.py"), str(tmp_path)], capture_output=True, text=True, timeout=120
    )
    assert "worker answered 3" in done.stdout, done.stderr[-2000:]


def test_the_cli_dispatches_tui_with_its_options_and_the_first_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[tuple[Options, str | None]] = []
    monkeypatch.setattr("intern.tui.app.main", lambda options, env=None, prompt=None: calls.append((options, prompt)))
    argv = ["tui", "--system2", "fake/m", "--no-session", "--cwd", str(tmp_path), "fix", "it"]
    assert cli_main(argv) == 0
    options, prompt = calls[0]
    assert options.system2 == "fake/m" and options.persist is False and prompt == "fix it"
    assert cli_main(["tui", "--cwd", str(tmp_path)]) == 0 and calls[1][1] is None


async def test_without_a_configured_system_2_the_app_starts_on_the_default_quietly(tmp_path: Path):
    async def default() -> str | None:
        return "fake/default"

    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")], configured_system2=None), default_system2=default)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await pilot.pause()
        assert app.session is not None and app.session.deps.system2.ref == "fake/default"
        assert "System 2 →" not in transcript_text(app)  # the masthead names it
        assert app.query_one(StatusBar).system2 == "fake/default" and not app.query_one(StatusBar).note
        await app.action_quit()


async def test_without_a_default_the_pickers_open_the_session_on_the_pick(tmp_path: Path):
    async def default() -> str | None:
        return None

    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")], configured_system2=None), default_system2=default)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: not app._opening)  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        assert app.session is None and "no model given" in transcript_text(app)
        assert app.query_one(StatusBar).note == "no session"
        # A System 1 pick before the session exists is kept for it; the System 2 pick opens it on both.
        await send(app, pilot, "/system1 drex-max-v1.1")
        await pilot.pause()
        assert app.session is None and app.pending.system1 == "drex-max-v1.1"
        await send(app, pilot, "/system2")
        picker = app.screen
        assert isinstance(picker, ModelPicker)
        await pilot.press("f", "a", "k", "e", "/", "b", "i", "g", "enter")  # nothing listed: the typed name
        await wait_for(lambda: app.session is not None)
        session = app.session
        assert session is not None
        assert (session.deps.system2.ref, session.system1) == ("fake/big", "drex-max-v1.1")
        assert app.query_one(StatusBar).system2 == "fake/big" and not app.query_one(StatusBar).note
        await send(app, pilot, "hello")
        await wait_for(lambda: app._run_task is not None and app._run_task.done())  # pyright: ignore[reportPrivateUsage]
        await app.action_quit()


def test_system_2_rows_put_known_models_first_and_read_without_the_provider_prefix():
    from intern.tui.app import system2_choice
    from intern.tui.pickers import fit

    known = system2_choice(ModelInfo(id="openrouter/qwen/qwen3.8-27b", name="x"))
    assert (known.group, known.label) == ("known models", "qwen/qwen3.8-27b")
    assert known.note == "openrouter · Qwen3.8 27B"
    listed = system2_choice(ModelInfo(id="deepinfra/Qwen/Qwen3-14B", name="Qwen/Qwen3-14B"))
    assert (listed.group, listed.label, listed.note) == ("deepinfra", "Qwen/Qwen3-14B", "")
    named = system2_choice(ModelInfo(id="openrouter/qwen/qwen3-coder", name="Qwen: Qwen3 Coder"))
    assert (named.group, named.label, named.note) == ("openrouter", "qwen/qwen3-coder", "Qwen: Qwen3 Coder")
    assert fit("abc", 6) == "abc   " and fit("abcdefghijkl", 8) == "abcd…jkl"
    assert ranked([known, listed, named], "qwen3.8") == [known]  # the known model's name matches


def test_token_counts_read_in_k_and_m():
    from intern.tui.widgets import tokens_text

    assert [tokens_text(n) for n in (512, 1_500, 131_072, 1_000_000)] == ["512", "1.5k", "131.1k", "1.0M"]


async def test_the_runner_is_one_choice_picked_with_slash_runner(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    opened: list[Models] = []
    inner = make_opener(tmp_path, [Turn(text="ok")], journal=journal)

    async def recording(emit: EventSink, picks: Models) -> Session:
        opened.append(Models(**vars(picks)))
        return await inner(emit, picks)

    app = HarnessApp(recording, capabilities=CapabilitySettings(), memory=tmp_path / "drive")
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        first = app.session
        await send(app, pilot, "/runner")
        assert isinstance(app.screen, ModelPicker)
        groups = {c.id: c.group for c in app.screen.choices}  # pyright: ignore[reportAttributeAccessIssue]
        assert groups["local"] == "in use" and groups["docker"] == "can run here"
        assert next(iter(groups)) == "local"  # the runner in use heads the list
        await pilot.press("escape")
        await pilot.pause()
        # A typed name picks it, and the session reopens on the same journal with commands running there.
        await send(app, pilot, "/runner docker")
        await wait_for(lambda: app.session is not None and app.session is not first)
        await pilot.pause()
        picks = opened[-1]
        assert picks.capabilities is not None and picks.capabilities.runner.name == "docker"
        assert picks.session_file == journal
        assert "runner docker" in transcript_text(app)
        # One that does not exist is refused with why, and nothing reopens.
        count = len(opened)
        await send(app, pilot, "/runner nowhere")
        await wait_for(lambda: "Unknown runner 'nowhere'" in transcript_text(app))
        assert len(opened) == count


async def test_secrets_are_added_listed_with_whether_they_load_and_removed(tmp_path: Path, monkeypatch):
    from textual.widgets import Input

    from intern.tui.pickers import SecretForm

    from .test_secrets_commands import FAKE_SECURITY

    monkeypatch.setattr(sys, "platform", "darwin")
    bin_dir, keychain, home = tmp_path / "bin", tmp_path / "keychain", tmp_path / "home"
    for folder in (bin_dir, keychain, home):
        folder.mkdir()
    (bin_dir / "security").write_text(FAKE_SECURITY)
    (bin_dir / "security").chmod(0o755)
    (home / "config.toml").write_text('[secrets]\nBROKEN = ["sh", "-c", "exit 3"]\n')
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_KEYCHAIN": str(keychain)}
    monkeypatch.setattr(app_module, "process_env", lambda: env)
    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")]), home=home)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await send(app, pilot, "/secrets set")
        assert isinstance(app.screen, SecretForm)
        app.screen.query_one("#secret-name", Input).value = "GITHUB_TOKEN"
        value = app.screen.query_one("#secret-value", Input)
        assert value.password
        value.value = "ghp-typed-secret-value"
        value.focus()
        await pilot.press("enter")
        await wait_for(lambda: "Stored GITHUB_TOKEN in the keychain" in transcript_text(app))
        assert 'GITHUB_TOKEN = "keychain"' in (home / "config.toml").read_text()

        await send(app, pilot, "/secrets")
        await wait_for(lambda: isinstance(app.screen, ModelPicker))
        rows = {c.id: (c.group, c.note) for c in app.screen.choices}  # pyright: ignore[reportAttributeAccessIssue]
        assert rows["GITHUB_TOKEN"] == ("loads", "keychain")
        assert rows["BROKEN"][0] == "does not load" and rows["BROKEN"][1].startswith("sh exited 3")
        assert list(rows) == ["BROKEN", "GITHUB_TOKEN", "+add"]  # problems first, adding last
        await pilot.press("escape")
        await pilot.pause()

        await send(app, pilot, "/secrets remove GITHUB_TOKEN")
        await wait_for(lambda: "Removed GITHUB_TOKEN" in transcript_text(app))
        assert "GITHUB_TOKEN" not in (home / "config.toml").read_text()
        assert "ghp-typed-secret-value" not in transcript_text(app)
        await app.action_quit()
