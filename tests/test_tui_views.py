"""Extension views: an extension's own drawing in the TUI, fed by the events it emits, hosted by the app."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel
from textual.binding import Binding, BindingType
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from intern.events import Event, ExtensionEvent, ToolCallStart, event_adapter, event_json_schema
from intern.hooks import Extension, RunContext
from intern.journal import EventEntry, Journal
from intern.messages import ToolCall, ToolResult
from intern.protocol import acp_mapping
from intern.providers.fake import Turn
from intern.tui.app import HarnessApp
from intern.tui.views import ExtensionView
from intern.tui.widgets import StatusBar, ToolBlock, Transcript

from .test_tui import SIZE, make_opener, plain, send, transcript_text, wait_for


class Seen(BaseModel):
    """The probe extension's payload: a tool it saw end."""

    tool: str
    status: str


class ProbeWindow(ModalScreen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "close", "Close")]
    DEFAULT_CSS = "ProbeWindow { align: center middle; } #probe-window { width: 40; height: 3; }"

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def compose(self):
        yield Static(Content(self.text), id="probe-window")

    def action_close(self) -> None:
        self.dismiss(None)


VIEWS: list[ProbeView] = []


class ProbeView(ExtensionView):
    """Puts a card under each bash call, fills it from the extension's `seen` events, opens a window on them."""

    def __init__(self, host, extension: str) -> None:
        super().__init__(host, extension)
        self.events: list[Event] = []
        self.card: Static | None = None
        self.window: ProbeWindow | None = None
        self.closed = False
        self.status_line = ""
        VIEWS.append(self)

    async def on_event(self, event: Event) -> None:
        self.events.append(event)
        match event:
            case ToolCallStart(call_id=call_id, name="bash"):
                self.card = Static(Content("probe: watching"), classes="probe")
                assert await self.host.attach(call_id, self.card)
            case ExtensionEvent(kind="seen", data=data) if event.extension == self.extension:
                seen = Seen.model_validate(data)
                assert self.card is not None
                self.card.update(Content(f"probe: {seen.tool} {seen.status}"))
                self.host.activity(f"probe saw {seen.tool}")
                self.status_line = plain(self.host.app.query_one(StatusBar))  # the next turn's start clears it
                self.window = ProbeWindow(f"{seen.tool} ended {seen.status}")
                await self.host.open(self.window)
            case _:
                pass

    async def close(self) -> None:
        self.closed = True
        if self.window is not None and self.window.is_current:
            self.window.dismiss(None)


class Probe(Extension):
    name = "probe"
    tui_view = "tests.test_tui_views:ProbeView"

    async def tool_result(self, call: ToolCall, result: ToolResult, ctx: RunContext) -> ToolResult | None:
        ctx.emit(ExtensionEvent.of(self.name, "seen", Seen(tool=call.name, status=result.status)))
        return None


class Broken(ExtensionView):
    async def on_event(self, event: Event) -> None:
        raise RuntimeError("boom")


class Breaks(Extension):
    name = "breaks"
    tui_view = "tests.test_tui_views:Broken"


class Missing(Extension):
    name = "missing"
    tui_view = "tests.no_such_module:View"


class NotAView(Extension):
    name = "not-a-view"
    tui_view = "tests.test_tui_views:Seen"


SCRIPT = [Turn(tool_calls=[("bash", {"command": "echo hi"})]), Turn(text="done")]


async def run_prompt(app: HarnessApp, pilot, text: str) -> None:
    await send(app, pilot, text)
    await wait_for(lambda: app._run_task is not None and app._run_task.done())  # pyright: ignore[reportPrivateUsage]
    await pilot.pause()


async def test_a_view_draws_what_its_extension_reports_under_the_call_and_in_a_window_of_its_own(tmp_path: Path):
    VIEWS.clear()
    journal = tmp_path / "session.jsonl"
    app = HarnessApp(make_opener(tmp_path, SCRIPT, journal=journal, extensions=(Probe,)))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await run_prompt(app, pilot, "say hi")
        [view] = VIEWS
        # The window the extension's view opened over the transcript, with the payload it validated back.
        assert isinstance(app.screen, ProbeWindow)
        assert plain(app.screen.query_one("#probe-window")) == "bash ended ok"
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ProbeWindow)
        # The card it put under the call, between the call and its output, and the status line.
        block = app.query_one(Transcript).query(ToolBlock).first()
        card = block.query_one(".probe", Static)
        assert plain(card) == "probe: bash ok"
        assert list(block.children).index(card) < list(block.children).index(block.out)
        assert "probe saw bash" in view.status_line
        assert "completed · 2 turns" in transcript_text(app)  # the TUI's own handling is unchanged
        kinds = [e.type for e in view.events]
        assert kinds.index("tool_call_start") < kinds.index("extension") < kinds.index("run_end")
        await app.action_quit()
    # Not journaled: a resumed session has none of them.
    entries = Journal(cwd=str(tmp_path), path=journal).active_path()
    assert not [e for e in entries if isinstance(e, EventEntry) and isinstance(e.event, ExtensionEvent)]


async def test_a_view_that_cannot_load_or_raises_is_named_and_the_tui_and_other_views_go_on(tmp_path: Path):
    VIEWS.clear()
    app = HarnessApp(make_opener(tmp_path, SCRIPT, extensions=(Missing, NotAView, Breaks, Probe)))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await pilot.pause()
        text = transcript_text(app)
        assert "Extension missing's view cannot load (tests.no_such_module:View): ModuleNotFoundError" in text
        assert "Extension not-a-view's view cannot load" in text and "not an ExtensionView subclass" in text
        await run_prompt(app, pilot, "say hi")
        await pilot.press("escape")  # the probe's window
        await pilot.pause()
        text = transcript_text(app)
        assert text.count("Extension breaks's view failed and is off for this session: RuntimeError: boom") == 1
        assert [type(v).__name__ for v in app.views.views] == ["ProbeView"]
        assert "probe: bash ok" in text and "completed · 2 turns" in text
        await app.action_quit()


async def test_views_get_live_events_only_and_go_with_their_session(tmp_path: Path):
    VIEWS.clear()
    journal = tmp_path / "session.jsonl"
    app = HarnessApp(make_opener(tmp_path, SCRIPT * 2, journal=journal, extensions=(Probe,)))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await run_prompt(app, pilot, "say hi")
        first = VIEWS[0]
        assert isinstance(app.screen, ProbeWindow)
        await app._clear()  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        # The old session's view closed its window; the new session has a view of its own.
        assert first.closed and not isinstance(app.screen, ProbeWindow)
        assert len(VIEWS) == 2 and app.views.views == [VIEWS[1]]
        await app.action_quit()
    # Resumed: the call is drawn again from the journal, but the view is given none of it.
    VIEWS.clear()
    app = HarnessApp(make_opener(tmp_path, [Turn(text="ok")], journal=journal, extensions=(Probe,)))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await pilot.pause()
        assert "$ echo hi" in transcript_text(app)
        [view] = VIEWS
        assert view.events == [] and view.card is None
        await app.action_quit()


def test_an_extension_event_is_part_of_the_protocol_and_acp_sends_nothing_for_it():
    event = ExtensionEvent.of("probe", "seen", Seen(tool="bash", status="ok"))
    assert event.data == {"tool": "bash", "status": "ok"}
    assert event_adapter.validate_json(event.model_dump_json()) == event
    assert "ExtensionEvent" in event_json_schema()["$defs"]
    assert acp_mapping.event_updates(event, acp_mapping.MappingContext(cwd=Path("/work"))) == []


async def test_a_widget_a_view_grows_under_a_call_keeps_the_transcript_at_its_end_unless_the_reader_scrolled_up(
    tmp_path: Path,
):
    VIEWS.clear()
    app = HarnessApp(make_opener(tmp_path, SCRIPT, extensions=(Probe,)))
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(lambda: app.session is not None)
        await run_prompt(app, pilot, "say hi")
        await pilot.press("escape")  # the probe's window
        [view] = VIEWS
        assert view.card is not None
        transcript = app.query_one(Transcript)

        async def grow(lines: int) -> None:
            view.card.update(Content("\n".join(f"probe line {i}" for i in range(lines))))  # pyright: ignore[reportOptionalMemberAccess]
            for _ in range(3):
                await pilot.pause()

        await grow(80)
        assert transcript.scroll_y == transcript.max_scroll_y > 0  # followed, though the view said nothing
        with view.host.growing():  # what views written for the old follow still wrap a change in
            await grow(160)
        assert transcript.scroll_y == transcript.max_scroll_y
        transcript.scroll_home(animate=False)
        await pilot.pause()
        await grow(240)
        assert transcript.scroll_y == 0  # the reader scrolled up: left where they are
        await app.action_quit()
