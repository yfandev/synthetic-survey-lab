"""Extension views: an extension's own drawing in the terminal UI, hosted by the TUI, written by the extension.

An extension names its view (`Extension.tui_view = "module:Class"`, a subclass of `ExtensionView`). When a
session opens, the TUI imports the views of the session's extensions, makes one per extension and gives each
every live event after its own handling: the `ExtensionEvent`s the extension emits (what it reports as it
works, in a payload type it owns) and the engine's own (a tool call starting, a run ending). A view draws
through its `ViewHost`: a window of its own over the transcript (a screen it supplies, with its own styles),
a widget under a tool call, a line in the transcript, the status line's activity. So what an extension shows
ships with the extension, and the TUI holds nothing particular to any of them.

Only live events reach views. A resumed session's journal holds no extension events, so a view would see
the replayed calls without what it drew for them.

Views fail open, as hooks do: one that cannot be imported, or raises, is named in a notice and dropped for
the rest of the session, and the TUI goes on.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from textual.content import Content
from textual.screen import Screen
from textual.widget import Widget

from intern.events import Event
from intern.tui.widgets import StatusBar, Transcript

if TYPE_CHECKING:
    from textual.app import App

    from intern.hooks import Extension


class ViewHost:
    """What a view may do on the screen. One per app, shared by its views."""

    def __init__(self, app: App[Any]) -> None:
        self.app = app
        """The Textual app, for what the host does not cover (a timer, a refresh); prefer the methods here."""

    async def open(self, screen: Screen[Any]) -> None:
        """A window of the view's own over the transcript, until it dismisses itself (on Esc, say)."""
        await self.app.push_screen(screen)

    async def attach(self, call_id: str, widget: Widget) -> bool:
        """`widget` under a running tool call, between the call and its output. False when that call is not
        running on screen."""
        transcript = self.app.query_one(Transcript)
        block = transcript.tool(call_id)
        if block is None:
            return False
        await block.attach(widget)
        return True

    @contextlib.contextmanager
    def growing(self) -> Iterator[None]:
        """Kept for views that wrap a change making an attached widget taller in it; it does nothing now: the
        transcript keeps showing its end as anything in it grows, and stays put if the reader has scrolled up."""
        yield

    async def marker(self, content: Content, key: str | None = None) -> None:
        """A muted line in the transcript; a later one with the same `key` replaces it."""
        await self.app.query_one(Transcript).marker(content, key=key)

    async def notice(self, text: str, kind: str = "info") -> None:
        """A line in the transcript in a notice's colour: `info`, `ok`, `warn` or `error`."""
        await self.app.query_one(Transcript).notice(text, kind)

    def activity(self, text: str) -> None:
        """What the run is doing, on the status line; '' clears it."""
        status = self.app.query_one(StatusBar)
        status.activity = text
        status.refresh_line()


class ExtensionView:
    """Base class of an extension's view: override `on_event`, and `close` when the view opens something that
    must go with the session."""

    def __init__(self, host: ViewHost, extension: str) -> None:
        self.host = host
        self.extension = extension
        """The extension's name: the `extension` of the `ExtensionEvent`s it emits."""

    async def on_event(self, event: Event) -> None:
        """One live event, after the TUI's own handling of it."""

    async def close(self) -> None:
        """The session is closing or reopening (a model pick, `/caps`, `/clear`): put away what the view opened."""


class Views:
    """The views of the open session's extensions."""

    def __init__(self, host: ViewHost) -> None:
        self.host = host
        self.views: list[ExtensionView] = []

    async def load(self, extensions: Iterable[Extension]) -> None:
        """The views of `extensions`, in place of the previous session's."""
        await self.close()
        for extension in extensions:
            if extension.tui_view is None:
                continue
            try:
                self.views.append(load_view(extension.tui_view)(self.host, extension.name))
            except Exception as e:
                await self.host.notice(
                    f"Extension {extension.name}'s view cannot load ({extension.tui_view}): {type(e).__name__}: {e}",
                    "error",
                )

    async def dispatch(self, event: Event) -> None:
        """Give `event` to every view, in the extensions' order; a view that raises is dropped."""
        for view in list(self.views):
            try:
                await view.on_event(event)
            except Exception as e:
                self.views.remove(view)
                with contextlib.suppress(Exception):
                    await view.close()
                await self.host.notice(
                    f"Extension {view.extension}'s view failed and is off for this session: {type(e).__name__}: {e}",
                    "error",
                )

    async def close(self) -> None:
        views, self.views = self.views, []
        for view in views:
            with contextlib.suppress(Exception):
                await view.close()


def load_view(spec: str) -> type[ExtensionView]:
    """The view class a `module:Class` names."""
    module_name, sep, class_name = spec.partition(":")
    if not sep:
        raise ValueError(f"{spec!r} is not module:Class")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, ExtensionView)):
        raise TypeError(f"{spec!r} is not an ExtensionView subclass")
    return cls
