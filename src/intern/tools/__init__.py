"""Tool packs. A task loads only the packs it needs; `base` is enough for any task with a shell.

Extensions add their tools after the packs'; an extension tool named like a pack tool takes its place.

`process` is a base tool: every session has a job manager wherever its commands can run in the
background. The `process` pack still names it, for configs that list it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from intern.models.registry import ConfigError
from intern.tools.base import Tool
from intern.tools.files import EditTool, ReadTool, WriteTool
from intern.tools.process import ProcessTool
from intern.tools.search import FindTool, GrepTool
from intern.tools.shell import BashTool
from intern.tools.web_read import WebReadTool
from intern.tools.web_search import WebSearchTool

if TYPE_CHECKING:
    from intern.hooks import Extension

PACKS: dict[str, list[type[Tool]]] = {
    "base": [ReadTool, WriteTool, EditTool, BashTool, GrepTool, FindTool, ProcessTool],
    "web-search": [WebSearchTool],
    "process": [ProcessTool],
    "web-read": [WebReadTool],
}


def load_packs(names: list[str]) -> list[Tool]:
    tools: dict[str, Tool] = {}
    for name in names:
        if name not in PACKS:
            raise ConfigError(f"unknown tool pack {name!r}; available: {', '.join(sorted(PACKS))}")
        for cls in PACKS[name]:
            tools.setdefault(cls.name, cls())
    return list(tools.values())


def with_extension_tools(tools: list[Tool], extensions: list[Extension]) -> list[Tool]:
    """The packs' tools, each replaced in place by an extension tool of the same name, then the new ones."""
    merged = {tool.name: tool for tool in tools}
    for extension in extensions:
        merged |= {tool.name: tool for tool in extension.tools()}
    return list(merged.values())
