"""The capabilities a user can switch on and off while a session runs: the TUI's `/caps` rows and the ACP `cap.*`
options are both these toggles. Behind a toggle is a tool pack, the browser, an MCP connection or an extension
(code search, the workspace, memory). Fixed extensions have no toggle: the profile and `--extension` choose them.
See CONTRIBUTING, "Extensions and capabilities"."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from intern.capability_settings import CapabilitySettings

PREFIX = "cap."
MCP_PREFIX = f"{PREFIX}mcp."
BASE = f"{PREFIX}base"
"""The `base` pack's row: listed by the TUI, never offered to ACP clients."""

PACKS = ("base", "web-search", "web-read")
"""The tool packs a session can switch (`docs/capability-settings.md`)."""
PACK_NAMES = {"base": "Base tools", "web-search": "Web search", "web-read": "Web read"}
PACK_NOTES = {
    "base": "read, write, edit, bash, grep, find, process: the agent's hands on the working directory",
    "web-search": "web_search: the web for source evidence, through Brave, Tavily, Exa or SearXNG",
    "web-read": "web_read: a URL fetched as source text the agent can read and quote",
}
RETIRED = frozenset({f"{PREFIX}process"})
"""Toggles no longer offered that a client or a journal may still set: accepted, and they change nothing. `process`
is a base tool."""
BROWSER_NOTE = "browser: Playwright pages the agent opens, reads, clicks, fills and screenshots"
CODE_SEARCH_NOTE = "decision_code: System 1 searches the repository for where something is defined or changed"
WORKSPACE_NOTE = "index_files, decision_files, sql, read: a folder of PDFs, Word files and workbooks the agent indexes"
MEMORY_NOTE = "a git repository of pages the agent recalls from at the start and adds to"


class Toggle(BaseModel):
    id: str
    """Stable across releases: the ACP config option id, and what the journal records."""
    name: str
    """What clients show."""
    label: str
    """What the TUI's `/caps` picker shows."""
    note: str
    on: bool


def packs_in_force(caps: CapabilitySettings, profile_packs: Sequence[str]) -> list[str]:
    """The tool packs in force: the settings' list, else the System 2 profile's."""
    return list(caps.tool_packs) if caps.tool_packs is not None else list(profile_packs)


def toggles(
    caps: CapabilitySettings, at_start: CapabilitySettings, profile_packs: Sequence[str], *, base: bool = False
) -> list[Toggle]:
    """One toggle per capability in capability order: the tool packs (`base` only when asked), the browser, code
    search, the workspace, memory, each MCP connection `at_start` (the configured settings) names, so one switched
    off is still listed. Where commands run is no toggle: it is the session's runner (`intern.runners`)."""
    packs = packs_in_force(caps, profile_packs)
    rows = [
        Toggle(
            id=f"{PREFIX}{pack}", name=PACK_NAMES[pack], label=f"{pack} tools", note=PACK_NOTES[pack], on=pack in packs
        )
        for pack in PACKS
        if base or pack != "base"
    ]
    rows += [
        Toggle(id=f"{PREFIX}browser", name="Browser", label="browser", note=BROWSER_NOTE, on=caps.browser.enabled),
        Toggle(
            id=f"{PREFIX}code-search",
            name="Code search",
            label="code search · System 1",
            note=CODE_SEARCH_NOTE,
            on=caps.code_search.enabled,
        ),
        Toggle(
            id=f"{PREFIX}workspace",
            name="Workspace",
            label="workspace · System 1",
            note=WORKSPACE_NOTE,
            on=caps.workspace.enabled,
        ),
        Toggle(id=f"{PREFIX}memory", name="Memory", label="memory drive", note=MEMORY_NOTE, on=caps.memory.enabled),
    ]
    for entry in at_start.mcp:
        names = list(entry.remote_tools)
        listed = ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
        rows.append(
            Toggle(
                id=f"{MCP_PREFIX}{entry.namespace}",
                name=f"MCP {entry.namespace}",
                label=f"MCP {entry.namespace}",
                note=f"remote tools over {entry.transport}: {listed}",
                on=any(e.namespace == entry.namespace for e in caps.mcp),
            )
        )
    return rows


def flip(
    caps: CapabilitySettings, at_start: CapabilitySettings, toggle_id: str, on: bool, profile_packs: Sequence[str]
) -> CapabilitySettings:
    """`caps` with one toggle set to `on`. Raises KeyError for an id that is not a toggle of these settings."""
    if toggle_id in RETIRED:
        return caps
    if toggle_id.startswith(MCP_PREFIX):
        name = toggle_id.removeprefix(MCP_PREFIX)
        entry = next((e for e in at_start.mcp if e.namespace == name), None)
        if entry is None:
            raise KeyError(f"no MCP connection called {name!r} in the config")
        mcp = [e for e in caps.mcp if e.namespace != name]
        return caps.model_copy(update={"mcp": [*mcp, entry] if on else mcp})
    pack = toggle_id.removeprefix(PREFIX)
    if toggle_id.startswith(PREFIX) and pack in PACKS:
        packs = [p for p in packs_in_force(caps, profile_packs) if p != pack]
        return caps.model_copy(update={"tool_packs": [*packs, pack] if on else packs})
    match toggle_id:
        case "cap.browser":
            return caps.model_copy(update={"browser": caps.browser.model_copy(update={"enabled": on})})
        case "cap.code-search":
            return caps.model_copy(update={"code_search": caps.code_search.model_copy(update={"enabled": on})})
        case "cap.workspace":
            return caps.model_copy(update={"workspace": caps.workspace.model_copy(update={"enabled": on})})
        case "cap.memory":
            return caps.model_copy(update={"memory": caps.memory.model_copy(update={"enabled": on})})
        case _:
            raise KeyError(f"no capability toggle {toggle_id!r}")


def values(caps: CapabilitySettings, at_start: CapabilitySettings, profile_packs: Sequence[str]) -> dict[str, bool]:
    """Each client toggle's id and whether it is on."""
    return {toggle.id: toggle.on for toggle in toggles(caps, at_start, profile_packs)}


def applied(
    caps: CapabilitySettings, at_start: CapabilitySettings, picks: Mapping[str, bool], profile_packs: Sequence[str]
) -> CapabilitySettings:
    """`caps` with every pick set. Raises KeyError for an id that is not a client toggle of these settings."""
    known = values(caps, at_start, profile_packs)
    for toggle_id in picks:
        if toggle_id not in known and toggle_id not in RETIRED:
            raise KeyError(f"no capability toggle {toggle_id!r}")
    for toggle_id, on in picks.items():
        caps = flip(caps, at_start, toggle_id, on, profile_packs)
    return caps
