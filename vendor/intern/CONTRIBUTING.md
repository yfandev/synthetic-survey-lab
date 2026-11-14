# Contributing

## Setup

```bash
uv sync
uv run pytest
```

Before pushing: `uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest`.

## Rules

- **One concern per PR**: a feature, a fix, a refactor or a dependency bump, not two.
- **Extensions, not branches.** Optional behaviour on the loop is code in an extension on one of the
  eight hooks. If a change needs a new hook, say why no existing one works. See
  [Extensions and capabilities](#extensions-and-capabilities) for which kind it is and what adding one
  involves.
- **System 1 calls go through `ctx.decisions`.** Ask System 1 only with
  `RunContext.decisions.ask()`, never with your own HTTP client, so every call gets the cache,
  retries and `decision` event. An ask System 1 does not answer raises `System1Unavailable`, which
  stops the run: let it through rather than answer for System 1. If `ask()` can't do what you need,
  change `DecisionClient` rather than work around it.
- **Evidence for defaults.** A new extension or a changed default needs a paired eval run: same
  tasks and models, at least two repeats, reported with pass rate, tokens, cost and wall time.
  "It fired N times" is not evidence it helped.
- **Fractions of the window for what shares it out.** Compaction, the reply reserve and kept history are profile
  thresholds, never hard-coded token counts. What one observation may hold is capped in tokens too
  (`max_tool_result_tokens`): a larger window does not make a model read more of one output, or pay less to be sent
  it again every turn.
- **Typed data.** Pydantic models across module boundaries; no free-form dicts.
- **Tests that can fail for a reason a user cares about.** The scripted FakeProvider and the local
  fake server in `tests/fake_openai.py` exercise real behaviour; don't pin prompt wording.

## Extensions and capabilities

### Words

- **Extension**: code on the eight hooks (`hooks.py`), which may also add tools and a terminal-UI view.
- **Capability**: a switch users flip during a session (`/caps`, the ACP `cap.*` options) and can save in
  `config.toml`. Behind it is an extension, a tool pack, an MCP connection or the browser. The code calls a
  switch a toggle.
- **Capability-backed extension**: an extension behind a capability (code search, the workspace, memory). It is
  not itself a capability; the capability is the switch in front of it.
- **Fixed extension**: an extension chosen for the whole session by the profile's `extensions` list or
  `--extension`, with no switch during the session.
- **Capability settings**: the `[capabilities]` table. Each top-level key is a switch setting, an option of one
  switch, or a session setting ([table](docs/capability-settings.md#what-each-setting-is)).
- **Tool pack**: a named group of built-in tools (`base`, `web-search`, `web-read`). `process` is in `base`;
  the `process` pack name is still accepted in a list, for older configs.
- **Profile**: a model's settings file (`profiles/*.toml`, or one in `~/.intern/profiles`).
- Not **protocol capabilities**: ACP and MCP use "capabilities" for feature negotiation
  (`agentCapabilities`). The harness's own names are always prefixed: `cap.*`, `[capabilities]`.

### Which kind

Whether users can switch it during a session decides the kind.

- **Users switch it during a session:** it sits behind a capability. Adding one means all of:
  - a `CapabilitySettings` field with its default, off unless a paired eval run supports on;
  - a row in `capability_toggles.toggles()`, a case in `flip()` and in the terminal UI's `_capability_key()`;
  - if it needs a package extra: the extra in `pyproject.toml`, a `_require_*` check and a branch in
    `session_setup.capability_problem()`, so ACP and `/caps` say why it is unavailable instead of failing on
    reopen;
  - for an extension: building it in `session_setup.capability_extensions()` and an entry in
    `config.CAPABILITY_EXTENSIONS` (class, switch id, the message shown when it is named as an extension);
  - rows in the `docs/acp.md` `cap.*` table and in [docs/capability-settings.md](docs/capability-settings.md).

  Capability-backed extensions run after fixed extensions. Memory is the one exception to where it is built:
  `Bootstrap.session` builds it with its drive and runs it last, so its write guard sees other extensions'
  argument changes.
- **Fixed for the session:** a fixed extension, registered in `extensions.REGISTRY`.

Either way, a new extension or a changed default needs the paired eval run above.
`tests/test_extension_kinds.py` fails when an extension the harness ships is neither, or both.

A fixed extension can later move behind a capability without breaking profiles: a profile that still lists it
in `extensions` turns the capability on by default for that profile and logs a deprecation warning, instead of
the refusal. Build that the first time an extension moves.

A switch can also go, when what it turned on becomes part of every session (the `process` pack, folded into
`base`): its id goes in `capability_toggles.RETIRED`, so a client or a journal that still sets it gets
no error and nothing changes, and its options stay in `[capabilities]` as session settings.

