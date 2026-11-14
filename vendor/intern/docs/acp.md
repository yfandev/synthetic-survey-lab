# Agent Client Protocol server

`intern acp` serves the [Agent Client Protocol](https://agentclientprotocol.com) (ACP v1) over
stdio, so t3code, Zed and other ACP clients can drive the harness and pick its System 2 and System 1
models. This page is the contract: a change to anything below ships as a release (see
[Versioning](#versioning)), and a breaking one moves the minimum version clients pin.

**Minimum version for clients: `0.25.0`.** Clients start `intern` and check `intern --version`. [0.2.1](#021) was
additive; [0.3.0](#030) changes the `system1` option and is where clients that showed `heuristic` need
a look. [0.4.0](#040) is additive: clients that offer a window picker need it. [0.6.0](#060) is
additive: clients that offer capability toggles need it, and [0.7.0](#070) tells them which toggles
cannot switch on here. [0.9.0](#090) and [0.10.0](#0100) added runner toggles, which [0.19.0](#0190) replaces with one
`runner` option. [0.14.0](#0140) removes the heuristic System 1: clients that send `heuristic` or read `fallenBack`
need a look, and a prompt may now fail with `system1_unavailable`. [0.15.0](#0150) is additive: clients that
offer a thinking picker read each model's `efforts`. From [0.23.0](#0230) a toggle switched on installs its missing
extra, and a toggle that cannot work here is not offered. [0.25.0](#0250) renames the harness to Intern, and
[0.27.0](#0270) removes its old names.

```bash
uv tool install "intern-agent[acp] @ git+https://github.com/nace-ai/intern@v0.30.1"
intern acp                         # System 2 from config.toml or $INTERN_SYSTEM2; System 1 drex-latest
intern acp --system2 openrouter/qwen/qwen3.8-27b --system1 drex-max-v1.1
```

`acp` takes the same options as `rpc` (`--system2`, `--system1`, `--window`, `--max-turns`,
`--profile`, `--no-context-files`, `--no-extensions`, `--extension`, `--no-session`) as defaults for
every session it opens. `--cwd` and `--session` are ignored: ACP sends the working directory with
each session, and journals are found by session id. The server needs the `acp` extra (the official
`agent-client-protocol` Python SDK); without it the command exits with code 2 and says so.

stdout carries only the protocol. Warnings and logs go to stderr. stdin and stdout must be pipes (or a
terminal); redirected to a file, the command exits with code 2. When stdin closes, every request
already received is still answered: runs in progress are aborted (their prompts answer `cancelled`)
and the server exits once the replies are written (at most 10 s later). On SIGTERM it aborts every
run and exits at once. Either way every session is closed.

## Lifecycle

| Method | What the harness does |
|---|---|
| `initialize` | Returns `protocolVersion: 1`, `agentInfo {name: "intern", title: "Intern", version}` and `agentCapabilities {loadSession: true, promptCapabilities {image: true, embeddedContext: true}}`. No auth methods. |
| `session/new {cwd, mcpServers, _meta?}` | Builds one harness `Session` for this ACP session; one process serves any number of sessions. `sessionId` is the session's journal id. Returns `configOptions` and `modes`. Opens on the models and capability toggles in [`_meta.nace`](#picking-models-at-open) when given. Fails with `-32602` when System 1 cannot be set up (its key is missing) or does not answer the check at open: the harness never runs without System 1. |
| `session/load {sessionId, cwd, mcpServers, _meta?}` | Finds the journal through `~/.intern/sessions/index.json` (falling back to a scan of `sessions/*.jsonl`), rebuilds the session on the System 2 and System 1 models and the capability toggles it last used (or those in `_meta.nace`), replays the active branch as `session/update`s, then returns `configOptions` and `modes`. Unknown ids fail with `-32002` (resource not found). |
| `session/prompt {sessionId, prompt}` | Runs one turn to the end, including follow-ups queued while it ran, streams `session/update`s, then resolves with `stopReason` and `usage`. When [capability toggles](#capability-toggles) changed since the last prompt, the session is first reopened on them. A prompt while another runs in the same session fails with `-32000` (`session busy`); a prompt before a System 2 is bound fails with `-32602` (see [below](#when-no-system-2-can-run)). |
| `session/cancel {sessionId}` | Aborts the run in progress; the pending `session/prompt` resolves with `stopReason: "cancelled"` once its last updates are sent. A cancel with nothing running is ignored. |
| `session/set_config_option {sessionId, configId, value}` | See [Config options](#config-options). Allowed only while the session is idle; returns every option with its current value. |
| `session/set_mode {sessionId, modeId}` | Accepts only `full-access`. |

Prompt content: text blocks are joined in order. An embedded text resource is added as a fenced block
headed by its URI; a resource link becomes its URI. An image is saved under the session's spill folder
and the prompt names its path, so the agent can `read` it (a model sees it when its profile sets `vision`
or its provider's model listing says it takes images).
Audio is refused.

`mcpServers` from the client are not used in v1: MCP servers come from the harness's own
`[[capabilities.mcp]]` config. `fs/*` and `terminal/*` client methods are not used: tools run in the
harness, on the session's `cwd`.

### Picking models at open

`session/new` and `session/load` take the models and capability toggles to open on in `_meta`, so a
session never has to start on the configured default first, or be rebuilt at its first prompt:

```json
{"_meta": {"nace": {"system2": "openrouter/qwen/qwen3.8-27b", "system1": "drex-latest", "runner": "docker",
                    "capabilities": {"cap.browser": true, "cap.web-search": false}}}}
```

Every key is optional; a missing one means the default (`session/new`) or what the journal last used
(`session/load`). A System 2 picked this way is journaled like a `model` change, and toggles like a
toggle change, so a later load without `_meta` continues on them. `capabilities` takes any of the
[`cap.*` ids](#capability-toggles); on `session/load` it is set over the journal's toggles. `runner` takes a
[runner](#runner) name, over the one the journal last ran on. A `_meta.nace` whose models are not strings, whose
`capabilities` names an id the session does not offer or a value that is not a JSON boolean, or whose `runner`
cannot run here, fails with `-32602`.

### When no System 2 can run

`session/new` and `session/load` succeed even when the System 2 they would open on cannot be bound:
none is configured, or its provider's API key is missing. The session is then open but unbound:

- `model` is advertised with `currentValue` the model it would have run on (`""` when none is
  configured) and the catalog as options; `system1` and the `cap.*` toggles as usual; no
  `thought_level` or `context_window`.
- `session/set_config_option model` binds it. A model that cannot bind either fails with `-32602` and
  the session stays unbound. `system1`, `context_window` and `cap.*` toggles may be set first and apply
  once a model binds.
- `session/prompt` fails with `-32602`: `no System 2 model can run: <reason>; pick a model`.
- `session/load` of an unbound session replays its history as usual.

Other configuration errors (an invalid `config.toml`, an unknown extension) still fail `session/new`
with `-32602`.

## Config options

Every session advertises these `configOptions`, in this order, then the
[capability toggles](#capability-toggles). Values are opaque strings, except the toggles' booleans.

| `id` | `category` | Type | Values | Default |
|---|---|---|---|---|
| `model` | `model` | select | System 2 as `provider/model`, grouped by provider | the session's System 2 |
| `system1` | `model_config` | select | decision models served by the console, then `decisions:<model>` for decision models on the decisions API | `drex-latest`, unless config or `--system1` says otherwise |
| `thought_level` | `thought_level` | select | `default` (no effort sent) through Anthropic's and OpenAI's APIs and where the profile lists `effort_levels`, then the levels the model takes: `low` to `max` through Anthropic's API, `low` to `xhigh` through OpenAI's, the profile's `effort_levels`, else `low`, `medium`, `high` | the model profile's `reasoning_effort`, else `default` |
| `context_window` | `model_config` | select | `auto`, or a token count: `32000`, `64000`, `128000`, `256000`, `512000`, `1000000` | `auto` |
| `runner` | `_nace_runner` | select | where commands run: `local`, `docker`, `opensandbox`, then each runner an installed package registers | `[capabilities.runner] name`, else `local` |

- **`model`** lists the [model catalog](#model-catalog)'s System 2 models, plus the current model. The
  catalog is fetched in the background when the first session opens, so `session/new` never waits on
  provider listings: until it arrives the list holds only the current model, and then every open
  session gets a `config_option_update` with the full list. Any `provider/model` the harness can
  resolve is accepted, listed or not, so clients may offer custom models. Changing it rebinds System 2
  (profile, window) and journals a `model_change` entry. Its options are ACP v1 option groups,
  `{group, name, options: [{value, name}]}`, with `group` the provider; that key is what
  `protocolVersion: 1` defines, so clients generated from a later ACP schema should not expect another
  name for it.
- **`system1`** lists the [model catalog](#model-catalog)'s System 1 models: the decision models the console serves (`POST {console_url}/systemone` with the key in `$NACE_API_KEY`; `[system1].models`,
  default `["drex-latest"]`, until the console has been listed), then the decision models the decisions
  API serves as `decisions:<model>` (`POST {base_url}/systemone` with the key in `$OPENROUTER_API_KEY`:
  Jev, `nace-ai/drex-max`, ...). Like `model`, the list fills in once the catalog arrives; until then it
  holds only the current value. The System 1 the session started on (from `[system1]` or `--system1`,
  as `decisions:<model>` when on the decisions API) is listed too and keeps its own endpoint and
  fallbacks. Any model name is accepted, listed or not; there is no `heuristic` (System 1 off). Changing
  it rebuilds the session's System 1 client, keeping its usage, and journals a `model_change` entry. A model whose key is missing is refused with
  `-32602` and the session keeps the System 1 it had.
- **`thought_level`** is advertised whenever System 2 takes a reasoning effort: every model on Anthropic's
  and OpenAI's APIs, which take one for every model they serve, and an OpenAI-compatible model when its
  profile lists `effort_levels` or sets `reasoning_effort` (some of those endpoints reject the field, so
  the profile is the only word). It sets the effort sent with each request, among the levels the model
  takes (Anthropic's `output_config.effort` goes to `max`, OpenAI's `reasoning.effort` to `xhigh`, a
  profile's `effort_levels` are its own, and a profile with only `reasoning_effort` takes `low`, `medium`,
  `high`). On Anthropic's and OpenAI's APIs and with `effort_levels`, `default` comes first and sends
  none, so the model uses its own; it is the current value when the profile names no effort. The packaged
  profiles list `effort_levels` for DeepSeek V4 Flash on Fireworks. It holds across a capability
  reopen but is not journaled: a loaded session starts at the profile's value. A `model` change may add or drop it; the
  `set_config_option` response always lists every option as it now stands.
- **`context_window`** caps System 2's working window for this session. `auto` is the process default:
  the model's window, or `--window` when the server was started with it (the option's name says which,
  for example `Auto (128K)`). The presets are listed up to what the model can use (its window, no larger
  than its profile's effective window), plus the current cap. Any positive token count is accepted,
  listed or not; one above what the model can use leaves the window there. The cap holds across `model`
  changes, is journaled as a `window_cap` entry and restored on `session/load`, and the next
  `usage_update` reports the new `size`. Anything other than `auto` or a positive integer fails with
  `-32602`.

```toml
# ~/.intern/config.toml
[system1]
models = ["drex-latest"]                  # the picker's Drex models until the console is listed; the first is the default
console_url = "https://console.nace.ai/v1" # where they are served and listed ($DREX_BASE_URL overrides it)
console_key_env = "NACE_API_KEY"           # ($DREX_API_KEY also works)
base_url = "https://openrouter.ai/api/v1"  # the decisions API, for decisions:<model>
api_key_env = "OPENROUTER_API_KEY"
```

An unknown `configId`, a `thought_level` or `context_window` value it does not take, or a model the harness cannot resolve (for
example, its provider's API key is missing) fails with `-32602` (invalid params) and the harness's
message. A change during a run fails with `-32000` (`session busy`).

### Runner

`runner` picks where the session's commands, file tools and background jobs run: one
[runner](capabilities/runners.md) per session. Its values are every runner this install knows, in order:
`local`, `docker`, `opensandbox`, each runner an installed package registers, then the configured one when it is a
`module:Class`. Each value is `{value, name, description}`. When the runner cannot run in this install and
environment, `description` starts with `Unavailable:` and gives the reason. The reason is a missing extra, an unset
key, or options that only `config.toml` can give. `models --json` gives the same list under
[`runners`](#model-catalog).

- `session/set_config_option {configId: "runner", value}` takes a runner name.
  - A runner that cannot run here fails with `-32602` and its reason, and the session keeps the runner it has.
  - Setting the runner in force changes nothing.
  - Any other runner is pending: the response shows it as the current value, and the next `session/prompt` applies
    it, with every capability change made by then, in one reopen on the same journal (see below).
- The configured runner runs with its configured `[capabilities.runner.options]`. Any other runner runs on its own
  defaults.
- The runner is journaled in the `capabilities_change` entry and restored on `session/load`, unless it can no
  longer run there. A session with no journal (`--no-session`) refuses a change with `-32602`.

### Capability toggles

After the options above, every session (unbound ones too) advertises one boolean option per capability
a user can switch on or off: the same list as the terminal UI's `/caps`, without the `base` pack.
Fixed extensions (a `module:Class`) have no toggle: the profile and `--extension`
choose them for the whole session. Extensions behind a capability (code search, the workspace, memory) are
switched by that capability's toggle ([CONTRIBUTING](../CONTRIBUTING.md#extensions-and-capabilities)).

| `id` | Turns on | Listed |
|---|---|---|
| `cap.web-search` | the `web-search` tool pack | always |
| `cap.web-read` | the `web-read` tool pack | always |
| `cap.browser` | `[capabilities.browser] enabled` | always |
| `cap.code-search` | `[capabilities.code_search] enabled` | always |
| `cap.workspace` | `[capabilities.workspace] enabled` | always |
| `cap.memory` | `[capabilities.memory] enabled` | always |
| `cap.mcp.<namespace>` | that `[[capabilities.mcp]]` connection | for each connection config.toml names |

A toggle that cannot switch on in this install and environment, even after the install below, is left out unless
it is on: web search with no provider or key, or an extra with no uv or pip to install it.
[`models --json`](#model-catalog) still lists it, with the reason.

Each is `{type: "boolean", category: "_nace_capability", name, description, currentValue}`, with
`currentValue` what config.toml and the profile turn on until the session changes it. Pack toggles
start from `tool_packs`, or the profile's packs when it is unset. Ids are stable; readers should show
any `cap.*` option they do not know by its `name` and `description`. `cap.process` is no longer offered: the
`process` tool is a base tool from 0.30.0. A client that still sets it, with `session/set_config_option` or in
`_meta.nace.capabilities`, gets no error, and nothing changes.

`session/set_config_option {configId: "cap.…", type: "boolean", value}` takes a JSON boolean
(`-32602` otherwise, or for an id the session does not offer) and an idle session (`-32000`). Setting
the value in force changes nothing, so a client may send every toggle on a thread's first turn. A
change is pending: the response shows it as the option's value, and the next `session/prompt` applies
every change made by then at once by reopening the session on the same journal (same `sessionId`,
conversation, models, thinking level, window cap and usage totals) before it runs.

A toggle switched on whose capability lacks its extra (`browser`, `code-search`, `workspace`, `mcp`) gets it
installed first, into the environment the harness runs in, with uv's pip or else pip, then the capability's setup
(`playwright install chromium` for the browser, which downloads Chromium once). The prompt waits for it, and the
session gets agent message text saying what is being installed. `session/cancel` stops the install, the prompt
answers `cancelled`, and the change stays pending for the next prompt. An install that fails leaves its toggle off
as below, with the reason and the command to run by hand.

A toggle switched on that still cannot switch on in this install and environment (the
[`unavailable`](#model-catalog) reason: a missing search provider, or an extra that could not be installed) stays
off while the other changes apply. The harness sends a `config_option_update` showing it off, and that prompt fails with
`-32602`: `<reason>. <Name> stays off; the other capability changes applied` (the last clause only
when another change applied). A client that sends the switch again gets the same answer at its next
prompt. When the session cannot open on the rest (for example an MCP connection that fails to start), it
opens again on the toggles it had, sends a `config_option_update` with them, and that prompt fails with
`-32602`: `capabilities not switched: <reason>; the session continues on the ones it had`.

The toggles are journaled as a `capabilities_change` entry and restored on `session/load`, except one
that can no longer switch on, which the loaded session leaves off. A session started with
`--no-session` (no journal) refuses a change with `-32602`.

A reopen is not the session's end, so it is not stewarded into memory; `session/close` and shutdown
are. It starts the capabilities again: code search indexes the working directory at the next turn, and
a configured `[capabilities.session_budget]` starts a new ledger.

## Modes

`modes` advertises one mode, `full-access` (the agent runs every tool without asking). There are no
`session/request_permission` calls in v1.

## Updates

`session/update` notifications, from harness events:

| Harness event | ACP update |
|---|---|
| `text_delta` | `agent_message_chunk` (text) |
| `reasoning_delta` | `agent_thought_chunk` (text) |
| `tool_call_start` | `tool_call`, `status: "in_progress"`, `kind` from the tool (below), `title` the tool name and its main argument, `rawInput` the arguments, `locations` from a `path` argument |
| `tool_call_end` | `tool_call_update`, `status` `completed` (`ok`) or `failed` (`error`, `refused`), `content` the result text the model saw; for a `write` or `edit` that changed a text file, also a `diff` item `{path, oldText, newText}` (`oldText` null for a new file); none for binary files, files over 256k characters before and after together, or the Docker workspace's tools |
| `context_budget` | `usage_update {used: tokens in the request, size: working window}` |
| `run_end` | `usage_update {used, size, cost: {amount: session cost so far, currency: "USD"}}` |
| everything else | nothing |

Tool `kind` comes from the tool's declared effects: `grep` and `find` are `search`; otherwise `exec`
is `execute`, then `network` is `fetch`, then `write` is `edit`, then `read` is `read`; a tool with no
effects is `other`.

On `session/load` the active branch is replayed: user prompts as `user_message_chunk`, replies as
`agent_thought_chunk` and `agent_message_chunk`, and each tool call as a `tool_call` followed by its
`tool_call_update` (without a diff: diffs are not journaled). Messages the harness added itself
(nudges) are not replayed.

A session that memory has stewarded since it last closed is then sent one more `usage_update`. Its
`cost` is what the steward spent on the session: its System 1 asks and its writer's System 2 calls.
The session's cost so far starts from it. `_meta.nace.steward` gives the totals of every steward run
on the session, split by system. A writer whose provider reports no cost and that has no price in the
harness's table has `costUsd` 0 with its tokens counted.

```json
{"sessionUpdate": "usage_update", "used": 0, "size": 65536, "cost": {"amount": 0.0191, "currency": "USD"},
 "_meta": {"nace": {"steward": {
   "runs": 2, "costUsd": 0.0191,
   "system1": {"asks": 43, "failed": 0, "inputTokens": 90000, "outputTokens": 900,
               "costUsd": 0.013},
   "writer": {"inputTokens": 12000, "outputTokens": 1500, "cachedReadTokens": 0, "cachedWriteTokens": 0,
              "thoughtTokens": 0, "costUsd": 0.0061}}}}}
```

## Stop reasons and errors

| Harness `run_end.reason` | Result |
|---|---|
| `completed`, `stopped_by_hook`, `stopped_by_tool`, `stopped_by_guard` | `stopReason: "end_turn"` |
| `aborted` | `stopReason: "cancelled"` |
| `turn_budget` | `stopReason: "max_turn_requests"` |
| `output_limit` | `stopReason: "max_tokens"` |
| `provider_error` | JSON-RPC error `-32001`, `data.kind` below |
| `system1_unavailable` | JSON-RPC error `-32603`, `data: {"reason": "system1_unavailable"}` |
| `internal_error` | JSON-RPC error `-32603` |

A provider error carries `data: {kind, detail, status}`, where `status` is the HTTP status when there
was one and `kind` is:

- `rate_limit`: HTTP 429, or the provider said so (rate limit, too many requests);
- `quota`: HTTP 402, or out of credits, quota or billing;
- `auth`: HTTP 401 or 403;
- `provider`: anything else.

Clients show `rate_limit` and `quota` as limited, not as a crash. The updates sent before the error
stay valid; the turn is journaled up to the failure.

`system1_unavailable`: System 1 did not answer an ask the run needed, after retrying with backoff
(1 s, doubling to 15 s) for about two minutes. The message names the backend and its last error. The
harness does not run without System 1, so nothing answered in its place; the session stays open and
the next prompt asks again. A session whose System 1 does not answer when it opens fails
`session/new` (or `session/load`) with `-32602`, as one whose System 1 key is missing does.

`usage` on the prompt response is the turn's System 2 tokens: `inputTokens`, `outputTokens`,
`cachedReadTokens`, `cachedWriteTokens`, `thoughtTokens`, `totalTokens`.

`_meta.nace.usage` splits the turn's work by system, follow-ups included. Clients sum turns for a
session's totals; `usage_update`'s `cost` is the session's total for both.

```json
{
  "system2": {
    "model": "openrouter/deepseek/deepseek-v4-flash",
    "inputTokens": 2152,
    "outputTokens": 5,
    "cachedReadTokens": 1792,
    "cachedWriteTokens": 0,
    "thoughtTokens": 0,
    "costUsd": 0.000087
  },
  "system1": {
    "model": "drex-latest",
    "asks": 3,
    "failed": 0,
    "inputTokens": 1840,
    "outputTokens": 96,
    "costUsd": 0.00042
  }
}
```

`asks` are those the backend answered; `failed` are those it did not answer after every retry (each
one stopped the run), and answers from the cache count in neither. Tokens and cost are the backend's.

System 2's cost is the provider's where it reports one (OpenRouter). Anthropic's and OpenAI's own APIs
and Fireworks report only tokens, so the harness prices them from the models' published prices
(`src/intern/models/prices.toml`): standard tier, the 5-minute cache on Anthropic and the
default 30-minute one on OpenAI, the higher rates for a long prompt where the model has them. A model
with no price there costs 0, such as a Fireworks dedicated deployment, which is billed per GPU-second.

## Model catalog

`intern models --json` prints the catalog without opening a session, for status probes:

```json
{
  "system2": [
    {"id": "fireworks/accounts/example/deployments/qwen3p8-27b", "name": "Qwen3.8 27B (Fireworks)", "default": true, "reasoning": true, "efforts": ["low", "medium", "high"], "window": 131072},
    {"id": "anthropic/claude-opus-5-5", "name": "Claude Opus 5.5", "reasoning": true, "efforts": ["default", "low", "medium", "high", "xhigh", "max"], "window": 1000000},
    {"id": "vllm/my-model", "name": "my-model"}
  ],
  "system1": [
    {"id": "drex-latest", "name": "drex-latest"}, {"id": "drex-v2.0", "name": "drex-v2.0"},
    {"id": "decisions:~typesafe/jev-latest", "name": "TypeSafe: Jev Latest"}
  ],
  "capabilities": [
    {"id": "cap.web-search", "name": "Web search", "description": "web_search: the web for ...", "enabled": false,
     "unavailable": "Web search is not set up: set INTERN_SEARCH_PROVIDER to brave, tavily, exa or searxng"},
    {"id": "cap.browser", "name": "Browser", "description": "browser: Playwright pages ...", "enabled": true}
  ],
  "runners": [
    {"id": "local", "name": "Local", "description": "commands run directly on this machine", "default": true},
    {"id": "docker", "name": "Docker", "description": "bash, the file tools and background jobs run in a container ...", "default": false},
    {"id": "opensandbox", "name": "OpenSandbox", "description": "...", "default": false,
     "unavailable": "OpenSandbox needs $OPEN_SANDBOX_API_KEY; a server without a key takes api_key_env = \"\""}
  ]
}
```

`runners` lists the values of the [`runner` option](#runner), in its order: `default` marks the configured one, and
`unavailable` (only when it cannot run here) says why. `capabilities` lists the [capability toggles](#capability-toggles) a session offers, in their order,
each with `enabled` as config.toml and the default System 2's profile turn it on. Every entry has
`id`, `name`, `description` and `enabled`. From 0.7.0, `unavailable` is present when the toggle cannot
switch on in this install and environment, with the reason: web search has no provider or key, or an extra is
missing and there is no uv or pip to install it. From 0.23.0, an extra that switching on installs is not a reason.
Combinations are not covered; a reopen still rejects them.

Every entry has `id` and `name`. System 2 entries may also carry these keys, each only when true or
known (readers must ignore keys they do not know):

- `default`: the System 2 a session opens on unless the client picks one (`--system2`, then
  `system2`/`model` in config.toml, then `$INTERN_SYSTEM2`). At most one entry has it; none when
  no default is configured or its provider cannot be reached.
- `reasoning`: the model takes a reasoning effort, so a session on it advertises `thought_level`.
- `efforts` (0.15.0 and later): the values that `thought_level` takes on it, in the order a session lists
  them: `default` first on Anthropic's and OpenAI's APIs and with a profile's `effort_levels`, then the
  levels, least to most.
- `window`: the most context, in tokens, a session on it can use: the profile's `window`, else what the
  provider's listing says, no larger than the profile's `effective_window`. Absent when neither says. A
  session's `context_window` presets go up to it.

System 2 lists every model served by a provider the harness can reach: each built-in provider and
each `[providers]` entry whose API key variable is set (or that needs none), asked with `GET
{base_url}/models` (5 s timeout each, all at once; one that fails or times out is left out), keeping
only models System 2 can chat with: the listing decides where it says (a `type` such as `embedding` or
`audio`, or Fireworks' `kind` `EMBEDDING_MODEL`, first; then OpenRouter's `architecture.output_modalities`
or Fireworks' `supports_chat`),
else a model whose id names an embedding, speech, image, moderation or completions-only model
(`text-embedding-3-small`, `tts-1`, `whisper-large-v3`, `dall-e-3`, `omni-moderation-latest`, `davinci-002`) is
left out. To those it adds `system2` from config and the concrete names in profile `match` lists (how models a provider does not
list, such as Fireworks dedicated deployments, get in). Ids are `provider/model`; names are the
provider's display name where it gives one, else the profile's `name` for a model from a profile. The server caches the catalog for 10
minutes. System 1 is listed from the console (`GET {console_url}/models` with `$NACE_API_KEY`,
retired models left out; `[system1].models` stand in when the key is missing or the console does not
answer) and from the decisions API (`GET {base_url}/models?output_modalities=decisions` with
`$OPENROUTER_API_KEY`, OpenRouter's filter for decision models), as `decisions:<model>`. Exit code 0,
even when every provider fails (the lists are then shorter); 2 on a configuration error.

## Versioning

- The harness's event protocol (`intern schema`, `intern rpc`) is at `PROTOCOL_VERSION`
  2: `tool_call_end` gained an optional `diff`.
- ACP clients pin a minimum harness version, never a commit. Releases are tagged `vX.Y.Z` by CI when
  a version bump lands on `main`; [AGENTS.md](../AGENTS.md#release-what-clients-see) says when to bump.
- Planned (not in v1): `session/request_permission` from the tool hook, which adds supervised modes.
- Journals written by 0.6.0 may hold `capabilities_change` entries, which older harnesses cannot load.

### 0.2.1

Additive; 0.2.0 clients keep working.

- `session/new` and `session/load` take `_meta.nace.system2` / `system1`, and succeed when no System 2
  can run, opening the session unbound until the client picks a model.
- The catalog lists the concrete ids in profiles' `match` lists, such as a Fireworks dedicated deployment;
  profiles take an optional `name` for them.
- `nace-harness models --json` marks the default System 2 with `"default": true` and models that take
  a reasoning effort with `"reasoning": true`.
- Documented: grouped `model` options use ACP v1's `group` key (unchanged since 0.2.0).

### 0.3.0

System 1 is required; clients that showed or sent `heuristic` need a look.

- The `system1` option no longer lists `heuristic`, defaults to `drex-latest`, lists the console's
  decision models and the decisions API's models as `decisions:<model>`, and reports a session started on
  `--system1 decisions:<m>` as `decisions:<m>`. `heuristic` is still accepted when sent, as the
  explicit opt-out.
- `session/new` fails with `-32602` when System 1 cannot be set up (its key is missing) instead of
  opening on heuristics; a `system1` pick whose key is missing is refused with `-32602` and the session
  keeps its System 1.
- `--system1` also takes a bare Drex model (`drex-max-v1.1`) for the console; `drex:<model>[,...]`
  keeps working.

### 0.4.0

Additive; 0.3.0 clients keep working.

- Live sessions advertise [`context_window`](#config-options), a per-session cap on System 2's working
  window that holds across model changes and loads.
- `nace-harness models --json` gives System 2 entries a `window` when the profile or the provider's
  listing says it.
- `thought_level` lists the levels the model takes: `low` to `max` through Anthropic's API, `low` to
  `xhigh` through OpenAI's, still `low`, `medium`, `high` elsewhere.

### 0.5.0

Additive; 0.4.0 clients keep working.

- The prompt response carries [`_meta.nace.usage`](#stop-reasons-and-errors): the turn's System 2 model,
  tokens and cost, and its System 1 model, asks, failures, fallbacks, tokens and cost.

### 0.5.1

Fix; 0.5.0 clients keep working.

- System 2 on Anthropic's and OpenAI's own APIs has a cost: `system2.costUsd` in
  [`_meta.nace.usage`](#stop-reasons-and-errors) and `usage_update`'s `cost` were 0 for them, since
  those APIs report only tokens, and are now priced from the models' published prices.

### 0.6.0

Additive; 0.5.0 clients keep working and see the toggles as extra boolean options.

- Sessions advertise [capability toggles](#capability-toggles): one boolean `cap.*` option per
  capability, category `_nace_capability`, applied by reopening the session at the next prompt,
  journaled as `capabilities_change` and restored on `session/load`.
- `session/new` and `session/load` take [`_meta.nace.capabilities`](#picking-models-at-open).
- `nace-harness models --json` lists [`capabilities`](#model-catalog) with each one's configured
  default.

### 0.7.0

Additive; 0.6.0 clients keep working.

- `nace-harness models --json` capabilities carry [`unavailable`](#model-catalog), the reason one
  cannot switch on here, so a client can show it before the user switches it on.
- A [capability switch](#capability-toggles) that cannot switch on here stays off and is named in the
  prompt's `-32602`, while the other changes made with it apply. Before, one such switch kept all of
  them off. The error no longer runs a reason's full stop into the next clause.
- `session/load` leaves off a journaled toggle that can no longer switch on, instead of failing.

### 0.8.0

Agent behaviour only; nothing in the protocol changes, so 0.7.0 clients keep working.

- A session's first message is no longer pinned in every request. Every message of the user's is a
  turn of the conversation, and once compaction cuts it, the State keeps the user's last three
  requests as they wrote them. A later request is no longer left to a summary of it while the first
  one stays forever.
- Every model brings the State up to date with edits after its first one (before, only Opus, GPT and
  Qwen3.8 27B did), and a new message from the user no longer makes the next State a full rewrite. A
  profile of your own that still sets `state_edits` loads as before; the key is ignored.

### 0.9.0

Additive; 0.8.0 clients keep working and show the new toggles as extra boolean options.

- A session picks where its commands run: next to `cap.docker`, one [`cap.executor.<name>`
  toggle](#capability-toggles) per executor backend an installed package registers (and the configured
  one when it is not built in), listed in `models --json` with `unavailable` when it cannot switch on.
  The executor toggles are one choice.
- `cap.docker` switched off goes back to the configured backend; before, it always went to `local`.

### 0.9.1

Fix; 0.9.0 clients keep working.

- `cap.process` works with Docker (and any runner that runs background jobs): each job runs where the session's
  commands run, in its own container under Docker. Before, the prompt after switching it on failed.

### 0.10.0

Additive; 0.9.0 clients keep working and show the new runner as one more toggle.

- `cap.executor.opensandbox`: the [OpenSandbox](capabilities/opensandbox.md) runner ships with the harness, so its
  toggle is always listed. `models --json` gives it `unavailable` until the `opensandbox` extra is installed and its
  API key is set.
- A runner toggle's `unavailable` now also carries the backend's own reason (a missing extra or key), not only
  "cannot load" or "needs options".

### 0.10.1

Additive; nothing an ACP session sends changes, so 0.10.0 clients keep working.

- The harness's event protocol (`nace-harness schema`, `nace-harness rpc`, `-p --json`) has one more
  event, `extension`: an extension's own report as it works (`extension`, `kind`, and a `data` object
  whose shape the extension owns). Still `PROTOCOL_VERSION` 2: no event changed. A client that reads the
  stream should skip event types it does not know; ACP sends nothing for this one. In the terminal UI,
  an extension can draw these with a view of its own ([docs/tui.md](tui.md#extension-views)).

### 0.10.2

Fix; 0.10.1 clients keep working.

- `session/close` over stdio closes the session and stewards it into memory. Before, it answered
  `-32601` Method not found, although `initialize` advertises it, so a closed session never reached
  memory.

### 0.10.3

Fix; nothing an ACP session sends changes, so 0.10.2 clients keep working.

- `decision_code` indexes a repository the size of audit-app (16k files git sees; 478k with its ignored client
  workspaces): in a git work tree the index holds the files git sees, so what `.gitignore` keeps out of git is
  neither indexed nor searched, and the index's TF-IDF rows and neighbours are sparse and computed in blocks. The
  build there takes 28 s and 1.9 GB where it took 12 GB, or ran out of memory and took the terminal with it. One
  index holds at most 100,000 files; past that `decision_code` and `nace-harness index` say so instead of building.
  The `code-search` extra now includes scipy.

### 0.11.0

Additive; nothing an ACP session sends changes, so 0.10.3 clients keep working.

- `decision_code`'s `scope` may name a folder outside the working directory, such as another repository: the git
  work tree holding it is indexed the first time (into its own `.nace/index.json`, kept and refreshed like the
  working directory's) and searched within the scope; the places come back with paths `read` opens from the
  working directory. An `indexing` event's `root` is then that repository, not the working directory.

### 0.12.0

Additive; 0.11.0 clients keep working, and show the steward's cost in the session's `cost` with no change.

- What the memory steward spent on a session after it closed now reaches the client: its System 1
  asks and its writer's System 2 calls. `session/load` sends a [`usage_update`](#updates) with
  that cost, and the session's cost so far starts from it. `_meta.nace.steward` on that update has the
  tokens and cost of every steward run on the session, split into `system1` and `writer`. Before, the
  steward's cost was shown nowhere.

### 0.13.0

Additive; 0.12.0 clients keep working with no change.

- A session the client keeps open is stewarded into memory once it has been idle
  `[capabilities.memory] steward_idle_s` seconds after a prompt (default 300; 0 waits for
  `session/close` as before). Before, a client that never closed its sessions, such as T3 Code with its
  threads, reached memory only when its agent process exited. A prompt after that is stewarded again;
  the steward's cost still arrives on `session/load`.

### 0.13.1

Agent behaviour only; nothing in the protocol changes, so 0.13.0 clients keep working.

- A session's first State is written once System 1 finds the work worth one, usually within a task's
  first few turns. Before, System 1 was asked whether a State that did not exist yet was out of date,
  so a chat or short session on a large window could go without a State until compaction, often for
  its whole life, while a single lookup sometimes got one. A greeting, a quick answer or a lookup
  still gets none.

### 0.14.0

Breaking: System 1 is required at every step. Clients that send `heuristic` or read `fallenBack` need a
look; others keep working, and see one more prompt error.

- There is no heuristic System 1. `--system1 heuristic` and a `system1` pick of `heuristic` are refused
  with "System 1 is required", and a config with `[system1] backend = "heuristic"` no longer loads. Nothing stands in for an
  answer: not for the supervisor, the guard, compaction's keep verdicts, memory recall or `decision_code`.
- An ask System 1 fails is retried with backoff for about two minutes; still unanswered, the prompt fails
  with [`system1_unavailable`](#stop-reasons-and-errors) (`-32603`). Before, a heuristic answered and, for
  the guard, the call went through unjudged.
- `session/new` and `session/load` check that System 1 answers (up to 15 s of retries) and fail with
  `-32602` when it does not.
- `fallenBack` is gone from `_meta.nace.usage.system1` and `_meta.nace.steward.system1`; `failed` counts
  the asks left unanswered after their retries.
- The System 1 cost cap and circuit breaker are gone; a session budget
  (`[capabilities.session_budget]`) still bounds System 1's calls and cost.

### 0.15.0

Additive; 0.14.0 clients keep working, and see a thinking picker on more models.

- `thought_level` is advertised for every model on Anthropic's and OpenAI's APIs, not only those whose
  profile sets `reasoning_effort`, as the terminal UI already offered it. Before, a Claude model without a
  packaged profile, and every GPT model, had none.
- On those APIs `thought_level` lists `default` first: no effort sent, so the model uses its own. It is
  the current value when the profile names no effort, and picking it goes back to sending none.
- `nace-harness models --json` gives System 2 entries `efforts`, the values `thought_level` takes on that
  model, so clients can offer the right levels before a session opens. `reasoning` now means the model
  takes an effort at all, as `thought_level` does.
- System 2 in `models --json` and the `model` option lists only models System 2 can chat with. Before,
  embedding, text-to-speech, transcription, image and moderation models a provider serves were listed too,
  and a session on one failed at its first prompt.

### 0.15.1

Fix; nothing an ACP session sends changes shape, so 0.15.0 clients keep working.

- A reply from a Fireworks serverless model now has a cost: Fireworks reports only tokens, and they are
  priced from its published Standard rates (input, cached input, output). Before, `usage_update`'s
  `cost` and `_meta.nace.usage.system2.costUsd` were 0 for them, and the TUI said `price unknown`. A
  dedicated deployment is billed per GPU-second, not per token, so it still costs 0.

### 0.15.2

A fix; older clients keep working with no change.

- Memory keeps what the user said in a long message. The steward used to read only the first 4,000
  characters of each user message, and T3 Code puts about 5 KB of its own instructions before the user's
  words in the first message of a thread. So nothing said in a thread's first message reached memory. A
  long message now keeps its end as well as its start.

### 0.15.3

Additive; 0.15.0 clients keep working, and see a thinking picker on more models.

- A profile can list `effort_levels`, the reasoning efforts an OpenAI-compatible model takes, which that
  API does not report. Its sessions then advertise `thought_level` with `default` first, and `models
  --json` lists them in `efforts`. Until a level is picked no effort is sent, so the model behaves as
  before; a model whose endpoint rejects the field fails only once a level is picked. Unknown levels make
  the profile fail to load.
- DeepSeek V4 Flash on Fireworks (`deepseek-v4p1-flash`, its `-1017` build and the spot and ultrafast
  routers) gets a packaged profile listing `low` to `max`. Each variant took every level, and refused an
  unknown one with 400.

### 0.15.4

Fix; 0.15.3 clients keep working.

- The OpenSandbox runner (`cap.executor.opensandbox`) runs commands on a real OpenSandbox server: every command
  failed with a 400 before, because commands went as an argument list, which the server's execd does not take.
  Now checked against a live server (copy and mount workspaces, deadlines, abort, background jobs).

### 0.15.5

A fix; older clients keep working with no change.

- Fireworks' embedding and reranker models (`qwen3-embedding-8b`, `qwen3-reranker-8b`) are no longer listed
  as System 2. Fireworks marks them `supports_chat`, which the catalog trusted; their `kind`,
  `EMBEDDING_MODEL`, now decides first.

### 0.15.6

A fix; nothing an ACP session sends or receives changes, so 0.15.5 clients keep working.

- `decision_code`'s `scope` starts the search at that folder. Before, the walk still began at the repository's
  root and judged every folder above the scope, so one low score on a broad card there ended the search
  after a single ask with nothing found (on audit-app, a question scoped to `services/main_service`). A scope
  that does not exist is refused, as one outside the working directory already was.
- A search where no folder scores high enough before any file is judged enters the best ones anyway, and
  says its places are weak; before, it ended with nothing found.
- A search with nothing to show says where its walk started, how far it got and the best folder it did not
  enter, instead of only "Nothing … looks related".

### 0.15.7

A fix; nothing an ACP session sends or receives changes, so 0.15.6 clients keep working.

- A session with memory on no longer fails at mount when the drive's working tree is off `main` or has
  edits. Before, a steward killed while it committed left the tree on `proposed/<session>`, and one shell
  write into a page left uncommitted changes; either failed every later session. A session reads
  `main`'s commit and only warns now.
- The steward builds its commit without touching the working tree, and a merging steward resets the tree
  to `main` and publishes or deletes a `proposed/` branch an earlier, killed steward left behind.

### 0.16.0

An addition; 0.15.7 clients keep working.

- `cap.workspace` ("Workspace"): a boolean capability option, on as `[capabilities.workspace] enabled` has it
  (off unless set). It gives the agent `index_files`, `decision_files`, `sql` and a document-aware `read` over a
  folder of PDFs, Word documents, workbooks, CSV and Parquet files, indexed in that folder's `.decision_workspace/`.
  Without the `[workspace]` extra it is listed with `unavailable` saying so.

### 0.17.0

An addition; 0.16.0 clients keep working.

- `cap.workspace` also gives the agent `sql_export`: it writes a `sql` result (a kept result, a scratch table
  or a SELECT) to a `.csv` or `.parquet` file in the working directory, never over an indexed document and
  never into `.decision_workspace/`, `.nace/` or `.git/`. Its calls come with kind `edit`.
- The workspace's `sql` reads tables of any size, keeps each SELECT's result as `res_N` (up to a million
  rows), and builds scratch tables. A write reaches only the session's scratch database, and `COPY`, `SET`,
  `ATTACH`, `INSTALL` and `PRAGMA` are refused. Each query is bounded in memory and time. `decision_files` and
  `DESCRIBE` show each table's column profile.

### 0.18.0

A rename; 0.17.0 clients keep working unless they match tool names.

- The search tools are renamed: `drex_code` is now `decision_code` and `drex_files` is now `decision_files`.
  Tool calls carry the new names. A client that matched the old names (to style or hide their calls) should
  match the new ones. Earlier entries in this list use the new names.
- The workspace index lives in `.decision_workspace/` (was `.drex_workspace/`). An existing index is moved
  there the first time its folder is opened, so nothing is rebuilt.
- System 1's descriptions in config options and `models --json` say "decision model" and "the console"; the
  `system1` values, `drex:<model>` references, `_meta.nace`, `_nace_capability` and every environment
  variable are unchanged. The bundled Qwen3.8 27B profile is named "Qwen3.8 27B (Fireworks)".

### 0.19.0

**Breaking**: the runner is one option, and its config is renamed. Clients that showed `cap.docker` or
`cap.executor.*` need a look; there is no alias.

- Where commands run is one [`runner`](#runner) select (category `_nace_runner`), listed in `models --json` as
  `runners`, and settable at open with `_meta.nace.runner`. The `cap.docker` and `cap.executor.<name>` toggles are
  gone from config options and from `capabilities`.
- A runner that cannot run here is refused when it is picked (`-32602` with the reason), not at the next prompt.
- config.toml: `[capabilities.executor] backend = "docker"` is now `[capabilities.runner] name = "docker"`, and
  `[capabilities.executor.options]` is `[capabilities.runner.options]`. The old table fails to load, with a
  message naming the new one.
- Plugins: the entry-point group is `nace_harness.runners` (was `nace_harness.executors`), and the base class is
  `nace_harness.runners.Runner` (was `nace_harness.executor_backends.ExecutorBackend`).

### 0.20.0

No change to the protocol. For Python hosts: `SecretResolver.resolve` is now `async`, and a resolver whose
`resolve` is a plain function is refused when the `SecretExecutor` is built, with a `TypeError` that says to wrap
blocking work in `asyncio.to_thread`. `[capabilities.secrets]` bindings can read a value from a command.

### 0.21.0

System 2 writes the State only for a compaction; 0.20.0 clients keep working.

- No State is written between compactions: `state_writing` and `state` events come only with a compaction
  (`trigger` `compaction`; `stale` and `first` no longer occur). `get_state`'s `state` and the State panel stay
  empty until a session's first compaction.
- System 1 reads a cut of the session built by code instead (the user's latest requests and the session in order,
  tool outputs as a note with their status, size, a check's tally and their first 300 characters, fitted to its
  window). An extension's `ctx.state()` returns it: a JSON object, `goal` and `history`, where it was text.
- The supervisor's `decision` events (purpose `supervise`) carry only the `mode` answer; `rewrite` and `worth` are
  gone.
- The supervisor's nudge (`nudge` event) has `reason` `stuck_gate`, was `stall_gate`; the rule that sends it is
  unchanged.

### 0.22.0

Additive; 0.21.0 clients keep working.

config.toml: a new top-level [`[secrets]`](secrets.md) table names keys kept in the OS keychain or a password
manager (`NAME = "keychain"` or `NAME = ["op", "read", "..."]`). Keys the harness reads itself (model, search, MCP)
stay out of the agent's commands. Keychain entries are read each session, command
entries once per process. One that can't be read doesn't stop the session: the client gets an agent message,
`Secret not loaded: NAME: <what to do>`, when the session opens (event `secret_unavailable`).
`nace-harness secrets set|list|test|remove [--json]` manages the table; a GUI can spawn it and pass a value with
`set --stdin`.
Stored values are replaced with `[REDACTED]` in every tool's output before it is recorded or sent, so
`tool_call_update` content, diffs and `RunEnd.detail` never carry one.

### 0.23.0

Additive; 0.22.0 clients keep working.

- A [capability toggle](#capability-toggles) switched on whose extra is missing (`browser`, `code-search`,
  `workspace`, `mcp`) gets it installed at the next prompt, then its setup (Chromium for the browser), as the
  terminal UI's `/caps` does. Progress comes as agent message text; `session/cancel` stops the install and keeps the
  change pending. Before, the prompt failed with the extra's install command.
- A toggle that cannot switch on here even then (web search with no provider or key) is no longer in a session's
  `configOptions` unless it is on. A client that still sends it gets the same `-32602` at its next prompt.
- `models --json`: a missing extra that switching on installs is no longer an `unavailable` reason.

### 0.24.0

0.23.0 clients keep working, unless they start the agent with `--extension requirements-check`.

- The `requirements-check` extension is gone. `--extension requirements-check`, or the name in a profile's
  `extensions`, now fails the session with `unknown extension 'requirements-check'`. No extension is packaged
  now; `--extension` still loads a `module:Class`.

### 0.24.1

Fix; 0.24.0 clients keep working.

- With the `process` capability on, the agent's `process poll` that waits ends as soon as the user sends a message
  (`session/prompt` while the turn is in progress), and the tool's description tells the agent to keep working
  while a job runs and to wait only when nothing else is left.
- A job the agent starts reads end-of-file on its input unless it asks for `stdin: true` (only those take `write`): a
  command that reads its input no longer hangs in a job.

### 0.25.0

The harness is now Intern. Nothing in the protocol changes; 0.24.0 clients keep working.

- The command is `intern`, the package to install `intern-agent` (`uv tool install "intern-agent[acp] @
  git+https://github.com/nace-ai/nace-harness@v0.25.0"`) and the import package `intern`. `initialize` returns
  `agentInfo {name: "intern", title: "Intern"}`; `--version` prints `intern 0.25.0`.
- Env vars are `INTERN_*`, the config home `~/.intern`, the per-project folder `.intern/` and the keychain
  service `intern`.
- `_meta.nace` and the `_nace_capability` / `_nace_runner` categories keep their names: Nace is the vendor
  namespace.
- Until 0.27.0 the old names still work. A `nace-harness` command runs `intern` (with a notice on stderr), so
  a client that starts `nace-harness acp` needs no change. A `NACE_HARNESS_*` variable is read when its
  `INTERN_*` one is unset, with a warning. On first run, the entries of `~/.nace-harness` move into `~/.intern`
  (beside the Intern app's own) and `~/.nace-harness` becomes a symlink to it, so session ids and paths saved
  before still load; an entry `~/.intern` already has is left in place, with a warning. A project's `.nace/`
  becomes `.intern/` and `.nace-resources/` becomes `.intern-resources/`. A keychain item stored under
  `nace-harness` is read from there and copied under `intern`.
- `uv tool upgrade nace-harness` cannot cross the rename: run `uv tool uninstall nace-harness` first, then
  install `intern-agent`. Both ship a `nace-harness` command, so uv refuses to install one beside the other.
- 0.27.0 removes the `nace-harness` command and the fallbacks, and raises the minimum version for clients to
  0.25.0. Clients should start `intern` before then.

### 0.27.0

The old `nace-harness` names are gone. **Minimum version for clients is now 0.25.0**: a client that starts
`nace-harness` or installs a harness older than 0.25.0 has to start `intern` and install `intern-agent`
0.25.0 or later. A client that already does keeps working.

- There is no `nace-harness` command any more, only `intern`.
- `NACE_HARNESS_*` variables (and `NACE_HARNESS_MODEL`) are not read: set the `INTERN_*` ones.
- `~/.nace-harness` no longer moves into `~/.intern`, a project's `.nace/` no longer becomes `.intern/`, and
  `.nace-resources/` no longer becomes `.intern-resources/`. Homes and projects that a harness from 0.25.0
  to 0.26.x has run in have moved already; move any others by hand. The `~/.nace-harness` symlink that the
  move left is kept, so session paths saved before the rename still resolve.
- A keychain item stored under service `nace-harness` is no longer read. Items that 0.25.0 to 0.26.x read
  were copied under `intern`; store any others again with `intern secrets set`.

### 0.27.1

The guard refuses reading your intern's own keys file, `~/.intern/.env`, with the read and grep tools or a
reading command (`cat ~/.intern/.env`), as it does `~/.ssh` and `~/.aws`: before, an agent asked to print it
printed your API keys. A project's `.env` can still be read; sending any `.env` stays refused. Nothing a client
sends or reads changed, so every client keeps working.

### 0.28.0

Additive; 0.27.0 clients keep working. Python hosts can disable automatic image attachments with
`ReadTool(attach_images=False)`, including through an extension's same-name tool replacement for CLI/RPC
sessions. Image paths then return a text notice without loading the bytes. Vision and other tools' images
are unaffected; the default still attaches images. The disabled reader's vision guidance uses the host's
viewing tools. Setup refuses an enabled Workspace capability that would override that fixed reader.
The setting does not remove images already in history; hosts with per-turn admission must keep filtering
or readmit those images. See [host-controlled image viewing](read-images.md).

### 0.28.1

The guard lets your intern install what its task needs from a standard package registry (PyPI, npm,
crates.io, apt, brew) or the maker's official download. Before, System 1 could refuse these as untrusted
code: `npm install -g typescript`, or the official Java runtime. Installs from an unknown index or host, or
from a stranger's repository, are still refused unless you ask for them.

When you tell your intern to follow a policy, an SOP or a runbook from your own organization, the guard
counts the records that document says to create or update, in the app you named, as asked for. Before, it
could refuse them, for example a HubSpot update a cleanup policy calls for. A "policy" from an outside sender
is not followed, and a step the document adds that your request does not need (deleting, overwriting data
with outside values, paying, refunding, or writing to people or places your request does not involve) is
still refused. Nothing a client sends or reads changed, so every client keeps working.

### 0.29.0

Additive; 0.28.0 clients keep working. System 1 has a second per-turn check: whether the request is done and
checked while the agent keeps changing what it did not ask for.

- A `decision` event with purpose `wrap_up` comes beside the supervisor's `supervise` one at each turn's end, until
  the wrap-up nudge has gone in. Its answer is `wrap`: a choice of `working`, `done_polishing`, `done_unsaved` or
  `unclear`, with probabilities.
- The nudge is a `nudge` event with `reason` `wrap_up`, at most once per run.

### 0.30.0

Additive for clients; 0.25.0 clients keep working. What the agent can call changed: `process` is in every session that
can run background jobs, and its `poll` no longer waits.

- The `process` tool is a base tool: every session whose runner runs background jobs (`local`, Docker, OpenSandbox)
  offers it, and a runner that runs none goes without it instead of refusing the session. `cap.process` is no longer
  listed in `configOptions` or `models --json`; a client that still sets it gets no error, and nothing changes.
- A `process` event reports each command that starts or ends in the session: a `bash` call, a job, or what a command
  left running (`cmd &`), with its command, state, exit code and times. It comes between runs too, is journaled, and
  is in the JSON-RPC and CLI `--json` streams. ACP sends nothing for it. Its `state` can be `moved` (a call that went
  on as a job), with `note` saying why.
- A State written for a compaction (`get_state`, the `state` event) has a `## Running processes` section before
  `## User requests` while jobs or what commands left running still run.
- `process` tool results carry the job's `command` and `elapsed_s`. `process start` returns the result of a job that
  ends within 3 s, else its handle.
- System 1's view of the session (an extension's `ctx.state()`) has a `processes` list while commands run.
- When a background job ends, or a process a command left running, the agent is told: a harness message (a `nudge`
  event, reason `process_ended`) at the run's next turn, with how it ended and the end of its output. A run does not
  end while such news is untold, so a reply that crossed a job's end gets one more turn. Between prompts an ACP session
  holds it, and tells it right after the next prompt; it starts no run of its own (the terminal UI's sessions do).
- System 1 looks at the commands running (at 30 s, 1, 2, 4, 8 and 16 minutes of each one's age, then every 30
  minutes): a `process_check` event (journaled; ACP sends nothing for it) with a verdict a command. A command it finds
  stalled, failing, or up and waited on is told to the agent as a harness message (`nudge`, reason `process_check`)
  at the run's next turn; between prompts, an ACP session holds it for the next prompt. A look System 1 cannot answer
  stops the run with `system1_unavailable`.
- A `bash` command goes on in the background as a job: as it starts when it is plainly a wait (a `sleep` of 10 s or
  more first, `tail -f`, `watch`, `gh run watch`); once it has run `[capabilities.process] move_after_s` (30 s by
  default); when System 1 finds it waits or serves, or finds it stuck; and at once when a message from the user comes
  in during it. Its `tool_call_update` completes with `[Running in the background as job <handle> (<why>). …]` and
  the output so far, and the agent is told when it ends. `decision` events with purpose `background` carry System 1's
  word on a command still running 2 s on.
- The agent's `process poll` no longer waits: `wait_s` is gone from its arguments (0.24.1's wait, which a user's
  message ended, is replaced). A turn that only polls (or lists) jobs that still run hands over, and the harness waits
  instead: an ACP prompt's turn, and its `session/prompt`, hold until the job ends, System 1 finds something, the user
  sends a message (`session/prompt` with the turn in progress), or System 1's next scheduled look at what runs (a
  `nudge` event, reason `process_look`, when it found nothing to tell); `session/cancel` stops it. A command System 1
  finds serves (a server, a watcher) is not waited for.
- `run_end` can have reason `waiting` (the agent handed over), in sessions that wake the agent when a job ends: the
  terminal UI, not ACP or JSON-RPC.

### 0.30.1

Fixes for background jobs; 0.30.0 clients keep working, and nothing a client sends or reads changed shape.

- A session reopened on its journal (`session/load`, a resumed terminal session, `/reload`) runs none of the jobs it
  ran before, since closing stopped them. The agent now hears of each one whose end it was not told: a harness message
  (`nudge`, reason `process_ended`) at the next prompt's first turn. Before, it was never told: it could wait for a job
  that was gone, or poll its handle and guess what had happened to it.
- Polling a handle from before a reopen says what became of its job. An unknown handle's error lists the session's
  jobs, so a mistyped handle is corrected in one call; it no longer says the handle came from another session.
- A `bash` command that moves to the background because it serves (System 1 judged it `serves`, or a later look found
  it up and `ready`) is no longer stopped at the call's `timeout_s`. It runs until it is stopped, the session closes,
  or 24 hours pass. Before, a server the agent reported as up was killed at the limit it had given a call it expected
  to block on, such as 60 s. Its `tool_call_update` says so. Any other moved command keeps its `timeout_s`, and its
  result now says when it will still be stopped.
- In the terminal UI, the nudge an agent gets before it hands over without a word (reason `process_waiting`) tells
  it, when the user asked how the job is going or whether it is stuck, to read the job's output or the file it writes
  to and answer from that. Before, it could answer "I can't tell" about a build whose output went to a log.
- The terminal UI keeps one line per command for System 1's routine looks, updated at each look, where it added a
  line per look. A finding the agent is told still gets its own line.
