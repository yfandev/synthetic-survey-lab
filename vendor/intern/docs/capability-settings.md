# Saved capability settings and session lifetime

Configure the existing optional capabilities in `~/.intern/config.toml`.
`INTERN_HOME` selects another configuration home. `Bootstrap` reads this
same file for the CLI, JSON-RPC server and Python API; users do not need to
construct browser, process or MCP adapters themselves.

An unchanged configuration keeps the existing model profiles and default tools.
No browser, MCP connection, Docker container or background job starts on import.
This adds setup and ownership for existing components, not another tool registry.

## Selection and precedence

`[capabilities] tool_packs` is optional. When omitted, the selected model's
`profile.tool_packs` remains authoritative. An explicit list replaces that list;
`[]` selects no pack tools. Available packs remain `base`, `web-search`, `web-read`
and `process`. Include `base` when its file and shell tools are needed. `base`
includes the `process` tool, which the `process` pack still names for configs that
list it (see [managed processes](#managed-processes)).

Search, reader and process options configure their tools; adding an
options table does not enable a pack. Browser uses `browser.enabled`, code search
`code_search.enabled`, the workspace `workspace.enabled`, memory `memory.enabled` (on by default; see [memory](memory.md)), and each MCP entry explicitly selects a connection and its
remote tools. These services are independent of `tool_packs`: setting
`tool_packs = []` does not disable an explicit browser, code search or MCP entry. Docker, secrets and model budgets configure
execution/services rather than adding model-facing tools.

Saved `[capabilities]` settings apply unless Python supplies
`Options(capabilities=CapabilitySettings(...))`. That is a **whole-object
replacement**, not a partial merge with the saved capability tables. The rest of
the user configuration still loads and must be valid. For example,
`CapabilitySettings(tool_packs=["base"])` also leaves browser, MCP and budgets
at their inactive defaults.

The saved settings are each session's defaults. A session can switch the tool
packs, browser, code search, memory and each configured MCP connection on or off
for itself (the terminal UI with `/caps`, ACP clients with the `cap.*` config
options, see [ACP](acp.md#capability-toggles)), and pick its
[runner](capabilities/runners.md#picking-the-runner-per-session) (`/runner`, the
ACP `runner` option). The session reopens on its journal with the change; the
saved file is not edited.

Existing model selection precedence is unchanged: explicit `Options.system2`
(CLI `--system2`), saved `system2`, then `INTERN_SYSTEM2`. Model profiles still determine model compatibility,
generation settings and window thresholds. Capability settings do not introduce
automatic project-local service configuration or execute commands found in an
untrusted repository.

## What each setting is

A capability is a switch users flip during a session; `[capabilities]` holds those switches, the options of
one switch each, and settings for the whole session. Which is an extension and which a capability:
[CONTRIBUTING](../CONTRIBUTING.md#extensions-and-capabilities).

| Key | Kind | What it is | Default |
|---|---|---|---|
| `tool_packs` | switch setting | the `base`, `web-search` and `web-read` pack switches, as one list (`process`, now in `base`, is still accepted) | the profile's `tool_packs` |
| `browser` | switch setting | `cap.browser`: Playwright pages, and the browser's options | off |
| `code_search` | switch setting | `cap.code-search`: the code-search extension | off |
| `workspace` | switch setting | `cap.workspace`: the workspace extension | off |
| `memory` | switch setting | `cap.memory`: the memory extension, its drive and steward | on |
| `mcp` | switch setting | one `cap.mcp.<namespace>` switch per listed connection | none listed |
| `search` | capability option | options of the `web-search` pack | no backend |
| `reader` | capability option | options of the `web-read` pack | built-in limits |
| `process` | session setting | the session's job manager, behind the `process` tool in `base`: running jobs, records and log bytes kept, and when a long `bash` command goes on in the background (`move_after_s`) | built-in limits; 30 s |
| `runner` | session setting | where commands run; `/runner` and the ACP `runner` option pick it per session | `local` |
| `secrets` | session setting | secret references for commands (bindings, `inject`); the top-level [`[secrets]`](secrets.md) store is separate | none |
| `session_budget` | session setting | one model-call budget for the session | none |
| `system2_fallbacks` | session setting | System 2 models to fall back to | none |

Each value is overridden, lowest first, by:

- Tool packs: the profile's `tool_packs`, then `[capabilities] tool_packs` (replaces the whole list), then the
  session's own choices.
- Other switches: the default above, then `[capabilities]`, then `--memory`/`--no-memory` (memory only; an ACP
  client that sends its own capabilities cancels these flags), then the session's own choices.
- `Options(capabilities=...)` from Python replaces `[capabilities]` as a whole.
- ACP: a loaded session applies its journal's last switches before the client's picks. That replay, and an
  unbound session, start pack switches from `["base"]`, not the session's profile.

## Small saved configuration

Replace model references and service addresses with your configured resources.
This example enables URL reading alongside the ordinary tools:

```toml
system2 = "vllm/my-model"

[capabilities]
tool_packs = ["base", "web-read"]

[capabilities.reader]
timeout_s = 30
max_bytes = 4000000
max_redirects = 5
allow_private = false
```

The following sections show independent fragments to add to your existing file.
Merge each table once; do not copy duplicate `[capabilities]` headers into TOML.
Export the referenced environment variables before starting the application.
Credentials are environment references, not literal values in the settings.

## Web search

Select `web-search` in the effective tool-pack list, then configure its backend:

```toml
[capabilities.search]
backend = "brave"
api_key_env = "BRAVE_SEARCH_API_KEY"
timeout_s = 20
max_attempts = 2
```

Supported backends are `brave`, `tavily`, `exa` and `searxng`. When backend/key
settings are omitted, the existing `INTERN_SEARCH_PROVIDER` and provider
key variables continue to work. Default key names are `BRAVE_SEARCH_API_KEY`,
`TAVILY_API_KEY` and `EXA_API_KEY`. An explicit backend overrides the legacy
provider variable; `api_key_env` selects another source variable for that backend.
Bootstrap captures its environment once rather than consulting a changing global
environment on every tool call.

For SearXNG, use `backend = "searxng"` with `searxng_url`, or retain `SEARXNG_URL`.
The current SearXNG adapter does not accept `api_key_env`. See
[search behavior and limits](capabilities/web-search.md).

## Retained URL text

The `web-read` pack uses the existing reader, including redirect, address,
compression and download-size checks. An optional base folder controls retained
full-text files:

```toml
[capabilities.reader]
storage_dir = "retained/web"
```

With local execution, the default base is `<config-home>/session-resources/web`.
With Docker, it is `<runner.workspace>/.intern-resources/web`, inside the existing
workspace mount. Explicit Docker reader storage must also resolve inside that
workspace; an outside path fails setup instead of adding another host mount.
Setup adds a separate private directory for each constructed Session, including
a new Session resumed from an existing journal. Closing retains these files.

For Docker, a truncated result advertises the translated
`/workspace/<relative-store>/<session-scope>/<digest>.txt` continuation path, so
the configured `read` tool can recover the full source with offset/limit. The
resource reference keeps the actual host path for the calling application.
See [reader details](capabilities/web-reader.md).

## Managed processes

Every session has one job manager wherever its commands can run in the background:
the `local` runner, Docker, or any runner that runs background jobs. The `process`
tool, a base tool, starts jobs in it, and it keeps a record of every command running
in the session (a `bash` call while it runs, what a command leaves running with
`cmd &`), reported as `process` events and named in the State written at a
compaction ([processes](capabilities/processes.md)). A runner that runs no background
jobs has no job manager, and the session goes without the `process` tool.

```toml
[capabilities.process]
max_running = 4
max_records = 64
max_log_bytes = 65536
move_after_s = 30   # a bash command still running after this long goes on in the background; 0 never
```

The Session owns this manager and terminates its owned jobs on close. Jobs also
have their existing per-job deadlines. Jobs run where the session's commands
run: on this machine with the `local` runner, in a container with Docker, or on
any runner that runs background jobs. With scoped command secrets,
a job gets the same scrubbed environment and injected values as a command, and
its log is redacted as it arrives, so it never holds a raw value. Process handles do not survive restart. See [process limits](capabilities/processes.md).

## MCP

Install the optional dependency with `uv sync --extra mcp`. To pick a server and
sign-in method for Gmail, Drive, Slack, GitHub, Linear or Notion, start with
[connecting outside services](capabilities/connections.md). Entries select exact
remote names and explicitly declare their effects:

```toml
[[capabilities.mcp]]
namespace = "docs"
transport = "http"
url = "https://docs.example.test/mcp"
timeout_s = 30
header_env = { Authorization = "DOCS_AUTHORIZATION" }
remote_tools = { search = ["read", "network"] }
```

`DOCS_AUTHORIZATION` contains the complete header value, including a `Bearer `
prefix when required. URLs cannot contain embedded credentials or a fragment.
The HTTP client does not use ambient proxy settings or follow redirects.

For a token that expires, name a command that prints the header value instead.
It runs once at setup and again after a 401:

```toml
[capabilities.mcp.credential_command]
command = "/usr/local/bin/docs-token"
env_refs = { CLIENT_SECRET = "DOCS_CLIENT_SECRET" }
header = "Authorization"
```

The command must be an absolute path. Its header cannot also be in `header_env`.
See [credential command](capabilities/mcp.md#credential-command) for the
contract, how it runs and its limits.

For a hosted server that signs you in with OAuth, add an empty
`[capabilities.mcp.oauth]` table to the entry and run
`intern mcp login <namespace>` once. Sessions then use and refresh the
saved tokens. An entry uses either OAuth or a credential command, not both, and
OAuth sets the `Authorization` header itself. See
[OAuth sign-in](capabilities/mcp.md#oauth-sign-in).

A local server uses `command`/`args` and explicit environment references instead:

```toml
[[capabilities.mcp]]
namespace = "local_docs"
transport = "stdio"
command = "/absolute/path/to/docs-mcp"
args = []
cwd = "."
env_refs = { API_TOKEN = "DOCS_API_TOKEN" }
remote_tools = { lookup = ["read"], echo = [] }
```

Commands and arguments are literal executable/argument values, not shell command
strings or credential interpolation. Allowed effects are `read`, `write`, `exec`
and `network`; an explicitly pure tool may declare `[]`. The mapping must contain
at least one exact remote tool name. Server annotations never grant permission.
Namespaces and exported names must be unique. Missing selected tools, invalid
schemas or connection failures fail setup with rollback. MCP retains its current
bounded text-result adapter; see [MCP support](capabilities/mcp.md).

## Browser

Prepare browser support explicitly:

```sh
uv sync --extra mcp --extra browser
uv run --extra browser playwright install chromium
```

Use only the extras you need. Then enable the existing five-action browser:

```toml
[capabilities.browser]
enabled = true
artifact_dir = "retained/browser"
allow_private_urls = false
timeout_s = 30
```

Setup owns a lazy BrowserSession; Chromium launches on its first action, not when
the settings are loaded. Setup checks the Python extra, but a missing Chromium
binary is reported on first use. In the terminal UI, `/caps browser` does both
steps when it switches the browser on: it installs the extra if it is missing and
runs `playwright install chromium`, which downloads Chromium once and does nothing
after ([docs/tui.md](tui.md)). An ACP client's `cap.browser` does the same at its
next prompt ([docs/acp.md](acp.md#capability-toggles)). The default screenshot base is
`<config-home>/session-resources/browser`; setup adds a private per-Session
directory. Screenshots remain after close. Browser network policy is independent
of the executor; private destinations stay disabled unless the operator opts in.
Aborting an in-flight browser action closes that browser permanently for the
remainder of the Session. Create a new Session to use the browser again; abort
does not undo an action that already reached a remote site.
See [browser scope and limits](capabilities/browser.md).

## Code search

Install the optional dependency (`uv sync --extra code-search`, or
`intern-agent[code-search]`), then turn on `decision_code`:

```toml
[capabilities.code_search]
enabled = true
```

This setting is the only way to turn code search on. `--extension code-search`, a
profile's `extensions` list and `module:Class` are refused with a pointer to it;
`--no-extensions` does not switch it off. Setup checks the extra and fails before
the run when numpy, scipy or networkx is missing; in the terminal UI, `/caps code-search`
installs it first.

`decision_code` asks System 1, which every session has (a decision model from the console, or
what `[system1]` or `--system1` picks). An ask System 1 does not answer, after its
retries, stops the run rather than reading as nothing found. The index is `.intern/index.json` in the working
directory: built before the first turn when it is missing or stale in a git work tree (elsewhere, such as a
home folder, only on `decision_code`'s first call: nothing is walked or written before), refreshed after
`write`, `edit` or `bash`, and never built with a model call. In a git work tree it
holds the files git sees (tracked, or untracked and not ignored), so what `.gitignore`
keeps out of git is not searched; one index holds at most 100,000 files. A working directory that holds
more documents (PDFs, Word, spreadsheets, Parquet: files the index cannot read) than code and text, counted
over its first 2,000 files in or outside git (images count as neither), is not indexed: the pinned note says
so in one line, and `decision_code` on such a folder, or on a document, says it cannot read it. `decision_code`'s
`scope` may name a folder outside the working directory, such as another repository the
agent was asked about: the git work tree holding it (or the folder itself, outside git)
is indexed the first time, into that repository's own `.intern/index.json`, and kept and
refreshed like the working directory's, so the next search there, or a run started
there, reuses it. A scope starts the walk at that folder: the folders above it are not
judged, so a broad card higher up cannot end the search before anything in scope is
seen; a path that does not exist is refused. When no folder or group scores high enough
before any file is judged, the best ones are entered anyway and the result says its
places are weak; a search with nothing to show says where its walk started, how far it
got and the best folder it did not enter. The question's words also steer the walk: an
SQLite FTS5 table of every chunk (`.intern/lexical.sqlite`, identifiers split, brought up
to date before a search, no model) gives the passages they match, and the groups
holding the best of them have their files judged first, beside the walk; what comes
back is still only what System 1 judged. `intern index`
builds the same index ahead of time. Code search owns no service that outlives a
run, so Session close has nothing extra to stop. In the terminal UI each `decision_code`
call draws its search live right under the call ([docs/tui.md](tui.md#layout)).

## Workspace

Install the optional dependency (`uv sync --extra workspace`, or `intern-agent[workspace]`;
`[ocr]` adds OCR of scanned pages), then turn on the document tools:

```toml
[capabilities.workspace]
enabled = true
```

This setting is the only way to turn the workspace on; `--extension workspace`, a profile's
`extensions` list and `module:Class` are refused with a pointer to it, and `--no-extensions` does
not switch it off. Setup checks the extra and fails before the run when it is missing; in the
terminal UI, `/caps workspace` installs it first. The agent
indexes the folder that holds the documents with `index_files` (PDFs, Word documents, workbooks,
CSV and Parquet; OCR can take minutes, so this is the agent's call, not run start), into that
folder's `.decision_workspace/` (text under `derived/`, every table in `tables.duckdb`); a working
directory that already has one is loaded at the start. Without one, the pinned note names the folders of the
working directory that hold documents (PDFs, Word, workbooks, Parquet; counted over its first 2,000 files, in or
outside git), each with its count and whether it is indexed; a working directory with none, such as a code
repository, gets no note. Then `decision_files` searches it with System 1 (the
tree walk and a BM25 first stage over its passages, judged together, then one SAG2 expansion round),
`sql` queries the tables, and `read` shows documents as text (`pages=`, `tab=`). It loads beside code
search: in a repository with a folder of documents, the code index is `.intern/index.json` at the
root and the documents' index `.decision_workspace/` in their folder. `intern index --documents
[folder]` builds it ahead of time.

## Runner (where commands run)

`[capabilities.runner] name` picks where commands run: `local` (default),
`docker`, `opensandbox`, or a runner a package registers. Each runner validates
its own `options`. See [runners](capabilities/runners.md) for the list and for
writing one. Docker's options:

```toml
[capabilities.runner]
name = "docker"
workspace = "."

[capabilities.runner.options]
image = "python:3.12.11-slim-bookworm"
network = "none"
memory_mb = 512
cpus = 1
pids_limit = 128
```

The workspace must exist and contain the Session working directory. Setup pairs
Docker with the matching selected file/shell tools, so those operations use
`/workspace` rather than silently accessing host files. Prepare the Docker daemon
and image yourself; no automatic pull/probe occurs during setup. Each command
owns a fresh container and removes it in its execution cleanup path.

The image must provide the commands used by your task; the matching file tools
need Python 3. Docker's network restriction covers its containers, not the
separate host search/reader/browser/MCP services. The `process` tool runs each
job in a container of its own, with the same limits. See [Docker trust and workspace limits](capabilities/docker-workspace.md).
The tool-call guard treats `/workspace` and the host working directory as the same place, and its
`git status` check runs through the container, so the image needs `git` for that check to apply.

## Scoped command secrets

```toml
[capabilities.secrets]
scope = "deployment"
inject = { API_TOKEN = "deploy" }

[capabilities.secrets.bindings.deploy]
source_env = "DEPLOY_TOKEN"
scopes = ["deployment"]
```

The left side of `inject` is the command's environment variable; its value names
a declared reference. Only that reference's authorized scope can resolve it.

A binding can get its value from a command instead of a variable, so a secret
kept in 1Password, the macOS keychain, `pass`, Vault or AWS SSM never has to be
exported into the shell that starts the harness:

```toml
[capabilities.secrets.bindings.github]
command = "/opt/homebrew/bin/op"                  # absolute path, no shell
args = ["read", "op://Private/GitHub/token"]
env_refs = { OP_SERVICE_ACCOUNT_TOKEN = "OP_SERVICE_ACCOUNT_TOKEN" }
scopes = ["executor"]
```

Other commands that print a secret: `security find-generic-password -w -s NAME`,
`pass show NAME`, `vault kv get -field=token secret/NAME`,
`aws ssm get-parameter --with-decryption --query Parameter.Value --output text --name NAME`.
A binding takes exactly one of `source_env` and `command`. The command runs the
way an MCP credential command does (own process group, closed stdin, bounded
output, `timeout_s`, 30 by default) with only the default variables plus its
`env_refs`, once at setup; a failure stops setup with its exit status and the
redacted end of its stderr. Its output, less one trailing newline, is the value,
so a multi-line key works. The value is kept for the session. A prompt such as
1Password's desktop unlock only appears when the harness runs in your desktop
session; use a service account token through `env_refs` elsewhere.
Required selected values are checked before opening MCP resources, then injected
by the existing SecretExecutor at execution. Optional `allowed_env` replaces its
ordinary base-environment allowlist; injected names cannot also be overridable.

This wraps the runner's executor (local or Docker); `process` jobs run through it too.
Authorized command code can read or transmit its injected values: literal output
redaction is not an exfiltration sandbox. No model-facing secret lookup tool is
added. See [secret boundaries](capabilities/secrets.md).

## Model fallback and a session budget

```toml
[capabilities]
system2_fallbacks = ["vllm/backup"]

[capabilities.session_budget]
max_calls = 100
max_observed_cost_usd = 0.25
deadline_after_s = 300
```

Fallback targets are explicit; at most seven distinct backups are allowed. Their
advertised and working context windows must cover the primary's, and sampling,
reasoning effort/replay, compatibility flags and vision settings must match.
Setup refuses incompatible profiles instead of guessing request conversions.
Fallback buffers a complete response, so it trades live token streaming for a
boundary that does not concatenate failed partial replies with a backup answer.

Native System 1 model fallback remains under the existing `[system1]` settings:

```toml
[system1]
backend = "decisions-api"
model = "your-primary-decision-model"
fallback_models = ["your-backup-decision-model"]
api_key_env = "DECISION_API_KEY"
```

The shared budget wraps that existing backend inside DecisionClient, preserving
its cache, retries and native model-list fallback. System 1
internal retries/batches count as one logical backend dispatch; System 2 fallback
targets each consume their own reservation. System 1 does not report token usage,
so these saved budget settings do not offer a shared output-token cap.

The ledger is **session cumulative**, across prompts, follow-ups and model
switches. An idle `Session.set_model` keeps the same ledger and revalidates backup
compatibility; a failed switch leaves the old binding in place. If a configured
backup becomes primary, it is not also its own backup.

`deadline_after_s` starts at setup and bounds model calls, not every tool or human
waiting period. Observed cost can overshoot and failed requests may have unknown
usage; it is not a hard provider billing cap. This ledger is in memory: creating
a new Session, even from the same journal, starts a new ledger. See
[provider and budget guarantees](provider-failover.md).

## Python lifetime and paths

```python
import asyncio
from pathlib import Path

from intern.config import Bootstrap, Options


async def main():
    bootstrap = Bootstrap(Options(cwd=Path.cwd()))
    async with await bootstrap.session(lambda event: None) as session:
        result = await session.run("Inspect the configured workspace")
        print(result.final_text)
        if session.model_budget is not None:
            print((await session.model_budget.snapshot()).calls)


asyncio.run(main())
```

The CLI and RPC server use the same setup and Session close path. Python hosts
should use the async context, or call `await session.aclose()` after active work.
Close permanently rejects new work, requests cancellation before resource
teardown, and can be joined again if the caller was canceled. Close from inside
the Session's active run is refused to avoid waiting on itself.

MCP context entry and exit stay on one resource-owner task. Startup failure rolls
back resources already owned by that task. Cleanup errors/timeouts are reported
as incomplete cleanup, not silently reported as success. A close deadline is not
a guarantee that arbitrary code which ignores cancellation has stopped. Process
kill or host loss still needs external supervision; there is no new crash
recovery store or browser orphan-sweep service.

Reader/browser base folders, Docker workspace and MCP `cwd` expand `~`; relative
paths resolve from `Options.cwd` (the Session working directory), not from the
TOML file's parent. Prefer an absolute `INTERN_HOME`. Retained files use
owner-private POSIX permissions; those permissions do not isolate other programs
running as the same user.

Settings reject unknown capability fields and incompatible transport/runner
options. Pack-created tool instances are tracked through extension merging;
an extension's configured stock-class tool is preserved too. Nondefault search/reader/process
options targeting a replacement fail clearly, and Docker refuses to replace
custom file/shell tools. Final tool names must not collide. These checks do not
grant extra authority to untrusted tools or page content.

Automatic ownership covers services created by these capability settings and
pack adapters. Extensions/applications that supply their own service-backed tool
instances keep responsibility for those services' lifetime; setup does not take
over a possibly shared provider or process manager.

## Upstream rationale

Hermes keeps saved configuration and tool selection separate from runtime tool
availability; its cleanup follows the owner of each service. See the pinned
[configuration](https://github.com/NousResearch/hermes-agent/blob/b52b8b00f0c2af6b6d6ede68d9071be8cf2d1b5a/website/docs/user-guide/configuration.md)
and [CLI lifecycle source](https://github.com/NousResearch/hermes-agent/blob/b52b8b00f0c2af6b6d6ede68d9071be8cf2d1b5a/hermes_cli/cli_shutdown.py).

Pi loads saved settings through its host resource loader and documents lazy
resource startup with idempotent session shutdown. See its pinned
[configuration](https://github.com/earendil-works/pi/blob/085c84e98e7f4c950f2be606466fff4f75d8fe7c/packages/coding-agent/docs/configuration.md)
and [extension lifecycle](https://github.com/earendil-works/pi/blob/085c84e98e7f4c950f2be606466fff4f75d8fe7c/packages/coding-agent/docs/extensions.md).
Current Pi's MCP support is a replaceable built-in extension, not evidence that
all capabilities belong in its core or are enabled by default.

The design choice is to reuse TOML, Bootstrap, existing components and an
explicit Session owner. Upstream implementation patterns do not establish task
quality gains or authorize changing defaults; those remain separate decisions.
