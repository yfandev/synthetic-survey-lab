# Using Intern

The README gets you to a first run. This page is the rest: every way to start a session, the flags, the
model providers and how a working window is chosen, System 1's settings, the tool-call guard's switches and
`config.toml`. [Architecture](architecture.md) explains how the loop works; [capability settings](capability-settings.md)
lists the optional tools and services.

## Starting a session

`intern` on a terminal opens the [terminal UI](tui.md): the transcript, the State System 2 writes
and System 1 judges against, and System 1's decisions turn by turn, with `/system1` and `/system2` pickers.
`-p` runs a task headlessly instead; so does a pipe. With nothing configured it starts on the first of
the harness's known models your keys reach and says so; `/system2` changes it.

## Modes and flags

```bash
export FIREWORKS_API_KEY=... NACE_API_KEY=...
intern --system2 fireworks/accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b \
  "Find why tests/test_parser.py fails and fix it"      # the terminal UI, with this as the first prompt
intern -p --system2 fireworks/accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b \
  "Find why tests/test_parser.py fails and fix it"      # headless: the reply on stdout
```

| Mode | Command |
|---|---|
| Terminal UI, the default on a terminal: the State and System 1's decisions on screen, model pickers (see [tui.md](tui.md)) | `intern --system2 M ["task"]` |
| One task, readable output, headless | `intern -p --system2 M "task"` |
| One task, JSON events (one per line) | `intern -p --json --system2 M -- "task"` |
| JSON-RPC server over stdio | `intern rpc --system2 M` |
| Agent Client Protocol server (t3code, Zed; needs `[acp]`, see [acp.md](acp.md)) | `intern acp` |
| Event JSON Schema | `intern schema` |

Useful flags: `--window N` caps the working window, `--max-turns N` sets the turn budget,
`--no-session` keeps the journal in memory, `--session FILE` resumes one, `--no-context-files`
ignores `AGENTS.md`, `SYSTEM2.md` and `SYSTEM1.md` (instruction files are followed literally, with whatever credentials the shell environment
holds: an `AGENTS.md` that says "file an issue" and a tracker key in `env` get a filed issue), `--extension NAME` loads an extension (a `module:Class`;
no extensions are packaged yet), and `--no-extensions` drops the profile's extensions (each `--extension` still loads;
capabilities such as code search and memory are switched in `[capabilities]`, not by these flags).

Stopping it: Ctrl-C (SIGINT) aborts the run in `-p`. SIGTERM or SIGHUP, in `-p` and `rpc` as in the terminal UI and
`acp`, ends the run as an abort does: the command it is running is stopped with everything it started, `run_end` says
`aborted`, and the process exits with 128 + the signal (143 for SIGTERM). Only SIGKILL leaves commands running.

## Models

A model is `provider/model`. Everything after the first slash is sent to the provider as is.

| Provider | Base URL | Key |
|---|---|---|
| `openrouter` | openrouter.ai | `OPENROUTER_API_KEY` |
| `fireworks` | api.fireworks.ai | `FIREWORKS_API_KEY` |
| `together`, `deepinfra`, `groq` | their public endpoints | `<NAME>_API_KEY` |
| `openai` | api.openai.com, native Responses API (encrypted reasoning replayed) | `OPENAI_API_KEY` |
| `anthropic` | api.anthropic.com, native Messages API (signed thinking replayed, prompt caching) | `ANTHROPIC_API_KEY` |
| `ollama`, `llamacpp`, `lmstudio`, `vllm` | localhost defaults | none |

Every base URL can be overridden with `<NAME>_BASE_URL`, and `config.toml` can add providers.

**Local models.** On a Mac with Apple silicon, run System 2 on [Kapitsa](https://github.com/nace-ai/kapitsa),
the recommended local engine: it decodes Qwen and Nemotron faster than Ollama, llama.cpp and MLX on most Macs.
[local-models.md](local-models.md) goes from install to a first task:

```bash
brew install nace-ai/tap/kapitsa
kapitsa serve --model 35b --private                 # in its own terminal; 9b on a 16-24 GB Mac
printf '[providers.kapitsa]\nbase_url = "http://127.0.0.1:8000/v1"\n' >> ~/.intern/config.toml
intern --system2 kapitsa/qwen3.6-35b-a3b
```

Keys are read from the environment, and from `~/.intern/.env` (`KEY=value` a line) for those the
shell does not export: one file, every project, every entry point; an exported key wins over the file.
To keep a key out of plain-text files, put it in the [secrets store](secrets.md) instead: the OS keychain
or your password manager, named in `[secrets]`, and redacted from what the agent's commands print.

**Profiles** (`src/intern/profiles/*.toml`) hold per-model data: window, output cap, sampling,
thresholds, provider quirks and enabled extensions. The most specific `match` wins over
`default.toml`; your own profiles go in `~/.intern/profiles/`.

**Working window.** Each model runs at the smallest of its advertised window (profile, then the
provider's `/models`, then llama.cpp's `/props`, then 32k), its measured effective window, and
`--window` (an ACP session can set its own cap with `context_window`). All thresholds are fractions of the working window, so the same profile holds at 32k and
at 256k; one tool result is also held to the profile's `max_tool_result_tokens` (20,000), so a 1M window does not let
one output fill a sixth of it. The working window bounds the prompt; the reply may use whatever the model's advertised
window leaves (`output_reserve` of it, at most the profile's `max_output_tokens`, and at most the output
cap the provider's listing reports for the model, as Anthropic's and OpenRouter's do), so a small
`--window` does not starve a thinking model's reply and a 1M-window model is never asked for more output
than it allows.

**System 2 and System 1.** System 2 (`--system2`) is the agent model that reasons and calls tools.
System 1 (`--system1`) answers the harness's small typed questions (choice, score, noul) fast, and is
required: by default it is a decision model from the console (`drex-latest`, key in `$NACE_API_KEY`). A
session that cannot set it up, or whose System 1 does not answer when it opens, is refused with the
reason; there is no heuristic stand-in. An ask System 1 fails mid-run is retried with backoff for about
two minutes, then the run stops with `system1_unavailable`. `--system1 drex-max-v1.1` picks another
console model; `--system1 decisions:<model>[,<fallback>...]` a decision model on the decisions API
(OpenRouter: `~typesafe/jev-latest`, `nace-ai/drex-max`, ...; key in `$OPENROUTER_API_KEY`).
`intern models` lists every choice the console and the decisions API serve. The old names
`--model`/`-m`, `--decider`, `model =` and `[decider]` still work.

**The tool-call guard.** Before a tool runs, fixed rules refuse the clearly dangerous: deleting `/`, `~`,
a parent of the working directory or a `.git`, wiping a disk, piping a download into a shell, force-pushing,
discarding uncommitted work, reading or sending credentials and `.env` files, `sudo`, and writing into `.git/`,
credential or shell-startup files. These rules are code and need no model. System 1
then judges the other calls that can do harm against the session: what the user asked, what the agent did
and what came back. A blocked call
returns a refused result and the run continues. See [architecture](architecture.md#system-2-and-system-1).
On a disposable machine, such as a container the agent runs in as root, `[guard] system_writes = "allow"`
lets writes under `/etc`, `/usr` and the other system directories through to System 1's judgement instead
of refusing them; deleting a system directory stays refused.

## Configuration

Everything user-level lives in `$INTERN_HOME` (default `~/.intern`). An empty folder
gives a fully isolated run. `SYSTEM2.md` there replaces
System 2's packaged prompt (`SYSTEM.md`, its old name, is still read). `SYSTEM1.md` there is your guidance for System 1: what work is normal for you and what
you still want to be asked. Every System 1 ask about the session reads it, and the guard counts what it calls
normal as asked. It is read from there only, never from a project.

```toml
# ~/.intern/config.toml
system2 = "openrouter/qwen/qwen3.8-27b"

[system1]
backend = "console"        # the default: a decision model from the console, key in $NACE_API_KEY
model = "drex-latest"
max_state_tokens = 32768   # the default; longer states keep their head and tail, with a marker
# A decision model on the decisions API instead (OpenRouter, key in $OPENROUTER_API_KEY):
# backend = "decisions-api"
# model = "~typesafe/jev-latest"

[providers.office]
base_url = "http://gpu-box:8000/v1"

[commands]
memory_share = 0.5         # the default: a shell command holding more than half the machine's memory is stopped

[guard]
system_writes = "refuse"   # the default; "allow" in a disposable container leaves /etc, /usr writes to System 1
```

Optional tools and services share the saved `[capabilities]` settings. See
[capability settings and session lifetime](capability-settings.md) for search,
URL reading, background processes, browser, MCP, Docker, secret references and
model fallback/budgets. Existing defaults stay unchanged. Python clients should
use `async with session` or call `await session.aclose()` when finished; the CLI
and RPC server handle this automatically.

## Code search

With code search, a repository becomes a tree System 1 can search: `decision_code` returns the
places where a question's subject is defined or would be changed. It is a capability, off by default
and turned on only by `[capabilities.code_search] enabled = true` in `~/.intern/config.toml`
([capability settings](capability-settings.md#code-search)). It asks System 1, which every session has.

```bash
uv tool install "intern-agent[code-search] @ git+https://github.com/nace-ai/intern"
printf '[capabilities.code_search]\nenabled = true\n' >> ~/.intern/config.toml
intern --system2 M "Where is the retry policy decided?"
intern index                                  # the same index ahead of time, with a progress bar (optional)
```

The index is `.intern/index.json` in the working directory, built before the first turn when it is missing (seconds;
a progress bar on a terminal) in a git repository, and elsewhere (a home folder, Downloads) only when the agent
first searches, refreshed after edits, and kept out of git by a line in `.intern/.gitignore`. In a git
repository it holds the files git sees: what `.gitignore` keeps out of git is not indexed or searched.

## Workspace

With the workspace, a folder of office files becomes a tree System 1 can search: the agent indexes the
folder (`index_files`; PDFs, Word documents, workbooks and tables, with a progress bar on a terminal), then
`decision_files` returns the places a question needs, `sql` queries the tables and `read` shows documents as text
(`[ocr]` adds OCR of scanned pages). A capability like code search, off by default and turned on only by
`[capabilities.workspace] enabled = true` ([capability settings](capability-settings.md#workspace)).
It loads beside code search: in a repository with a folder of documents, the code index is `.intern/index.json`
at the root and the documents' index `.decision_workspace/` in their folder.

```bash
uv tool install "intern-agent[workspace,ocr] @ git+https://github.com/nace-ai/intern"
printf '[capabilities.workspace]\nenabled = true\n' >> ~/.intern/config.toml
intern --system2 M --system1 decisions:nace-ai/drex-max "Which quarter had the highest revenue?"
intern index --documents [folder]             # the same index ahead of time (OCR can take minutes)
```

## From a clone

```bash
git clone https://github.com/nace-ai/intern && cd intern
uv sync --all-extras
uv run intern --help
```

The checks to run before a push are in [CONTRIBUTING](../CONTRIBUTING.md).
