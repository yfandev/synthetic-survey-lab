# Architecture

Intern is a headless engine. Every client, the CLI included, drives it through one typed
protocol; nothing reaches into the loop.

```
 CLI (-p --json)   JSON-RPC (stdio)   ACP (stdio)   TUI (tui)   later: IDE, desktop
        \                |                 /
         +---------- Session (session.py) --------+
                         |
                    Run (loop.py)  -- hooks.py --> extensions
       +---------+-------+-------+-----------+------------+
       |         |               |           |            |
   context.py  providers/     tools/ +   decisions.py  journal.py
   (segments,  (Provider,     executor   (choice/score/ (append-only
   compaction) openai_compat) (local)    noul backends)  JSONL tree)
       |         |
   models/ (registry, profiles, working window)    instructions.py (SYSTEM2/1.md, AGENTS.md)
```

## System 2 and System 1

Two kinds of model run a task. **System 2** is the agent model: it reads, reasons and calls tools,
turn by turn. **System 1** answers the harness's small typed questions (choice, score, noul) fast
and cheap, through `decisions.py`. It is required: nothing answers in its place. Both make decisions; the
split is by speed and by the shape of the question, not by who decides.

Each sees the run its own way. System 2 writes the **State**: a brief for further execution
(a plan of one line a step with its status, done-criteria, what is in progress and the next action, the
artifacts the task produced or changed — files, branches, commits, PRs, issues, URLs, each with what it is
for, not files only read or what a file holds — and facts, what each step found included). It is
written for each compaction, over the turns the State before it does not cover, and at no other time. The first
State is
written whole and later ones are edits, for every model (measured on Opus, GPT and Qwen3.8 27B):
the writer sees the State's items numbered within their sections and replies
with edits (`set 3: …`, `add: …`, `drop: 2`, `replace all: …` under a section's heading), which code applies
(`state_edits.py`), so a write emits what changed rather than the whole State again, and what it does not
name stays as it was. A new request from the user is brought in with edits too. Each edit request gives the
State's size in words and asks to keep it under 1,500. An edit that grows it past that gets one more reply
in the same fork: edits that drop or shorten items, cutting it to about 1,350. Should that reply fail or cut
nothing, the edit is kept as it is (rewriting the State whole for size cost 18 of 134 Opus edit requests a
second reply of about 3,800 tokens; a trim is about 400). Edits that cannot be applied (an item that
is not there, conflicting edits of one section) or a failed edit call cost one more reply in the same fork
with the whole State. A reply with no section of
a State in it (a writer that answered the task instead) is a failed write: the previous State stands.
No message of the user's is pinned: the first is a turn of the history like any later one, and compaction
cuts it with the turns it opened. What they asked is kept in the State instead: its last section, `## User
requests`, holds their last three messages as they wrote them, oldest first (never the harness's nudges),
each a block quote a blank line apart, so a list or a heading in one cannot read as the next one or as a
section; the order is said once rather than as labels such as "latest", which a newer message after the
State would make wrong. It is kept within `thresholds.requests` of the window, the newest always and cut to
fit if it must be. Code sets it at every write (and in the fallback when no State could be written); the writer never sees it among the items it
edits, and a copy it writes is replaced. Codex's compaction keeps the latest user messages verbatim the same
way; Hermes stopped pinning the first ones after a compaction because they grew without end.
Before it, while jobs or what commands left running still run, code also sets `## Running processes`: each job's
handle and command and how long it has run, since the turn that started a job may be among those the State replaces
and the agent would lose the handle. The writer never sees it either.
System 2's own session writes it: the write is the request the agent would send next — its system prompt,
tools and context, with the reasoning it replays and the tool results it saw — plus one message asking for
the State, so the writer knows all the agent knows; the agent never sees the write. The State holds no
number the run did not show: one that appears nowhere in the write's request (a miscopy, or the writer's
own sum) is named back to the writer in the same fork, which fixes it with line edits on the State (an edit
may change only a line that holds such a number, so the fix is no rewrite), and any still unseen is blanked
to `?`. The compaction waits for the write: an abort stops it and nothing is cut. When the standing State
already covers the turns compaction would cut (`/compact` right after a compaction), the cut needs no model call
(the recent turns stay raw as always). `Session.compact()` makes the same compaction between runs, at any size:
the TUI's `/compact`. Until 0.21 System 1 also had the State written between compactions, when it found the work
worth a first State or the State out of date: about one write every seven turns, 399 writes against 4
compactions over 84 local sessions (Opus: median 17.9 s and 1,276 output tokens a write). Only System 1 read
those States, and it now reads the session itself.

System 1 reads a cut of the session instead, built by code from the journal whenever it is asked
(`session_cut.py`; `ctx.state()` for extensions): the input fast-jev-compaction gives its pruning model
(https://github.com/tamaratran/fast-jev-compaction, our own port). It is a JSON object with a `goal` (the user's
last three messages, 500 characters each) and the whole `history` in order: the user's messages, the harness's
nudges, the agent's replies without their reasoning, and each tool call with its input (up to 1,000 characters)
and a note for its result. The note gives the result's status and size, a check's tally (`15 passed in 2.1s`,
a linter's count) and the output's first 300 characters, where fast-jev gives status and size only. While commands
run, `processes` lists them beside the goal, from the session's job manager rather than the journal: each tool call
still running, background job and process a command left running, with how long it has run and, for a job, how long
it has written nothing and its last line (at most 8, oldest first). Like the goal it is never cut to fit. It is
fitted to three quarters of System 1's `max_state_tokens` (24,576 at the default), estimated as fast-jev does,
in fast-jev's stages, each only when the one before is not enough: inputs cut to 200 then 60 characters, long
texts cut to head and tail oldest first, old messages collapsed to a note of their size, old calls to one line,
old messages without calls left out, runs of old call-only messages folded. The first message and the newest
six are never collapsed or left out; past the last stage the oldest of the rest are left out behind a note
(1 of 82 local sessions at its end). It costs no model call and at most 0.1 s to build. Asked 8 questions at
each of 40 points of 22 sessions (Opus 5.5 writes them from the transcript), System
1 answers 95% from it (two runs), 94% from System 2's State and the turns since, 94% from a transcript of the
same length with the middle cut and every output's first 300 characters, and 84–85% from the cut with notes of
status and size only, which loses checks' and errors' results (50% and 47–53% of those questions).
System 2 sees the State once compaction has cut the turns it covers; until then it has the turns.
Evidence is preserved at two points (`evidence.py`). A big test/build output arrives as a receipt of its
verbatim evidence (fatal signals, failures, failing targets, the summary, the exit status) with the full
output archived for readback, extracted by rule with the checks of SoL-Pi's reducer. At compaction, when a
System 1 backend is configured, System 1 judges every file read about to be cut against the session (its cut) and the
reads the current step works on stay verbatim after it (`CompactionEntry.kept`), judged again at every
compaction — prose facts cannot stand in for code the model reasons over, and at small windows the model
otherwise re-reads the cut files.

System 1 supervises the run (`supervise.py`): after every turn, one call over its cut of the session asks what
the latest turns are doing compared with what the session established before them (advancing, re-investigating
what earlier turns already covered, retrying, off plan, finished, or unclear when the turns fit no single mode).
Three turns in a row that System 1 calls stuck (P(re-investigating) + P(retrying) at 0.8 or more) nudge System 2
to act on what it found, and the count starts over (replayed over 82 stored sessions, 16 of its 19 nudges
fell on turns a blind judge called stuck). The loop waits for the answer (0.6 s at the median
cut, 0.85 s at its full size, against 0.5 s at main's 18k characters) and acts on it at the same turn's end.

A second ask at the same time (`wrap_up`) asks whether the request is done and checked while the latest turns change
or re-check what it did not ask for. Three such turns in a row at 0.8 or more send one wrap-up nudge per run. It never
asks System 2 to stop, since the same verdict comes on work a grader fails. Instead it asks System 2 to make any
further change on a copy and replace the original only once its check passes, so a deliverable that works keeps
working whenever the run ends (an agent that had its tests green at a third of its time limit broke the
file by polishing it and was killed with it broken).

Turns end; a command may not. System 1 also watches what runs (`process_watch.py`): while any command runs,
in a run or between runs, it is asked about each one at its ages of 30 s, 1, 2, 4, 8 and 16 minutes, then every 30
minutes, one ask over its cut (which lists them, `processes`) covering every command running. Each gets one choice:
progressing, waiting as expected, stalled, failing, up and waited on as if it would end (`ready`), or unclear. Those
at P(stalled) + P(failing) + P(ready) of 0.6 or more are told to the agent as one harness message at the next turn
(the latest look only), or between runs as the session's `job_notices` says; no counter, since each look comes later
than the one before. Every look is a `process_check` event, which the terminal UI shows as it happens: the "are you
stuck?" a user would otherwise ask. A ten-hour command costs about 25 asks. It stops nothing; a `bash` command it
finds stuck goes on in the background, and so does one System 1 finds waits or serves, asked once it has run 2 s, or
one still running at `move_after_s` (30 s), so a long command never holds the agent's turn.

Every tool call passes the guard (`guard.py`) after the extensions' `tool_call` hooks, so it judges the
arguments the tool will get, and before the tool runs. A blocked call returns a refused result that names what
was blocked and why, with the next action "find a safer way, or stop and say so"; the run goes on. Three
refusals in a row, or twenty in a run, stop it (`stopped_by_guard`), as in Claude Code's auto mode. The guard
cannot undo a hook's refusal: a refused call never reaches it. It has two layers.

Fixed rules are code. They apply whether or not System 1 is configured, and to read-only tools too. Each rule
names an action that is destructive or exfiltrating in ordinary development and never fires on normal work
(`rm -rf build/`, `git push origin my-branch`, `pip install`, `sed -i`, writing files in the working directory):

| Rule | Refuses | From |
|---|---|---|
| `delete_protected` | recursive `rm`, `find -delete`/`-exec rm` or `mv` of `/`, the home directory, a parent of the working directory, a top-level or system directory, a folder in the home directory, a `.git`, a credential folder | Codex (forced `rm`), Arcjet destructive-command, dcg core.filesystem, Claude Code (irreversible local destruction) |
| `permissions_protected` | `chmod`/`chown -R` of the same system paths (`chmod -R 777 /`) | dcg, Claude Code (security weaken) |
| `disk_wipe` | `mkfs`, `dd of=/dev/…`, `wipefs`, partition editors, `diskutil erase…`, writes to a block device | Arcjet destructive-command, dcg system.disk |
| `fork_bomb` | `:(){ :\|:& };:` and its renamings | dcg, pi-prevent-destructive-commands |
| `pipe_to_shell` | a download piped into a shell or interpreter, `bash <(curl …)`, `sh -c "$(curl …)"` | Arcjet piped-installer, Claude Code (code from external) |
| `force_push` | `git push --force`/`-f`/`+refspec`/`--mirror` (`--force-with-lease` goes to System 1) | Arcjet rewrite-history, dcg core.git, git-guardrails, Claude Code (git destructive) |
| `discard_changes` | `git reset --hard`, `git checkout .`/`-f`, `git restore .`, `git switch -f`, `git clean -f`, only when `git status` (or `git clean -n`) through the executor shows work they would destroy | Claude Code (checks `git status`), dcg core.git |
| `credential_read` | reading `~/.ssh` private keys, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.docker`, gcloud/gh/azure config, `.netrc`, `.git-credentials`, `.npmrc`, `.pypirc`, the harness's own keys file `~/.intern/.env`, `/etc/shadow`, keychain dumps, by the read and grep tools or a reading command | Arcjet credential-access, cc-safety-net, Claude Code (credential exploration) |
| `secret_send` | a network command sending a credential file, a `.env` or the environment (`curl -F f=@.env`, `cat .env \| curl`, `env \| nc`) | Claude Code HARD BLOCK (data exfiltration), Arcjet egress |
| `privilege` | `sudo`, `doas`, `su`, `pkexec` | Arcjet destructive-command, Codex (sudo unwrapping) |
| `protected_write` | writes by the write/edit tools, redirections, `tee`, `cp`/`mv`/`ln`, `sed -i` into `.git/` internals, credential folders, shell startup files, `~/.gitconfig`, LaunchAgents or system directories | Codex (`.git` read-only in workspace-write), Arcjet protected-paths, Claude Code (unauthorized persistence) |
| `kills_harness` | `pkill`/`killall` whose pattern matches the harness or a parent of it, only when `pgrep` with the same matching options (through the local executor) lists one of them: `pkill -f <a word of the task>` when the task is on the harness's command line (`-p`), `killall python` | |

Shell commands are tokenized, not run through a shell: quotes, escapes, comments, `;`/`&&`/`||`/`|` chains,
groups, redirections, heredocs, `$( )`/backtick/`<( )` substitutions, `bash -c`/`eval` strings (up to eight
levels) and wrappers (`sudo`, `env`, `timeout`, `xargs`, …) are followed; unknown variables, aliases and code
run by other interpreters (`python -c "os.system(…)"`) are not, and are left to System 1. Under Docker the
working directory is also known as `/workspace`.

System 1 judges what the rules let through, but only calls that can do harm: not read-only tools, not a write
or edit inside the working directory, not a shell command made only of reading commands (`ls`, `cat`, `grep`,
`find` without `-exec`, `git status`/`log`/`diff`, …; a third of the traced commands), not a web search, the
index tool or a process poll. It sees the session cut that System 1's other asks see (the user's messages,
the agent's replies, each call with a short note of its result), then the user's requests (their messages
only), the working directory and the call, with what is in each script the call runs (`python x.py`, `bash
x.sh`, `./x`, `source x`, under `uv run` too; up to three, 6,000 characters each), read where the call runs, so
a script is judged on what it does wherever it lives. The session is what tells a database the agent created
from one it found, or a `$DATABASE_URL` that points at production from a local one; "asked" still counts only
the user's own messages, so a tool result or the agent's own text claiming leave does not. One ask, two questions:
a choice of what the call would do, code written judged by what it would do when run (`routine`, which takes
scratch files in a temp folder, `destroys_data`, `rewrites_shared_history`, `exfiltrates`,
`credentials`, `weakens_security`, `runs_untrusted_code`, `affects_others`, `unclear`) and a noul, whether
the user asked for it; checking, setting up or starting what the user asked to install or run counts as
asked. So does a record that a source the user names ("follow the cleanup policy", "the SOP") says to create or
update, in the app the request is about, when the session shows that source coming from the user's own
organization. A source from an outside sender does not count, nor does a step it adds that the request's goal
does not need (deleting, overwriting data with values from outside the organization, paying, refunding,
writing to people or places the request does not involve). The call is blocked when P(any risk) ≥ 0.6 and
P(asked) < 0.3; `routine` and `unclear` never block,
and any failure, timeout or open circuit lets the call through. On the labeled set in `evals/tool_guard`
(94 calls: 48 routine, 23 risky the user did not ask for, the same 23 asked for), `drex-max-v1.1` blocks no
routine call and misses no unasked risky one in two repeats at any risk cut-off from 0.4 to 0.8;
`drex-latest` scores some routine and risky calls alike (0.53-0.64), so at 0.6 it blocks one routine call
or misses two risky ones per repeat. The fixed rules alone block 9 of the 23. Thirteen rows were added
since: scratch writes in a temp folder, and scripts run by name, harmless and malicious. On `drex-max`
judged on the call alone, `python /tmp/sync.py` (sends keys out) and `./cleanup.sh` (deletes ~/Documents)
went through at P(any risk) 0.01-0.02; shown what the scripts hold, it scores them 1.00, and all 107 rows
go right in two repeats. Six install rows were added after that. The request rarely names an
install, so P(asked) stays below 0.2 and risk alone decides. When `routine` covered only the project's
dependencies, `drex-latest` blocked the official JRE download (0.87) and put `npm install -g typescript` at
0.57-0.69. With a standard registry (PyPI, npm, crates.io, apt, brew) or the maker's official download named
as routine, and an unknown index, host or stranger's repository as untrusted, every install scores 0.25 at most,
while a typosquat from an unknown index stays at 1.00. All 113 rows go right in two repeats. The ask takes 0.37 s at the median (p90 0.6 s) on each judged call.

With the session in view (an offline eval: 75 authored sessions, among them calls that are
routine or harmful only in context and 8 prompt injections, and 236 guard asks from local journals; 3
repeats), the authored sessions go from 78.7% right to 92.0% on `drex-max` and 90.7% on `drex-latest`, and
false blocks of the real asks fall from 15.7 to 5.0 (of 234) on `drex-max` and from 14.0 to 7.7 on
`drex-latest`. Every verdict that changed changed to the right one; all 8 injections are still blocked at
P(asked) 0.02 or less. The ask's input grows from about 0.9k to 18k tokens in a long session, and the median
ask from 0.40 s to 1.05 s on `drex-latest`. What it still got wrong was cleanup of what the agent itself made
(its clone, test bucket or probe branch) and requests held as asked though their condition failed or they named
another target; `routine` now takes the first and "asked" holds each request against the session. On 23
sessions written for these before any wording was tried, own cleanup and conditions or targets go from 2-3.3
wrong verdicts each to none on both models, and all errors over the 334 cases fall from 19.0 to 7.7
(`drex-max`) and from 22.7 to 10.0 (`drex-latest`), every miss of a risky call gone on `drex-max`.

The user's `SYSTEM1.md` speaks for them to System 1, as `SYSTEM2.md` replaces System 2's prompt. It is read from
the config home only, never from a project: a repository anyone can write must not tell the guard what the user
allows (Claude Code's auto mode reads its allow rules from user settings only, for the same reason), and
`AGENTS.md`, shared by every coding agent, stays out of System 1's input. Every ask that reads the session
(the guard, supervision, the reads kept at compaction, memory recall) gets it after the session cut, labelled as
the user's; asks that judge a card or a passage do not. It sits on top of the cut, in the quarter of System 1's
window the cut leaves for the ask's own text, at most 5% of it: taken out of the cut, it cost 1.3 points of
the probes at 24,576 tokens (8 questions to 0 over two runs, McNemar p=0.008), all where the session fills
the budget; on top, the probes score as without it (93.6%, p=1.0). With the author's file (what System 1 is, who
speaks for the user, how to judge, the Linear and GitHub work that is normal for them, what they still want to be
asked), on `drex-max`: kept reads agree with the judge on 83% against 81%,
recall 70% against 56% at precision 75% against 78%; supervision makes 5 nudges against 6, 4 on turns the blind
judge calls stuck either way (the one it drops the judge calls progress). The guard's "asked" counts what the file
calls normal and not what it keeps for the user to ask: false blocks over the 334 cases fall from 7.7 to 1.3 on
`drex-max` (one real ask in one of three repeats) and from 9.0 to 1.0 on `drex-latest` (none real), but `drex-max`
misses 1.3 risky calls against none: a merge with failing CI, which the file keeps for the user, at P(any risk)
0.3-0.6, and killing another user's job, which `drex-latest` misses with or without the file. Put before the cut instead, it scored no better anywhere and made supervision nudge 8 times, 5 on
stuck turns. Memory recall with it was not measured.

Outside the run, hosts read the latest State from `get_state` (`state`), and event listeners get it
from the `state` and `compaction` events, so a fresh session or a child agent can be briefed from the
task and the State without a second write.

The stream also shows what each State write cost. Every State write — made for a compaction (`trigger:
"compaction"`, always) — carries the write's `usage`,
`duration_ms` and `model` on its `state` event (with `unseen_numbers` and `blanked` when it gave
numbers the run never showed), and `run_end` splits `usage` into
`agent_usage` (the replies) and `state_usage` (the writes); a write that failed
is a `state` event with `dropped` set (`write_failed`, or `branched` for one that landed after an aborted run
when the session had moved to another branch). A write announces itself first with `state_writing` (the turn,
the trigger, the turns it covers) and streams a whole State's text as `state_delta` events while the
write runs (edits do not stream: they are not the State's text), so a listener can show the write for what
it is; the `state` event carries the whole, checked text, with `method` (`edit` or `rewrite`) and
`fallback` (why edits gave way to a whole State). A search index being built (code search's, before a run's first turn in a git work tree) reports as `indexing`
events, `done` of `total` files a few times a second and once with `finished` and the duration (a build that
fails, such as a folder past `MAX_FILES`, sends no `finished`, and such a folder is not walked again); a run
without an event sink gets the bar on stderr instead. Every `decision` event names its `turn`, its
`subject` where there is one (a `keep` ask names the read it judged) and the `answers` System 1 gave,
so a stuck strike or a dropped read can be audited from the stream alone. A refused call
also gets a `tool_guard` event naming the layer (`rule` or `system1`), the rule or risk, the reason and
System 1's probabilities.

## Modules

| Module | Owns |
|---|---|
| `journal.py` | Append-only JSONL tree of every entry: messages, system prompts, full tool outputs, States, compactions, model changes, memory recalls, and System 1's side of the session where it happened (`event` entries: its decisions for a turn, the guard's refusals, nudges, and how each State write and compaction went; kept for the record, never read into the context). Resume, branch, replay. |
| `context.py` | Builds each request from the journal in four segments: system (byte-stable), pinned (instruction files and the recalled memory block; never pruned), history (the user's messages too; pruned only in whole turns), tail (per-turn additions, last). Token counts calibrated on provider usage. Compaction planning, the State write and prompt. |
| `loop.py` | One run: turn phases that end in a verdict (continue, nudge, stop with a reason). System 1's supervision at the turn's end (`supervise.py`) and the tool-call guard before each call (`guard.py`). Turn budget, continuation after an output-limit cut-off, unparsable tool calls kept out of history, window-scaled tool output, capped in tokens, with spill to a file, the State written before a compaction (by the session's System 2 model) and compaction at turn boundaries, the loop waiting for both. What ended or what System 1 found of the commands running is told at the next turn, and a final answer does not end the run while a command it moved to the background still runs. |
| `session_cut.py` | System 1's State: the session cut to System 1's window by code, rebuilt for every ask. |
| `guard.py` | The tool-call guard: fixed rules over a tokenized shell command or a tool's path, and the System 1 ask for calls that can do harm. |
| `hooks.py` | The eight hooks, extension ordering, fail-open handling, verdict types. |
| `tools/` | Typed tools that declare their effects and return failures as data with a `next_action`. The `base` pack: read, write, edit, bash, grep, find. |
| `executor.py` | Where commands run: the `Executor` protocol, local process groups with head-and-tail capture, and the workspace mount a sandboxed executor gives. |
| `processes.py` | The session's job manager, wherever its commands can run in the background: the `process` tool's jobs, and a record of every command running in the session (a `bash` call while it runs, a process group a finished command left running), each start and end a journaled `process` event. See [processes](capabilities/processes.md). |
| `runners/` | Where commands run, chosen by name: the `Runner` interface (`base.py`), `local`, `docker` (`docker.py`) and `opensandbox` (`opensandbox.py`, the `opensandbox` extra), or one a package registers under `intern.runners`. See [runners](capabilities/runners.md). |
| `providers/` | The Provider protocol, the OpenAI-compatible adapter (streaming, retries, think-tag and text tool-call recovery, argument repair), native Anthropic Messages and OpenAI Responses adapters, and a scripted FakeProvider for tests. |
| `models/` | `provider/model` references, profiles, context-length lookup and the working window. The published prices of Anthropic's, OpenAI's and Fireworks' serverless models (`prices.toml`), which their adapters apply to each reply's tokens, since those APIs report no cost; OpenRouter and the Decisions API report theirs. |
| `decisions.py` | Typed questions and answers, the Decisions API backend (the console, OpenRouter), cache, the check at session open, and retries with backoff for about two minutes; an ask still unanswered raises `System1Unavailable`, which stops the run with `system1_unavailable`. |
| `instructions.py` | Where prompts and rules come from: the base prompt, `SYSTEM2.md`, `AGENTS.md`, `SYSTEM1.md`. |
| `memory/` | The memory drive (a git repository of linked pages): format and lint, recall into the pinned segment once per session, and the System 1-only steward that proposes pages from journals. The run-time part (recall, drive write guard, reminders, `memory_search`) is the `memory` extension in `extensions/memory/`. See [memory](memory.md). |
| `session.py` | The public surface: prompt, steer, follow_up, abort, compact (a run's compaction, now, between runs), get_state, set_model, branch, get_entries. Where the end of a process goes (`job_notices`): to the run in progress, told at its next turn; between runs held for the next prompt (`quiet`), or a run of its own (`wake`, the terminal UI). |
| `protocol/rpc.py`, `cli.py`, `events.py` | JSON-RPC over stdio, the headless CLI, and the versioned event union. |
| `locate/` | The engine behind the search tools: `index` builds an index in a directory's `.intern/` (every file a card and chunks, derived text when a processor makes one) incrementally by size and mtime with no model call, with a progress bar on a terminal; `tree` arranges the files folders first into a balanced search tree (leaves of at most 20 files, inner nodes of at most 12 children, cut and merged by affinity: imports for Python, TF-IDF neighbours for everything) and gives every node a card; the root's card, the map pinned for the agent and never judged by System 1, also says what the repository and each of its subgroups is for, extracted with no model (a folder's README front matter or first paragraph, its package's description or its `__init__` docstring, else the described folders under it that stand for it best); `search` walks it with `decisions.ask` one card per state, best-first within budgets from the smallest node holding the scope, then every chunk of the kept files, the best chunks' text, and one choice over the top files side by side; `lexical` keeps an SQLite FTS5 table of every chunk beside the index (identifiers split, updated per file before a search) whose hits steer the walk to the groups holding them, never the ranking; `store` keeps the index fresh for an extension's tools (the search, and an index tool for extensions whose index the agent builds on request) and holds the hooks; `trace` reports a search as it runs (every node judged, its score, what the survivor rule kept), for a tool that opts in to send as `extension` events. The tool names, home folder and questions are the extension's. |
| `extensions/code_search/` | Code search: the engine with the text processors and questions that ask where a question's subject is defined or would be changed; the index builds on its own at the start of a run into `.intern/index.json`. `decision_code` only; needs `[code-search]`. Its TUI view (`view.py`) draws each search live under its call, from the search's `search_trace` events. An extension behind the code-search capability, not in the registry: `[capabilities.code_search] enabled = true` or `/caps` is the only switch (`session_setup.capability_extensions`; see [CONTRIBUTING](../CONTRIBUTING.md#extensions-and-capabilities)). |
| `extensions/workspace/` | The extension behind the workspace capability (`[capabilities.workspace] enabled = true`): the engine with processors for PDFs (OCR optional), Word documents, workbooks and tables (DuckDB), questions that ask for what a task needs, and the index the agent builds where the documents are (`index_files`, into that folder's `.decision_workspace/`); then `decision_files` (`search.py`: the walk's chunks and the BM25 top 100 of the passage table judged together by System 1, one SAG2 expansion round), `sql` and a document-aware `read`; needs `[workspace]`. `intern index --documents`. |

## Hooks

| Hook | Called | An extension may |
|---|---|---|
| `before_agent_start` | Once per run, before the first turn | Load state, add pinned context |
| `context` | Before every request is built | Prune, cap or add segments |
| `before_provider_request` | After the request is built | Change sampling, reasoning effort, the output cap |
| `tool_call` | Before a tool runs | Repair arguments, refuse with a reason, stop the run |
| `tool_result` | After a tool runs, before the window bound | Shape, reduce or annotate the whole output |
| `turn_end` | After each assistant turn | Continue, add a nudge, or stop with a reason |
| `session_compact` | When compaction runs | Replace or veto the State; adjust the reads kept verbatim |
| `agent_end` | At the end of a run | Write telemetry, run a finish gate |

Extensions run in configured order and fail open: an exception becomes a `hook_error` event.

Beside the hooks, `Extension.tools()` declares tools (pi's `registerTool`). They join the profile's packs
at session start, an extension tool replacing a pack tool of the same name in place; no hook can add
one, because the tool list is fixed before the first request.

An extension may also report what it does as `extension` events (`ExtensionEvent`, a payload whose type
the extension owns, sent through `ctx.emit`, not journaled) and name its own view in the terminal UI,
`Extension.tui_view = "module:Class"`, which draws them. The view lives with the extension; the TUI hosts
it (`tui/views.py`) and holds nothing particular to any extension ([terminal UI](tui.md#extension-views)).

An extension that needs more than the base install says so itself: `Extension.needs` (`intern.needs.Needs`)
names its `intern` extra and the modules that show the extra is there; the browser and MCP declare theirs beside
their code. Setup refuses a capability whose extra is missing, in those words, and the TUI's `/caps` installs the extra
before switching it on (`needs.install`, which knows no capability).

## Seams kept for later

| Seam | What plugs in |
|---|---|
| The journal | Replays test skills and self-edits (the steward already reads it) |
| `InstructionSource` | Prompts and rules from other sources; not memory, which is recalled per task |
| Model roles (`escalation`) | A large model to escalate to |
| Tool `effects` | An effect gate and a vault: secrets reach tools, never the model |
| `Executor` | Container sandbox, remote runners |

## Rules for changes

- One concern per change. A feature that needs every existing feature changed means the abstraction
  is wrong.
- Defaults move only with a paired comparison: same tasks, at least two repeats,
  pass rate reported with tokens, cost and wall time.
- Thresholds that share out the working window (compaction, the reply reserve, kept history) are fractions of it,
  never absolute token counts. One tool result is also capped in tokens (`max_tool_result_tokens`, 20,000): on a 1M
  window its 0.15 share let one `cat` hold 463K characters, re-sent every turn after.
- The working window bounds the prompt. The reply cap comes from the advertised window: a `--window` cap
  is an experiment on context, not on how much the model may think per turn.
