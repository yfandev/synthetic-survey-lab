# Memory

Memory lets a task agent start with what earlier sessions in the same world learned: where things
are, how calls are made, what failed, and what documents and people said.

## Turning it on

Memory is a capability like code search or the browser, switched in `[capabilities.memory]` and
with `/caps memory` in the TUI, but it is the one that is on by default. With no drive given, the
drive is `memory/` in the config home (`$INTERN_HOME`, default `~/.intern`), created
from the template on first use. To pick a drive or turn memory off:

```toml
[capabilities.memory]
drive = "~/mem-org"    # or --memory PATH, or $INTERN_MEMORY
enabled = false        # or --no-memory
```

The old top-level `[memory]` table still loads as `[capabilities.memory]`. `--memory` wins over
`drive`, which wins over `$INTERN_MEMORY`. A drive must be the
root of a git repository with a `main` branch; a session reads its pages from `main`'s commit. A working
tree that is off `main` or has edits (a shell command can write into it) only gets a warning at mount,
and the next merging steward resets it to `main`. With memory on the
session loads the `memory` extension (`extensions/memory/`), which does the recall and everything
under "During the run". With memory off it is not loaded: no recall runs, no memory tool is added
and no path is write-protected.
Memory is switched only this way: naming `memory` (or its class) in a profile's `extensions` or with
`--extension` is refused with a pointer to `[capabilities.memory]`, `--memory`/`--no-memory` and `/caps`.

## Drive

A drive is a git repository of linked markdown pages, created with `intern memory init DRIVE`
and checked with `intern memory lint DRIVE [--fix]`.

- `MEMORY.md` and a per-folder `index.md` are tables of contents.
- `map/`: places the agent works with (services, hosts, tools, folders, the environment).
- `user/`: the user, the people, organisations and things they deal with, and standing instructions.
- `lessons/` and `skills/`: ways of working learned across sessions.
- `work/`: one page per session with what was asked, what was done and how it ended.
- `sessions/`: session transcripts, searchable with the `memory_search` tool.

A page has front matter, stable `{#b-…}` blocks, a `## Contents` table and `[[links]]` for
relations. Every item records the session event it came from.

## Recall (reading)

Once per session, before the first request, System 1 walks the tables of contents from `MEMORY.md`
and picks the pages for the task, then up to three linked neighbours. The picked pages become one
block in the pinned segment, so compaction never drops it; its size is `thresholds.memory` of the
window (default 0.08). The recall is journaled and reused on resume. Each recall ask may take 30
seconds, so a busy System 1 can retry. A recall ask System 1 does not answer, after its retries,
stops the run with `system1_unavailable`, as any ask does; pages are never ranked without it. Lines on lesson and skill pages that the task text
already states are left out. The block tells the agent to use a line by going straight to the
place it names, and to confirm it with one direct read only before a change that rests on it. When
that read differs from the line, the read is current: the agent works with it and does not change a
file or record back to what memory says.
Skill lines from fewer than two sessions that succeeded are shown under Suggestions, to use only
when they fit the task.

## During the run

These come from the `memory` extension, which runs after the profile's extensions.

- **Write guard.** `write` and `edit` into the drive are refused. Memory changes only through the
  steward: a shell command can still edit the working tree, but sessions never read it and the next
  merging steward discards the edit.
- **Reminders.** Before a call that may change something, the agent is shown the memory lines about
  what the call uses (a page about its host or tool, or a line naming its tool, path or command),
  once per line. When a line names a document or record that holds what the call needs, the agent is
  asked to read it first. Reads are not held, and neither is a call that already runs the command a
  line quotes, unless the line warns against it.
- **`memory_search`** finds earlier sessions by words when no page holds what the task needs.

## Steward (writing)

Every session that ran and kept a journal is stewarded when it closes: a detached
`intern steward JOURNAL --drive DRIVE --merge` starts, so the client exits without waiting,
and logs to `<config home>/steward/<session>.log`. A session left open is stewarded once it has been idle
`steward_idle_s` seconds after a run (default 300; 0 waits for the close), so a client that keeps its
sessions open, as T3 Code does with its threads, still reaches memory before its next session starts. A
prompt after that is stewarded again, at the next idle time or the close; a close with no run since the
last steward starts none. The work page records how many user turns the session had when it was
stewarded (`user_turns` in its front matter), and a steward finding more redoes the session. Its writer is `[capabilities.memory] writer`, else the
session's System 2 when it is OpenAI-compatible; without one only the outcome and the work page are
written. `steward = false` in `[capabilities.memory]` turns this off. Stewards on one drive take turns
(`.git/nace-steward.lock`), and a session mounting the drive waits only while a steward moves its
working tree (`.git/nace-drive.lock`). A merging steward first publishes a `proposed/` branch that an
earlier one left when it was killed, if it still fast-forwards `main`, and deletes it otherwise.

`intern steward JOURNAL... --drive DRIVE [--system1 drex:<model>] [--writer provider/model]
[--world LABEL] [--merge]` turns finished sessions into proposed pages:

1. **Extract** (code): the user's messages, tool calls with their targets and results, and the
   files and places the session touched.
2. **Draft** (writer model): items of a few kinds (where, how, fails, state, event, rule,
   preference), each citing one line of the session. What documents state is kept with the
   document as its source.
3. **Outcome** (System 1): a session succeeded only with evidence: a step that checked a change the
   session is known to have made, or a user message confirming the result. For a step whose tool does
   not show whether it reads or writes (a command-line or custom tool), System 1 judges from the call and
   its result whether it changed something, and whether the result is a reply from the person the
   agent is helping; such a reply counts as a user message. Otherwise a session
   the agent called finished is unverified.
4. **Gate** (System 1): each item must be lasting, not only about this task, not a result the session
   worked out, and not a value from the data. What the assistant told the user is kept only when
   the session's outcome was confirmed. An item that restates, replaces or contradicts a line already
   on its page is marked so; lines that differ in a number (a port, a limit, a version) are never
   restatements. A replaced line moves to Earlier with `(until <date>)`, and so does any other line on
   map/, user/, lessons/ or skills/ that System 1 finds still states the replaced value.
5. **Lessons**: ways of working seen in at least two sessions; a lesson to follow needs a session
   that succeeded, and every lesson names a concrete step, not only more searching or care. Each
   line says how many of its sessions succeeded.
6. **Commit**: pages are written, indexes regenerated, lint passes, and one commit is made on
   `proposed/<session>` for review, built on `main` without touching the working tree. With
   `--merge`, `main` is fast-forwarded to it, the working tree is reset to `main` and the branch is
   deleted, so the next session recalls it.

System 1 only: if any System 1 ask goes unanswered after its retries, the session is deferred with
exit code 3 and no branch, and the next steward run on the same drive checkout retries it (the
queue is `.git/nace-steward-deferred`). System 1 retries a 429 or 529 twice with backoff before an
ask fails. A drive holds one world; a session from another world is refused.

## Observability

`memory_mount` (drive, commit) per session and `memory_recall` (commit, pages, chars, steps,
backend, reused, duration_ms) per recall. System 1 calls emit `decision` events with `memory_*`
and `steward_*` purposes.

Recall's asks are part of the run, so they count in its System 1 cost. The `memory_recall` journal entry
keeps what they cost, so the TUI's System 1 cost comes back with a resumed journal. The steward's asks and its
writer's calls come after the session closes. Each steward run adds what they cost on the session
(System 1 asks, tokens and cost; the writer's tokens and cost) to `<config home>/steward/<session>.json`,
beside its log, whether the session was proposed, deferred or skipped. The writer's cost is the one its
provider reports, else priced from the harness's price table. A resumed session shows it: in the TUI, on
the masthead and in the transcript; over ACP, in the `usage_update` after `session/load`
([acp.md](acp.md#updates)).
