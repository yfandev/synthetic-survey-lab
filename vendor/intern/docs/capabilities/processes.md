# Managed processes

Every session has a job manager wherever its commands can run in the background (the `local`
runner, Docker, or any runner that runs background jobs), and the `process` tool is one of the
base tools. It supports `start`, `poll`, `write`, `terminate` and `list`. Start returns an opaque
handle; polling returns merged stdout/stderr, exit status, the command, its elapsed seconds and
absolute byte cursors. Pass `next_cursor` as the next `cursor` to avoid rereading output. If
older bytes were evicted from the bounded log, `omitted_bytes` reports the gap. A runner that runs
no background jobs has no job manager, and its sessions go without the `process` tool. Configs
that list the `process` pack (`tool_packs = ["base", "process"]`) keep working: the pack names the
same tool.

## What runs in a session

The job manager also keeps a record of every other command that runs in the session, so what is
running at any moment is known in one place:

- a `bash` call, from its start to its end (`call`);
- a job the `process` tool started (`job`), with when it last wrote output;
- a process group a finished command left running, as `cmd > out.log 2>&1 &` does
  (`left_running`), watched every second until it has no members. The session does not own
  it: nothing here stops it.

Each start and end is a `process` event (`ProcessEvent`, with a `ProcessInfo`: id, kind, command,
state, exit code, started and ended times), reported whenever it happens, between runs too, and
journaled as an `event` entry. ACP sends nothing for it.

System 1 reads what runs beside the user's goal: its view of the session (`session_cut.py`) has a
`processes` list while commands run, each with its kind, command, how long it has run and, for a
job, how long it has written nothing (`quiet_s`) and its last line of output. Like the goal, it is
never cut to fit System 1's window. The terminal UI's State panel shows it under the goal.

When System 2 writes the State for a compaction, code adds a `## Running processes` section
before `## User requests` while jobs or what commands left running still run: each job's handle
and command, and how long it has run. The turn that started a job may be among those the State
replaces; the section keeps the handle, so the agent can still poll or stop it. The writer never
sees it among the items it edits.

## When a process ends

The agent is told when a job, or a process a command left running, ends, so it never waits on
one with `sleep` or a poll loop. The notice is the harness's message (a `nudge` event,
reason `process_ended`): how the job ended, its exit code, how long it ran and the last 1,500
characters of its output, or, for what a command left running, that it ended. Ends the agent
already saw are left out: a job it polled, listed or terminated after it ended. Several ends
since the last turn are one message.

- During a run, it comes at the next turn, before the model is asked again; a run does not end
  while one is untold, so a final reply that crossed a job's end gets one more turn.
- Between runs, the session's `job_notices` decides. `quiet` (ACP, JSON-RPC and `-p`, whose runs
  answer a client's request) holds it for the next prompt, told right after it. `wake` (the
  terminal UI) starts a run of its own that tells it, with no message of the user's; a prompt sent
  as it starts runs right after it. An end the user's abort caused (the run's jobs, state
  `aborted`) wakes nothing. A headless `-p` session closes when its run ends, stopping its jobs,
  as before.
- A session reopened on its journal (resumed, `/reload`, ACP `session/load`) runs none of the jobs it
  ran before: closing stopped them. Of those whose end the agent was not told, it hears at the next
  run's first turn (``Background job … is from before this session was reopened, and no job outlives
  its session: it ended: terminated, after 4m 52s. …``), so it neither waits for a job that is gone
  nor guesses what became of it.

## Waiting is the harness's

The agent never waits on a background job; the harness does. `process poll` reads a
job's output and state and returns at once: it has no `wait_s`. `process start` waits up to 3 s:
a job that ends by then comes back with its result, as a `bash` call would, so a quick command
costs one turn rather than a start and a poll (agents run 1–3 s repro scripts as jobs). A turn whose only calls are polls
(or a list) of jobs that still run is the agent handing over:

- In the terminal UI (`wake`) the run ends with reason `waiting` and the transcript says
  ``⏸ Waiting on `./ci.sh`: you get the result when it ends; System 1 looks at it meanwhile.`` You
  have the session; the job's end, or a System 1 finding, brings the agent back.
- In a quiet session (`-p`, ACP, JSON-RPC) the run holds for news: the job's end, a System 1
  finding, a message from the user, or System 1's next scheduled look at what runs. A look that
  found nothing to tell is news here (`nudge`, reason `process_look`), so a job that never ends, a
  server, brings the agent back at 30 s, 1, 2, 4… minutes instead of holding the run until its
  timeout.

In the terminal UI, before it hands over, the agent owes the user a word: the first time it would
hand over without having written to the user since their latest message, it is told instead
(`nudge`, reason `process_waiting`) to answer whatever does not depend on the job and say what it
waits for. Not in a quiet session, where ending its turn would end the run and stop the job. Asked
"run CI, and why do the tests hang?", it answers the second part while CI runs. Asked how the job is
going, or whether it is stuck, it is told to read the job's output, or the file the job writes to, and
answer from that first: a `docker build … > build.log` writes nothing to the job's own output.

A message from the user never waits behind a command: a `bash` call still running when one comes
in goes on in the background at once (below), and a held run takes its next turn.

The `process` tool's description says poll never waits, that the agent is told when a job ends,
and that a turn that only polls running jobs ends its turn until then.

## System 1's look at what runs

While any command runs, System 1 is asked what each is doing (`process_watch.py`): at its
ages of 30 s, 1, 2, 4, 8 and 16 minutes, then every 30 minutes, one ask covering every command
running, over its view of the session with `processes`. Nothing running, no ask; a command that runs
ten hours costs about 25. Its verdicts: `progressing`, `waiting_as_expected` (a CI watch, a sleep, a
server, as meant), `stalled`, `failing`, `ready` (a server or watcher that is up while the agent
waits on it as if it would end) and `unclear`. Each look is a `process_check` event; the terminal UI
keeps a line a command, updated at each look (``⧗ System 1 · `npm test` at 4m 00s: progressing``). Commands at
P(stalled) + P(failing) + P(ready) of 0.6 or more are told to the agent as one harness message
(`nudge`, reason `process_check`), with how long each has run, how long it has written nothing and
its last line: at the run's next turn, keeping only the latest look, or between runs as
`job_notices` says (a run of its own in the terminal UI). A command that stays stuck is told again at
its next look, each further apart: there is no counter. Nothing here stops or moves a command. A look
System 1 cannot answer during a run stops that run (`system1_unavailable`), as its other asks do;
between runs, the next look tries again. The terminal UI's status bar lists, on System 1's row, the
background jobs it watches: each one's run time, its verdict at the latest look, and the next look.

## A long command goes on in the background

A `bash` command never holds the agent's turn for long. Every one runs as a task; one
still running goes on in the background as a job of the session's job manager, the same process,
not started again, when the first of these comes:

- it is plainly a wait, as it starts: a `sleep` of 10 s or more first, `tail -f`,
  `watch`, `gh run watch` or `gh pr checks --watch`;
- a message from the user comes in;
- it has run `[capabilities.process] move_after_s` (30 s by default; `0` never moves one);
- System 1, asked once it has run 2 s (so quick commands cost no ask), finds it waits or serves
  (a sleep, a CI watch, a poll loop, a server, a watcher) at P(waits) + P(serves) of 0.6 or more,
  from the user's requests and the command alone, as the guard judges a call;
- System 1's look at what runs finds it stalled, failing, or up and waited on.

The call returns `[Running in the background as job <handle> (<why>). …]` with its output so far,
where why is `still running at 30s`, ``System 1 judged it waits (0.88)``, `System 1 found it
stuck`, ``a message from the user came in`` or ``` `sleep 60` is a wait```; the same reason is the
`note` of the `process` events for the call (state `moved`) and its job, and the terminal UI shows
it (``⇢ `./ci.sh` went on in the background: still running at 30s``). The agent is told when it ends (above). A command that
works keeps the `timeout_s` it was given, and the call's result says when it will still be stopped. One that serves does
not: System 1 judged it `serves` as it moved, or a later look found it up and `ready`, and the `timeout_s` the agent
gave a call it expected to block on would only kill it. It runs until it is stopped, the session closes, or 24 hours
pass, as long as a `process` job may. In a session that tells the agent nothing between runs (`quiet`: `-p`,
ACP, JSON-RPC), a run does not end while a command it moved still runs: a final answer waits for
news for the agent (the command's end, a look, a message of the user's) and gets one more turn, so
a `-p` run gets the result it would have had, by the same deadline, and relies on no prompt. A
command that serves is not waited for, since it does not end: one System 1 judged `serves` as it
moved it, or later found up and `ready`. In the terminal UI (`wake`) the run ends at once, and the
command's end wakes the agent. Esc (an abort) stops the run and the command. A moved command takes no input; its output
arrives through its capture file, which the job reads as it grows and removes at its end (with
command secrets, the redacted output comes at the end). The `bash` description says a long command
goes on in the background; a session that never moves one says how to leave one running instead.

## Lifetime and limits

For applications, use `async with ProcessManager() as jobs` or call
`ProcessManager.aclose()` when its session closes; a session built by `Bootstrap` does that
itself. Up to four jobs run by default; retained histories and per-job output are bounded. Jobs
always have a deadline, and run cancellation terminates the owned process group. A job reads
end-of-file on its input, as a `bash` call does, unless it was started with `stdin: true`: a
command that reads its input (`docker exec -i`, a test runner's prompt) would otherwise wait on it
forever. Only a job started with `stdin: true` takes
`write`; sending input is bounded and writing after process exit returns an error. No process
starts when the session opens.
A job's process group may hold the memory a shell command may (`[commands] memory_share`
in `config.toml`, a share of physical memory; read every second): past it, the group is
stopped, the state is `out_of_memory` and the log says how much it held.
When a command exits, times out, is cancelled or its manager closes, any remaining
members of its process group receive a termination signal, a bounded grace period,
and a final kill signal. Redirecting output or ignoring termination cannot retain
that owned group. A child that deliberately creates a new session requires a
stronger operating-system isolation boundary.

Jobs run where the session's commands run. With the local executor they are the
owned process groups above, and that is not a sandbox. With an executor that runs
background jobs (Docker, or a backend whose executor has `start_job`), the executor
starts and stops them: each Docker job is its own container, attached through a
local `docker run --interactive` client, bounded by the container's memory and
process limits and removed when the job ends. The secret-injection wrapper prepares
a job as it prepares a command and runs it where the wrapped executor runs. Handles
belong to one live manager and cannot be reused by another session or after a
harness restart; a PID does not prove ownership. Polling a handle from before a reopen says
what became of its job, and an unknown handle's error lists the session's jobs, so a
mistyped one is corrected in one call. Use a durable backend job service
for restartable jobs.
