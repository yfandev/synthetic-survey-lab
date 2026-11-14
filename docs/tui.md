# Terminal UI

The terminal UI is what `intern` opens on a terminal: the harness run interactively, on the same
`Session` the headless CLI, the JSON-RPC server and the ACP server drive. Its point is to make the
System 1 / System 2 integration visible: the State System 1 works from (the session as it reads it), the
summary System 2 continues from after a compaction, and System 1's decisions, turn by turn.

```bash
uv tool install "intern-agent @ git+https://github.com/nace-ai/intern"
intern                                       # System 2 from config.toml or $INTERN_SYSTEM2
intern --system2 openrouter/qwen/qwen3.8-27b "Find why tests/test_parser.py fails and fix it"
intern --session ~/.intern/sessions/20261108-213000-4242.jsonl   # resume one
intern -p --system2 openrouter/qwen/qwen3.8-27b "task"                  # headless instead
```

The UI opens when neither `-p` nor `--json` is given and stdin and stdout are a terminal; piped, or with
`-p`, the same command runs the task headlessly and prints the reply, so scripts and evals are unchanged.
With no `--system2`, `$INTERN_SYSTEM2` or `system2` in `config.toml`, the UI starts on the first of
the harness's known models your keys reach (Claude Haiku 5.5, Qwen3.8 27B on
OpenRouter, Nemotron 3.5 Lightning), says so, and `/system2` changes it; with no provider key at all it opens
without a session and the first `/system2` pick opens one. System 1 is a decision model from the console by
default and needs `$NACE_API_KEY` in the environment; when a session cannot open, the reason is shown
and `/system1` or `/system2` reopens it on the pick.
`intern tui` is the explicit form. It takes the CLI's options (`--system2`, `--system1`, `--window`,
`--max-turns`, `--profile`, `--cwd`, `--no-context-files`, `--no-extensions`, `--extension`,
`--no-session`, `--session`). A task on the command line is sent as the first prompt. `--session FILE` resumes a
journal where it stands, not replayed step by step: the transcript shows its last 30 turns (the earlier ones are
in the journal, their work in the State), the State panel the session as System 1 reads it, a compaction's
card the summary its journal entry keeps, the
masthead each system's cost and the context from all of it; it continues on the models it last
switched to. When memory stewarded the session (once it went idle or closed), System 1's cost includes the
steward's asks and System 2's its writer's calls, and a line under the conversation says what it spent
(`◇ Memory steward: 43 System 1 asks and 12.0k in / 1.5k out to the writer stewarding this session, $0.0191
(2 runs)`). A session of 1,700 journal entries opens in about 2 seconds; drawn whole, one widget at a time,
it took over 2 minutes. On quit the UI prints the command that resumes the session
(`intern --session ~/.intern/sessions/<stamp>-<id>.jsonl`). Every session has System 1 (a decision model from the console by default,
see the README); a run whose System 1 stops answering ends with `system1_unavailable` after about two minutes of retries.

## Layout

```
           ▄▄██▄           Intern  0.25.0
      ▄▄██████████▄
 ▄▄▄██▀▀▀█████▀▀▀████▄     System 2  anthropic/claude-haiku-5-5    thinking high   ▰▰▰▱▱    price unknown   ⠸ 11% of 131.1k
  ███  ██ ███  ██ ██▀      System 1  nace-ai/drex-max             decisions API             $0.0312
  ▀███▄▄▄█████▄▄▄███
    ▀██████████████▀
      ▀▀████▀▀▀            /help commands
┌───────────────────────────────────────────────────┬────────────────────────────────────────────────┐
│▐System 2▌ thinking slow ░░░░░░░░░░░░░░░░░░░░░░░░░│▐System 1▌ thinking fast ░░░░░░░░░░░░░░░░░░░░░░░░│
│ › Find why tests/test_parser.py fails and fix it  │ 22 turns · 2 prompts                           │
│ ▸ Thought for 3.4s                                │                                                │
│ Let me look at the failing test first.            │ watch ━ off-plan 0.01  ━ stuck 0.05            │
│                                                   │       ━ finished 0.70  ━ advancing 0.24        │
│ $ pytest -q tests/test_parser.py                  │ 1 ┤⣀⠤⠤⠤⠤⠤⡄       ⡤⠤⠤⠤⠤⣀⣀⣀⣀⣀⡀   ⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⡀     │
│ … (31 earlier lines, click or ctrl+o to expand)   │   │      ⢣ ⡖⠒⠒⠒⡆⢠⠃         ⢣  ⢠⠃         ⢣     │
│ FAILED tests/test_parser.py::test_dates           │   │      ⢸⢰⠁   ⢱⢸          ⢸⢀⢇⢸          ⠸⡀⡖⠒⠂ │
│ 1 failed, 41 passed in 0.52s                      │   │       ⡿    ⠸⡇           ⣿⢸⡇           ⣿    │
│ Took 2.1s                                         │   │       ⣧    ⢠⡇           ⣿⢸⡇           ⣿    │
│                                                   │   │      ⢰⢹    ⢸⢱          ⢠⠻⡸⢣          ⢀⠟⡄   │
│ ▎ System 1 · 0.6s   advancing 0.78                │   │      ⡸ ⣇⣀⣀⣀⡇⠸⡀         ⢸ ⠃⢸          ⢸ ⠉⠉⠁ │
│ ▎ finished 0.12   re-investigating 0.06           │ 0 ┤      ⠇       ⠇         ⠇   ⠇         ⠇     │
│                                                   │ guard ━ risk 0.00  ━ asked 0.90                │
│                                                   │       ✗ refused 1 of 8                         │
│ $ git push --force origin main                    │ 1 ┤  ⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢄    ⢠✗   ⡤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠄       │
│ ✗ System 1 refused   rewrites shared history 0.96 │   │              ⠈⠢⡀ ⡰⠁⠈⢆ ⡜                    │
│ risk 0.96   asked 0.02   0.4s                     │   │                ⢑⡜   ⠈⡞                     │
│ System 1 judged that it rewrites history that…    │   │               ⢠⠊⠈⢆  ⡜⠘⡄                    │
│                                                   │ 0 ┤              ⡰⠁   ⠱⡜  ⠘⡄                   │
│ refused · 0.0s                                    │   └────────────────────────────────────────────│
│                                                   │    1       5         10        15        20    │
│ ⟲ Compaction · turn 12 · 6 turns cut to a summary │                                                │
│  Context ━━━━━━━━╌╌╌╌╌╌╌╌╌╌───  14.1k → 6.2k −56% ├────────────────────────────────────────────────┤
│                                                   │▐State▌ fast and slow ░░░░░░░░░░░░░░░░░░░░░░░░░░│
│                                                   │                                                │
│                                                   │ ████▆▆▆▆▆▆▆▆▆██▆▆▆▆▆▆▆▆▆██████████████████████ │
│                                                   │ 24.0k of 24.6k tokens  ·  62.9k uncut  ━━━━━━━ │
│                                                   │ █ 15 whole   ▆ 10 abridged   inputs ≤ 60 chars │
│                                                   │                                                │
│                                                   │ ▌ › Find why tests/test_parser.py fails and…   │
│                                                   │ ▌ › Also keep the old API working              │
│                                                   │                                                │
│                                                   │                                                │
│                                                   │ █   ✓ edit src/parser.py                       │
│                                                   │ █   ✓ $ pytest tests/test_parser.py  12 passed │
│                                                   │ █   The parser handles %V now.                 │
│                                                   │ █ › Also keep the old API working              │
│                                                   │ █   I'll check the callers of the old name.    │
│                                                   │ █   ✓ grep parse_date_legacy in .              │
│                                                   │ █   Keeping it as an alias.                    │
│                                                   │ █   ✓ edit src/api.py                          │
│                                                   │ █   ✓ $ pytest -q          58 passed in 1.31s  │
│                                                   │ █   Pushing.                                   │
│                                                   │ █   ✗ $ git push --force origin main           │
│                                                   │ █   ✓ $ git push origin fix-iso-weeks          │
│                                                   │ █   Fixed: parse_date handles ISO weeks (%V)…  │
├───────────────────────────────────────────────────┴────────────────────────────────────────────────┤
│ Describe the task… (/help for commands)                                                            │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Look.** paper, ink and the mascot's violet (`tui/theme.py`): an off-white page, white cards, hairline rules,
near-black text, a warm gray for what is secondary; the mascot's violet for what is Intern's own (the wordmark, the
panels' tabs, the bar beside the user's messages and the goal, the prompt's border while it has the focus, the
cursors); green only for what went right, amber for what needs a look, red for what went wrong. The mark is the
mascot, the violet cube in round glasses at the top of the README, its three faces lit from the top, in its own
colours in both looks, fourteen cells by seven, once,
in the masthead: drawn in half blocks, a pixel the upper or lower half of a cell, so it is square on screen and
each cell can take two of the mascot's colours (`widgets.MASCOT`), and nothing on the page says it twice: the
System 2 panel opens on its heading alone. The panels are headed after Kahneman's two systems: System 2 `thinking slow` (it reasons and
acts), System 1 `thinking fast` (it reviews every turn, judges the tool
calls that can do harm and what a compaction keeps), its panel above the State's, headed `fast and slow`:
the "and" of Kahneman's title, where the two systems meet, since the State is what System 1 sees of System 2's work. Each heading is a band with the
panel's name on it as a tab, paper on the mascot's violet. Text the mouse selects takes a shade
of the page.

**Masthead.** Beside the mark, level with its top, the wordmark and version; level with its bottom,
`/help`, which lists the commands and the keys. Between them, set off by a blank row each way, a table:
System 2 (the model, then the thinking level) and System 1 (the model, then where it is served), the
labels in one column, the models in the next, the rest in a third. At the end of System 2's row, the run:
a spinner while one is in progress (`○` when none is), the latest request's tokens as a share of the
working window, and what the run is waiting on or what needs attention. The thinking level is in a filled box with a pip per level the model takes, filled
up to the one in force (none for the model's own), so Shift+Tab is seen to change it. Each system's cost
so far is in a fourth column. It moves as the calls are made, during a run too
(System 2's replies and State writes, System 1's asks, memory recall's among them), and comes back with a
resumed journal, but for code search's asks, which the journal does not keep. A
model whose cost is not known, its provider reporting none and the harness having no price for it,
leaves System 2's cost `price unknown` (`≥ $…` when other calls had one).

**System 2 panel**, the transcript, headed like the other two. Prompts (`›`), the user's own, in bold on a deeper shade behind a purple bar, so they stand apart from
the harness's blocks when scrolling, in the light look and the dark; the model's thinking folded to one line, a spinner and `Thinking…` while it
thinks and `▸ Thought for 3.4s` once it is done (click it to read the reasoning, click again to fold
it); replies as Markdown, in full; each tool call the way pi shows it: the call on a lighter shaded line in softer ink (`$ command` for bash, else the
tool and what it is about), under it the output the model saw (the end of a command's output, the start
of a read's; two lines, with a note of how many are hidden) and how long it took; while it runs, `running… 12s`,
counting each second (a resumed session's calls are shown without the count, and a call the run ended without a
result for says `no result` and how long it had run). Click a call, or press ctrl+o for the latest, to see its
whole output.

**System 1's cards.** Each of System 1's decisions is a card where it happened: one line, the System 1
panel's row in brief, in the panel's own lowercase words: `System 1 · 0.6s` (how long System 1 took, and
`cached` when it was), then each answer as a word and its probability, then after an arrow what came of
them, so a result does not read as one more option (`→ strike 1/3`, `→ refused`); the meters are the
panel's. The supervisor's card comes at the turn boundary: the mode in its colour (`advancing 0.78`), the two
next likeliest muted (`finished 0.12   re-investigating 0.06`; none that rounds to 0.00), then what followed
(a stuck review's strikes, a nudge). The guard (`guard.py`, which checks every tool call before it runs) is not a card but the last line of
the call's shaded head, so it reads as part of the call and not as the supervisor: `✓ System 1` (`✗ System 1
refused`, red), what the call would do (routine, one of the risks, or unclear), `risk` (P(any risk)) and
`asked` (P(the user asked for it)), muted unless one is past its threshold, how long it took, and the
reason under it when it refused. A call a fixed rule refused reads `✗ refused   rule · <rule>`; no
model was asked. A number past its threshold is red. The
supervisor card's edge is a thin muted bar.

**A compaction** is a card where it happened, about System 2's context, and the whole of it shows there.
First System 2 writes the summary it will continue from (the State, in the engine's words; the compaction
waits for it): the card says `⟲ Compacting · turn 12 · System 2 is writing its summary` with a spinner and
the latest line as the model writes it (the engine's `state_writing` and `state_delta` events; edits to a
standing summary do not stream), and the masthead says `writing the summary`. Then System 1 judges the reads being cut:
each verdict lands as a line of its own, a spinner and a stroke sweeping its meter, then the meter filling
to the probability as the verdict shows. The asks run at once, so the lines land a frame or less apart (a
long list in a few seconds), the latest eight in view. Once the cut is made the card settles: `⟲ Compaction
· turn 12 · 6 turns cut to a summary`, the context as a bar against the working window (what is left solid,
what was cut dashed, the room left a hairline) with the tokens before and after, what the agent continues
from (System 2's summary and its size, or a list of the cut turns' calls when no summary covered them or the
write did not land, and the recent turns as they were), and how many of the reads being cut System 1 kept
verbatim. Under that, a line per read: kept or why it was cut (`reference`, `done with`), P(working on) as a
meter, and the read; the kept ones first, then the closest to being kept, eight of them. A click shows the
summary itself, section by section (the Plan as a checklist: `✓` done, `▶` in progress, `○` pending, `·`
unmarked, read from the words the writer used; Done when, Now, Artifacts and Facts as lines; the User requests
one shaded block each), and every read; another folds them. A compaction stopped before its cut (Esc, an
extension's veto) says `⟲ Compaction stopped · nothing was cut`. A resumed session's compactions are drawn
settled, with the summary their journal entries keep.

The harness's other notes are muted one-liners at turn boundaries: receipts for big test output (`⌁`), nudges (`⚑`) and the run's end (`■`, with how many calls the
guard refused). When a background job ends, or a process a command left running, the agent is told at its next turn,
and the transcript shows each one's line (`⚑ Background job … ended: exited, exit code 0, after 4m 05s.`); between
runs, that end starts a run by itself, so the agent reports it without being asked. While commands run,
System 1 looks at them now and then (30 s, 1, 2, 4, 8 and 16 minutes into each, then every 30 minutes); each command has a
line that each look updates (``⧗ System 1 · `npm test` at 4m 00s: progressing``), and one it finds stalled, failing or
waited on gets an amber `⚑` line of its own, and the agent is told. A command that goes on in the background says why
(``⇢ `./ci.sh` went on in the background: System 1 judged it waits (0.88)``, or the timer, a stuck finding, a rule,
your message), and System 1's row of the status bar lists the jobs it watches, each with its run time, its verdict at
the latest look and the next look (`⧗ ./ci.sh 40s · waiting as expected · next look 20s`). When the agent only polls a
running job it hands the session back (``⏸ Waiting on `./ci.sh`: you get the result when it ends; …``); the job's end
brings it back. With code search on,
the index build (before a run's first turn in a git repository, else on `decision_code`'s first call) shows as a live meter
(`⌁ Indexing the working directory for decision_code ━━━━━━──── 1,234 of 5,712 files`, from the engine's
`indexing` events) that settles into `⌁ Indexed 5,712 files for decision_code in 6.1s`; the masthead counts
along. A folder outside the working directory that `decision_code`'s scope named is indexed the same way, the line
naming it.

**The search panel.** Each `decision_code` call draws its search live right under the call, in the transcript,
between the call and its output: the code-search extension's own view ([extension views](#extension-views)).

```
 decision_code where are malformed tool call arguments from the model repaired before the tool runs?
 ▌ ⌕ System 1 walks the index                                                          ⠹ 2.0s
 ▌ ● descend 1.3s  ──  ⠹ parts  ──  ○ read  ──  ○ compare  ──  ○ ranked
 ▌ judging every part of the kept files by its card
 ▌
 ▌ ◉ src/intern                109 files       LEADERBOARD
 ▌ ├─ ◆ providers        ━━━┿━━━━━━━━━── 0.85 ✓        1 • providers/repair.py  ━━━━━─ 0.90
 ▌ │  ├─ ■ repair.py     ━━━┿━━━━━━━━━━─ 0.90 ✓        2 ▼ providers/fake.py    ━──── 0.30
 ▌ │  │  ├─ · def parse_arguments(  ━━━━━━━━━━━━─ 0.87
 ▌ │  │  └─ … 8 more parts, best 0.21                  COMPARE
 ▌ │  ├─ □ __init__.py   ━━━┼─────────── 0.26 ✗        The top files side by side, once their text …
 ▌ ├─ ◇ files 3/4, protocol  ━━━─────────── 0.05 ✗
 ▌ └─ … 7 more pruned, best 0.05                       SYSTEM 1
 ▌ ┿ cut (best × 0.25)  ✓ kept  ✗ pruned  ⊘ budget     folders 1/40   ▱▱▱▱▱▱▱▱▱▱
 ▌                                                     asks 62 · judged 191 · failed 0
 ▌ ▾ fold   ▾ 54 more rows   latest 0.11 async def MemoryActions.t…   0.10 def MemoryActions.ent…
 running… 8s
```

The tree grows top-down as System 1 judges each card: the folders and groups of files, best first, then
the files of the groups kept, then every part of the kept files, then the text of the best parts. Each
bar fills to System 1's probability, an amber dash runs along it while the card is being asked, and the
tick across a bar is the survivor rule's cut among its siblings (the best of them times the gap): what
falls below it is pruned and dims, `⊘` marks a node kept but left when a budget ran out, `•` a part whose
text was read (its text's probability beside it), and `★` the files and places of the result. A file's
`→` is its score in the ranking. The leaderboard reorders as files are judged (▲▼ for a move, • for a
new entry) and becomes the result, with the best place of each file it stands behind; the compare
section fills one order at a time with the side-by-side votes over the top files; the System 1 section
counts the folders and files against their budgets, the asks, what they cost and how the scores spread.
The transcript keeps following the panel as it grows, unless you have scrolled up. The tree shows as many
rows as the sections beside it take (18 at least); the control row under it (click) shows the rest
(`▾ 54 more rows`), folds the panel to one line (`▾ fold`, `▸ unfold`) and, once the search is ranked,
replays it in place at twice its speed (`⟲ replay`). Searches in one turn run side by side, each in the
panel under its own call. A search that cannot run (a scope that does not exist, no index yet) says why in its panel. The tool's result,
what the agent reads, is the same with or without the panel; the search reaches it as `extension` events
(`kind: search_trace`, ten a second at most). They are not journaled: a resumed session shows the calls
of earlier searches without their panels.

**The workspace search panel.** Each `decision_files` call (the workspace extension) draws its union search the
same way, under the call: the workspace extension's view (`extensions/workspace/view.py`).

```
 decision_files When did the board approve the Zorblax merger, and who compiled it?
 ▌ ⌕ System 1 searches the workspace                                              ✓ ranked in 2.1s
 ▌ ● walk ∥ bm25 1.2s  ──  ● judge 0.4s  ──  ● seeds → terms 0.0s  ──  ● judge 0.3s  ──  ● ranked
 ▌
 ▌ WALK  folders → files → parts                    BM25  top 100 of 146 passages
 ▌ ◉ .                              38 files          1   § board/minutes_march.md · Minutes  ━━━━━━ 0.92 ✦
 ▌ ├─ ◆ board, staff   ━━━━━┿━━━━━━─ 0.90 ✓           2 ◆ § staff/quill.md · Staff          ━━━━━─ 0.80 ✦
 ▌ │  ├─ ■ doc_0.md     ━━━━━┿━━━━━━─ 0.90 ✓           3   § finance/2023/doc_0.md · Section 0      0.00
 ▌ ╰─ 20 from the walk ──────────────────▶ ⚖ ◀──────────────── 100 from BM25 · 15 in both ─╯
 ▌ JUDGE  ▰▰▰▰▰▰▰▰▰▰ 150/150 passages  scores 0 ▇▁▁▁▁▁▁▁▂▃ 1
 ▌ SEEDS  ✦ ops/vendors/doc_2.md 0.98  ✦ ops/vendors/doc_4.md 0.94  … 8 more
 ▌ HOP    20 terms → 45 new passages, 43 in files the walk left
 ▌   board/minutes_march.md ─── ada quill  ▮▮▮▮▯▯ ──▶  2  ↯ staff/quill.md 0.80
 ▌
 ▌ RESULT  top 10, in diversity order                          SYSTEM 1
 ▌  1 ★ § ops/vendors/doc_2.md · Section 2  bm25    ━━━━━━ 0.98   asks 65 · judged 223
 ▌  2 ★ § board/minutes_march.md · Minutes  bm25    ━━━━━━ 0.92 ▲3 walk files 32/120
 ▌  6   § finance/2023/doc_0.md · Sect…     ↯ 2     ━━━━━━ 0.96 ▼4 passages 146 · 3 by card
 ▌ ▾ fold   ⟲ replay   ▾ 144 more rows   latest 0.43 · doc_4.md · Section 0
```

The two lanes run at once: on the left the walk's tree grows as `decision_code`'s does (the same drawing), on
the right the BM25 lane fills its ranked list of passages (`◆`: the walk read it too). They converge on
System 1's judge, which judges both lanes' passages eight to an ask: a dash runs along a passage's bar while
its ask is out and the bar fills when the answer comes back. The ten best light up as seeds (`✦`); the hop
shows each seed's terms that reached a new passage, by weight, how many they reached and the best of them,
`↯` when the walk had left its file (pruned or out of budget). The result closes it: the top ten in
diversity order with each passage's kind (`▤` page, `§` section, `⊞` sheet tab, `▦` table, `◩` image, `□`
file), the lane that brought it (`walk`, `bm25`, `walk+bm25`, `walk card` for a chunk ranked by its prior
unread, or the term that reached it) and how far the diversity order moved it from the score order (▲▼).
The stage chips time each stage. The controls are `decision_code`'s; once ranked the lanes and the hop shrink to
five rows, and `▾ more rows` shows them all. The events are the same `search_trace` extension events, sent
under the workspace extension's name, their payload the walk's trace plus the union's passages, stages and
terms (`extensions/workspace/trace.py`).

**State panel.** What System 1 works from (`tui/state_panel.py`). System 1's State is the session as it
reads it: rebuilt from the journal for every ask, the user's latest requests as the goal and the whole history
after them, fitted to its budget (75% of `max_state_tokens`) by abridging long texts, collapsing old ones,
shortening old calls to one line and at last leaving out the oldest (`session_cut.py`, `docs/architecture.md`).
The panel draws that cut, built the same way from the journal half a second after messages land (off the event
loop: a long session's takes a tenth of a second) and on resume:

- the strip: the whole session in one gray row, oldest left, each cell a stretch of messages as tall as
  System 1 reads them, a full block whole and the floor left out, so where the cut wore the session down shows
  at a glance; it is not drawn until the cut wears something down, as a row of full blocks says only what the
  legend does;
- the tokens the cut takes of its budget, with a meter, and what the history would take uncut when that differs; then how many
  messages were kept each way (`█` whole, `▆` abridged, `▄` collapsed, `▂` one line, `▁` left out) and what tool
  inputs were cut to;
- the goal, a line per request, shaded as the transcript's prompts; on a panel under 22 rows it is not drawn, as
  the history has the requests too;
- the history, oldest first, following the end: a row per prompt (`›`), harness message (`⚑`), reply and call
  (`✓` or `✗`, the tool or `$` and the command, what it was about, a check's tally in green or red), each behind a
  rail in its message's glyph; runs left out are one row, and a rule marks where System 2's context starts.
  The newest 400 rows are drawn.

The panel is System 1's alone. The summary System 2 writes at a compaction is System 2's own context, not what
System 1 reads, so it is not here: each compaction's card in the transcript shows it. `/state` opens System 1's
State in full, as the exact text it is given (its JSON laid out).

**System 1 panel.** A chart of the session (`tui/chart.py`): two line plots, drawn in braille (two dots
across and four down in a cell), on one turn axis. The transcript's cards say each turn in full; this says how
the work went.

- **watch**: what the supervisor (`supervise.py`) answered at each turn's end, a point a turn: P(advancing)
  green, P(stuck) amber (re-investigating and retrying together: either way the work is not moving on),
  P(off-plan) red and P(finished) in ink, which rises where a prompt's work is done, so advancing's drop there
  reads as an end, not as stuck.
- **guard**: at the turns where System 1 judged a tool call, P(risk) amber and P(the user asked for it) of the
  turn's riskiest call, joined across the turns between. A refused call is a red `✗` on its point.

A line under 0.1 is not drawn, so the floor stays clean and a line that shows means something; there are no
threshold lines. Where lines cross in a cell the more alarming one shows (off-plan, then stuck, finished,
advancing; risk over asked). Each plot's legend gives each line's latest value, the current verdict's in bold,
and the guard's says how many calls it refused of those it judged. Above the plots: the turns and the prompts.
Under the axis, its turn numbers. The plots run on across prompts: a short session takes up to two
cells a turn, growing to the right; once the turns outnumber the dots across, a point takes a few turns (the
line above says how many), the watch lines their mean and the guard its riskiest call, so a risky call
is never averaged away. The chart fits the panel's height and never loses its axis: the watch plot and the
axis come first, then, while they fit, the guard's plot and a spacer; rows left over make the
plots taller, the larger share to the watch. System 2's context is the masthead's.

The chart comes from the event stream alone (`decision`, `state`, `nudge`, `compaction`, `tool_guard`), so
what it shows is what `intern -p --json` would print. The journal
keeps those events where they happened (`event` entries), and each reply's prompt size, so a resumed session
(`--session`, `/reload`) draws the same chart; the System 1 cards and the compaction cards in the transcript
come back as they were for the turns it shows. A journal from before the events were kept resumes with no chart.

## Extension views

What an extension shows on the screen ships with the extension; the TUI only hosts it. An extension
names its view, `tui_view = "module:Class"`, a subclass of `intern.tui.views.ExtensionView`, and
reports what it does as `extension` events (`ExtensionEvent.of(name, kind, payload)` through `ctx.emit`;
the payload is the extension's own pydantic model, which its view validates back). When a session opens,
the TUI imports the views of its extensions, one per extension, and gives each every live event after its
own handling: its extension's reports and the engine's events, such as a tool call starting. A view draws
through its `ViewHost`:

| `ViewHost` | Does |
|---|---|
| `open(screen)` | a window of the view's own over the transcript (a Textual screen it supplies, with its own `DEFAULT_CSS`), until the view dismisses it |
| `attach(call_id, widget)` | a widget under a running tool call, between the call and its output |
| `with growing():` | nothing now, kept for views that wrap a change in it: the transcript keeps showing its end as anything in it grows, an attached widget too, unless the reader has scrolled up |
| `marker(content, key)`, `notice(text, kind)` | a line in the transcript; a marker with the same key replaces the last |
| `activity(text)` | what the run is doing, at the end of the masthead's System 2 row |

```python
class Probe(Extension):
    name = "probe"
    tui_view = "my_package.probe_view:ProbeView"  # imported by the TUI alone

    async def tool_result(self, call, result, ctx):
        ctx.emit(ExtensionEvent.of(self.name, "seen", Seen(tool=call.name)))
```

Code search's [search panel](#layout) is one: `extensions/code_search/view.py`, named by
`CodeSearch.tui_view`, fed by the search's `search_trace` events; the workspace search panel is another
(`extensions/workspace/view.py`, `Workspace.tui_view`). Only the TUI imports a view, so a
headless or ACP run never loads view code; `intern rpc` and
`-p --json` print `extension` events with the rest, and ACP sends nothing for them. Views get live events
only: extension events are not journaled, so a resumed session gives its views none of the replayed
calls. A view goes with its session (a model pick, `/caps` and `/clear` reopen it: `close()` puts away
what the old one opened). Views fail open, as hooks do: one that cannot be imported, or raises, is named in
a notice and is off for the rest of the session, and the TUI goes on.

## Commands and keys

| Command | Does |
|---|---|
| `/system2` | pick the System 2 model from the catalog (`intern models`): type to filter, ↑↓, Enter; `/system2 provider/model` sets it directly. A name not listed is accepted as typed. |
| `/system1` | pick the System 1 model: the decision models the console serves, then the decision models the decisions API serves as `decisions:<model>` (OpenRouter: Jev, Perplexity Decider, …); `/system1 <model>` sets it directly. A model whose key is missing is refused with the reason, and the current one stays. |
| `/state` | System 1's State in full: the exact text it is given |
| `/caps` (`/capabilities`) | pick a capability to switch on or off: the tool packs (`base`, which has `process`, `web-search`, `web-read`), the browser, code search (System 1), the workspace (System 1 over a folder of office files), the memory drive, each MCP connection the config names; grouped by what is on, each with what it gives the agent. `/caps <name>` flips one directly; `/caps memory <path>` names a drive. Switching on one that needs an extra this install lacks (the browser, code search, the workspace, an MCP connection) installs it first, into the environment the TUI runs in, as the installed `intern` lists it (with `uv pip`, else `pip`), and for the browser runs `playwright install chromium`, which downloads Chromium once; the transcript says what it installs, and a failed install says why and the command to run by hand, the capability left off. The session reopens on its journal with the change (`docs/capability-settings.md` has what each one is), the screen as it was; if it cannot open on the pick, it comes back as it was. |
| `/runner` | pick where commands run: `local`, `docker`, `opensandbox` or an installed runner, the one in use first, each that cannot run here saying why. `/runner <name>` sets it directly. The session reopens on its journal on the new runner ([runners](capabilities/runners.md)). |
| `/secrets` | the [secrets store](secrets.md): each entry with whether it loads now (problems first, with what to do; a key the harness reads itself, such as a model key, is marked harness only), and **Add a secret…**. Enter on a keychain entry replaces its value in a form whose value field is hidden. `/secrets set <NAME>` opens the form, `/secrets remove <NAME>` deletes one. New sessions get the change; `/reload` uses it now. Values are never shown. |
| `/thinking <level>` | the thinking level: `low`, `medium`, `high`, `xhigh` or `max` through Anthropic's API, `low` to `xhigh` through OpenAI's, and `low`, `medium` or `high` elsewhere when the model's profile names one. Shift+Tab cycles it, ←/→ in the `/system2` picker set it with the pick, and the masthead shows it (`default` is the model's own). It applies from the next request, during a run too, and stays across model switches, `/caps` and `/clear`; a model that lacks the level keeps its own, the switch notice says so, and the choice is back on the next model that takes it. |
| `/compact` | compact the conversation now, between runs, exactly as a run does once its request passes `compact_at` of the working window (75% by default): the turns before the most recent `keep_recent` share (35%) cut to System 2's summary (the State), written first when it does not cover them (trigger `compaction`), the file reads System 1 says the work needs kept verbatim, the extensions' say. Only the size check is skipped. When the whole conversation fits in the recent share there is nothing to cut, and it says so; with a 1M window that is anything under 350k tokens. The masthead reads `compacting` meanwhile and a prompt waits for it; Esc stops it before anything is cut. |
| `/abort` | stop the run in progress |
| `/clear` | start over: the screen cleared and a new session opened on a fresh journal, with the models and capabilities as they are. The journal so far stays on disk, to resume with `--session`. |
| `/reload` | restart on the code, config and keys as they are now (an editable install's latest edits, a reinstall, a key added to `~/.intern/.env`) and resume this session on its journal, with the models, capabilities and thinking level picked here. The process starts again on the same command line, the task on it not sent twice, so everything is loaded anew: the harness, the UI, extensions, profiles, `config.toml` and `.env`. It installs nothing: a changed dependency needs `uv tool install --reinstall` (or `uv sync`) first. The screen stays up across the restart, the last frame on it until the new process paints over it. Waits for the run to finish; refused with `--no-session`, whose journal is in memory only. |
| `/panels` | hide or show the right-hand panels |
| `/dark`, `/light` | the dark look (paper and ink swapped, each a little grayed down, and the mascot's lighter violet) or the light one, the default; `/reload` keeps it |
| `/help`, `/quit` | the commands and the keys; close the session and exit (closing the terminal window, or a SIGTERM, quits the same way, so the session is still stewarded) |

Enter sends the prompt, or, while a run is in progress, steers it: the text shows as `queued` and goes
into the run before its next model call, where its line then moves. Esc aborts the run and sends what is
still queued at once, as the next run's prompt. The prompt takes several lines: Shift+Enter starts a new one (reported by a
terminal that tells it from Enter, such as Ghostty, kitty or WezTerm; on a Mac, read off the keyboard
when Terminal.app or a multiplexer such as herdr or tmux delivers it as a plain Enter, as pi does), and
so does Ctrl+J, in any terminal;
a paste keeps its lines, and the box grows to eight lines of text, then scrolls. A file dropped on the
terminal lands as its path (a `file://` URL some terminals drop lands as the path), for the agent to
`read`. The images the prompt names, dropped from anywhere, pasted or typed, show on its bottom border
(`image logo.jpeg`), with a warning when System 2 cannot see images. A dropped image from a
temporary folder (a screenshot dragged from its thumbnail lives in one macOS empties within a minute or
two) is copied to `$TMPDIR/intern-images` first, and the prompt names the copy. A drop or a paste
lands in the prompt even when the prompt does not have the focus: dragging a file in from Finder takes
the focus from the terminal, and the drop still lands, after what is typed. A click puts the
cursor where it lands; a command being typed is completed in ghost text that → accepts. Esc aborts the
run (or clears the prompt when idle). Ctrl+O expands or
folds the latest tool call's output; clicking a call does the same for that one. F2 toggles the panels. PageUp and PageDown scroll the transcript; the prompt keeps the keyboard, whatever is clicked. The transcript keeps the end of a run in view as it comes in, a long reply to its last line; scrolled up, it stays where you are until you scroll back to the end. Shift+Tab cycles the thinking level.

Drag over text in any panel to select it: it is copied as the mouse is released (the masthead says so for a moment). The copy goes to the terminal through OSC 52 and to the machine's clipboard through `pbcopy` (`wl-copy`, `xclip` or `xsel` on Linux), so it works in Terminal.app and in iTerm2 without its clipboard setting.
Ctrl+C copies what is selected in the prompt, and with nothing selected clears it; on an empty prompt
it says how to quit. Ctrl+Q quits. Cmd+C and Cmd+Q do the same where the terminal passes them on, but
macOS terminals keep them (Ghostty and Terminal.app copy and quit the terminal itself), so Ctrl is the
way. Model changes wait for the session to be idle.

The prompt edits as a text box does. Shift with the arrows selects (Alt+Shift by word), as does a drag; a
double click selects a word (a path is one word), a triple click the line, and F7 all of it. Typing,
Backspace, Delete or a paste replace what is selected. Ctrl+X cuts the selection (the line when nothing
is selected), Ctrl+Z undoes and Ctrl+Y redoes. Alt+Backspace or Ctrl+W deletes the word before the
cursor, Alt+Delete the word after it, Ctrl+U and Ctrl+K delete to the start and end of the line (or the
selection), and Alt+←/→ move by word. Where the terminal passes the Cmd key on (kitty, or Ghostty with
its own Cmd bindings removed), Cmd+A selects all, Cmd+X cuts, Cmd+Z and Cmd+Shift+Z undo and redo,
Cmd+Backspace deletes to the line's start, and Cmd+←/→/↑/↓ (with Shift to select) move to the ends of
the line and the text.

## Not yet

- No permission prompts: the agent runs every tool, as in every other client.
