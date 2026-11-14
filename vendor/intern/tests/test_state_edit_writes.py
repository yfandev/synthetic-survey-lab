"""State writes as edits: System 2 writes the State at each compaction (these runs' outputs fill a small window every
two or three turns); once there is a State, the writer's fork asks for edits on its numbered items, code applies
them, and only edits that cannot be applied cost a second reply with the whole State (edits that grow it past the
size limit, a second reply with edits that trim it). Every model edits, and a new request from the user is brought in
with edits too; the user's requests are a section code keeps, not the writer."""

from __future__ import annotations

from pathlib import Path

from intern.context import STATE_PROMPT, requests_section
from intern.events import State, StateDelta
from intern.journal import StateEntry
from intern.loop import STATE_EFFORT
from intern.messages import UserMessage
from intern.models.profile import Profile, Thresholds, load_profile
from intern.providers.base import ProviderRequest
from intern.providers.fake import Turn
from intern.state_edits import MAX_EDITED_WORDS

from .support import is_state_edit, is_state_write
from .test_loop import Opinionated

FIRST = "## Plan\n1. print: in progress\n2. report\n\n## Now\nprinting\n\n## Facts\n- printf works"
# What code adds to every State of a run asked `do the task` (the harness's default prompt in these tests).
ASKED = "\n\n" + requests_section(["do the task"], 10_000)
WINDOW = 12_000  # with 6,000-character outputs, a compaction (and a State write) every two or three turns


def output(size: int = 6_000) -> str:
    return f"printf '{'x' * size}'"


def agent(turns: int, edits, whole: str = FIRST, writes: list[ProviderRequest] | None = None, size: int = 6_000):
    """The agent runs `turns` bash calls printing `size` characters, then answers; a State write gets `whole`, or for
    edits, `edits(n)` with n the number of writes so far."""
    calls: list[int] = []
    seen = writes if writes is not None else []

    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            seen.append(request)
            return Turn(text=edits(len(seen)) if is_state_edit(request) else whole)
        calls.append(1)
        return Turn(text="done") if len(calls) > turns else Turn(tool_calls=[("bash", {"command": output(size)})])

    return script


async def test_after_the_first_state_the_writer_sends_edits_on_numbered_items_and_code_applies_them(make_harness):
    writes: list[ProviderRequest] = []
    # Words, not digits: a number the run never showed costs a reply fixing it (unless the cwd happens to show it).
    word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    edits = lambda n: f"## Plan\nset 1: print: done ({word[n]})\n## Now\nreplace: printing {word[n]}"  # noqa: E731
    harness = make_harness(agent(8, edits, writes=writes), window=WINDOW, system1=Opinionated())
    assert (await harness.run()).reason == "completed"
    states = harness.recorder.of(State)
    assert len(states) >= 3 and len(writes) == len(states), "one request a write: no fallback"
    assert states[0].method == "rewrite" and all(s.method == "edit" and s.fallback is None for s in states[1:])
    last = word[len(writes)]
    assert (
        states[-1].text
        == f"## Plan\n1. print: done ({last})\n2. report\n\n## Now\nprinting {last}\n\n## Facts\n- printf works{ASKED}"
    )
    # The edit request numbers every item, ends the agent's own request like the whole one does, and is not it.
    # The user's requests are not among the items: code keeps them.
    ask = writes[1].messages[-1]
    assert isinstance(ask, UserMessage) and STATE_PROMPT not in ask.content and "User requests" not in ask.content
    assert ask.content.endswith(
        "## Plan\n1. print: in progress\n2. report\n\n## Now\n1. printing\n\n## Facts\n1. printf works"
    )
    assert writes[1].tools == writes[0].tools
    # Only the whole State streams: edits are not the State's text.
    assert {d.turn for d in harness.recorder.of(StateDelta)} == {states[0].turn}


async def test_edits_that_cannot_be_applied_cost_one_more_reply_in_the_same_fork_with_the_whole_state(make_harness):
    writes: list[ProviderRequest] = []
    rewritten = FIRST.replace("printing", "printing, after the fallback")

    def edits(n: int) -> str:
        return "## Plan\nset 9: the ninth step is done."  # the Plan has two

    script = agent(6, edits, writes=writes)

    def with_fallback(request: ProviderRequest) -> Turn:
        if is_state_write(request) and not is_state_edit(request) and len(writes) >= 1:
            writes.append(request)
            return Turn(text=rewritten)
        return script(request)

    harness = make_harness(with_fallback, window=WINDOW, system1=Opinionated())
    assert (await harness.run()).reason == "completed"
    states = harness.recorder.of(State)
    second = states[1]
    assert (second.method, second.text) == ("rewrite", rewritten + ASKED)
    assert second.fallback is not None and "Plan has no item 9" in second.fallback
    edit, whole = writes[1], writes[2]
    assert is_state_edit(edit) and not is_state_edit(whole)
    # The whole State is asked in the same fork: same prefix, only the request at the end differs.
    assert whole.messages[:-1] == edit.messages[:-1]
    last = whole.messages[-1]
    # The previous State is the summary the compaction left at the start of the agent's context.
    assert isinstance(last, UserMessage) and STATE_PROMPT in last.content
    assert "The previous state is the one given at the start of this conversation." in last.content
    assert second.turn in {d.turn for d in harness.recorder.of(StateDelta)}, "the whole State streams"


async def test_a_number_an_edit_adds_that_the_run_never_showed_goes_back_to_the_writer_with_the_state(make_harness):
    writes: list[ProviderRequest] = []
    edits = lambda n: "## Facts\nadd: total 777"  # noqa: E731
    harness = make_harness(agent(6, edits, writes=writes), window=WINDOW, system1=Opinionated())
    assert (await harness.run()).reason == "completed"
    edited = next(s for s in harness.recorder.of(State) if s.method == "edit")
    assert edited.unseen_numbers == ["777"] and edited.text.endswith(f"- printf works\n- total ?{ASKED}")
    fix = next(r for r in writes if "Your edits are applied" in str(r.messages[-1]))
    shown = fix.messages[-1]
    assert isinstance(shown, UserMessage) and "777" in shown.content and "- total 777" in shown.content


async def test_no_change_keeps_the_state(make_harness):
    harness = make_harness(agent(6, lambda n: "No change."), window=WINDOW, system1=Opinionated())
    assert (await harness.run()).reason == "completed"
    states = harness.recorder.of(State)
    assert states[1].method == "edit" and states[1].text == FIRST + ASKED and states[1].landed


async def test_an_edit_past_the_size_limit_is_trimmed_by_edits_in_the_same_fork_and_kept_if_they_fail(make_harness):
    long = " ".join(["word"] * MAX_EDITED_WORDS)
    for trim, trimmed in (("## Facts\ndrop: 2", True), ("## Facts\ndrop: 9", False)):
        writes: list[ProviderRequest] = []
        replies = {2: f"## Facts\nadd: {long}", 3: trim}
        harness = make_harness(
            agent(12, lambda n, r=replies: r.get(n, "No change."), writes=writes),
            window=WINDOW,
            system1=Opinionated(),
            # Compactions before half the window: the trim's fork (the request, the edited State, the trim prompt)
            # fits beside the base tools' schemas.
            profile=Profile(thresholds=Thresholds(compact_at=0.45, keep_recent=0.2)),
        )
        assert (await harness.run()).reason == "completed"
        edited = harness.recorder.of(State)[1]
        assert edited.method == "edit" and edited.fallback is None, "never written whole for its size"
        assert edited.text == FIRST + ASKED if trimmed else long in edited.text
        edit, trim_request = writes[1], writes[2]
        assert trim_request.messages[: len(edit.messages)] == edit.messages, "the edit's own fork"
        asked = str(trim_request.messages[-1]).replace(",", "")
        assert str(MAX_EDITED_WORDS) in asked and str(MAX_EDITED_WORDS * 9 // 10) in asked


async def test_every_model_edits_after_its_first_state_whatever_its_profile_says(make_harness, tmp_path: Path):
    # A profile of the user's own from before still says whether to edit: the key is ignored, not refused.
    own = tmp_path / "mine.toml"
    own.write_text('match = ["local/small-model"]\nstate_edits = false\n')
    for profile in (load_profile("local/small-model"), load_profile("local/small-model", override=own)):
        writes: list[ProviderRequest] = []
        harness = make_harness(
            agent(6, lambda n: "## Now\nreplace all: printing more", writes=writes),
            window=WINDOW,
            system1=Opinionated(),
            profile=profile,
        )
        assert (await harness.run()).reason == "completed"
        assert [is_state_edit(r) for r in writes][:1] == [False] and len(writes) >= 2
        assert all(is_state_edit(r) for r in writes[1:]), "after the first State, edits only"
        assert all(r.reasoning_effort == STATE_EFFORT for r in writes)


async def test_a_failed_edit_call_gets_the_whole_state_and_edits_beside_a_tool_call_still_count(make_harness):
    writes: list[ProviderRequest] = []

    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            writes.append(request)
            if not is_state_edit(request):
                return Turn(text=FIRST)
            if len(writes) == 2:
                return Turn(error="the edit request was refused")
            return Turn(text="## Now\nreplace all: printing again", tool_calls=[("bash", {"command": "echo hi"})])
        calls.append(1)
        return Turn(text="done") if len(calls) > 8 else Turn(tool_calls=[("bash", {"command": output()})])

    calls: list[int] = []
    harness = make_harness(script, window=WINDOW, system1=Opinionated())
    assert (await harness.run()).reason == "completed"
    states = harness.recorder.of(State)
    failed = ("rewrite", "the edit request failed", FIRST + ASKED)
    assert (states[1].method, states[1].fallback, states[1].text) == failed
    assert states[2].method == "edit"
    assert states[2].text.endswith(f"## Now\nprinting again\n\n## Facts\n- printf works{ASKED}")


async def test_a_reply_with_no_state_in_it_is_a_failed_write_and_the_previous_state_stands(make_harness):
    # Qwen3.8 thought through its budget and replied the first; Claude, asked about State writes, answered the
    # user under headings of its own. On main each landed as the State.
    junk = "The state was written as requested — no tools called."
    answer = (
        "That's a real State, and a good example of why it's slow.\n\n## Why it's slow\n- a full call\n## Options\n1. a"
    )
    writes: list[ProviderRequest] = []

    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            writes.append(request)
            return Turn(text=FIRST if len(writes) == 1 else junk if len(writes) % 2 else answer)
        calls.append(1)
        return Turn(text="done") if len(calls) > 10 else Turn(tool_calls=[("bash", {"command": output()})])

    calls: list[int] = []
    harness = make_harness(script, window=WINDOW, system1=Opinionated())
    assert (await harness.run()).reason == "completed"
    first, *rest = harness.recorder.of(State)
    assert first.landed and len(rest) >= 3
    # As edits (no section of the State to edit) and then whole: no State in it, nothing lands.
    assert all((s.landed, s.dropped, s.text) == (0, "write_failed", "") for s in rest)
    entries = [e for e in harness.deps.journal.active_path() if isinstance(e, StateEntry)]
    assert [e.text for e in entries] == [FIRST + ASKED], "the junk never became the State"


async def test_a_new_request_since_the_state_is_brought_in_with_edits_and_the_state_keeps_the_last_three(
    make_harness,
):
    writes: list[ProviderRequest] = []
    calls: list[int] = []

    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            writes.append(request)
            return Turn(text="## Now\nreplace all: going on" if is_state_edit(request) else FIRST)
        calls.append(1)
        return Turn(text="done") if len(calls) % 4 == 0 else Turn(tool_calls=[("bash", {"command": output()})])

    harness = make_harness(script, window=WINDOW, system1=Opinionated())
    for request in ("print things", "now report them", "as a table", "and sort it"):
        assert (await harness.run(request)).reason == "completed"
    # No rule makes the write after a new request whole: every write after the first State is edits.
    assert len(writes) > 3 and all(is_state_edit(r) for r in writes[1:])
    # The State holds the last three requests, oldest first; not the first.
    last = harness.recorder.of(State)[-1]
    assert last.text.endswith(
        "## Facts\n- printf works\n\n" + requests_section(["now report them", "as a table", "and sort it"], 10_000)
    )
