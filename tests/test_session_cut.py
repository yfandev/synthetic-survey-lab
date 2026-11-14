"""System 1's State as a cut of the session (`session_cut`): within its budget whatever the session's length, the newest
turns and the user's requests kept, a tool's output only as a note: its size, a check's tally and its first lines."""

from __future__ import annotations

import json
from typing import Literal

from intern.journal import MessageEntry
from intern.messages import AssistantMessage, ToolCall, ToolResult, ToolResultMessage, UserMessage
from intern.session_cut import cut_view, estimate_tokens, session_cut


def turn(
    n: int,
    text: str = "",
    output: str = "ok",
    status: Literal["ok", "error", "refused"] = "ok",
    command: str | None = None,
):
    call = ToolCall(id=f"c{n}", name="bash", arguments={"command": command or f"python step.py --n {n}"})
    return [
        MessageEntry(message=AssistantMessage(text=text, reasoning="private thoughts", tool_calls=[call])),
        MessageEntry(
            message=ToolResultMessage(
                result=ToolResult(call_id=call.id, tool_name="bash", status=status, content=output)
            )
        ),
    ]


OUTPUT = "first line\n" + "filler " * 100 + "LATE-DETAIL"


def session(turns: int, text_chars: int = 2_000) -> list:
    entries = [MessageEntry(message=UserMessage(content="fix the parser"))]
    for n in range(1, turns + 1):
        if n == turns // 2:
            entries.append(MessageEntry(message=UserMessage(content="and keep the old API")))
            entries.append(MessageEntry(message=UserMessage(content="Stop investigating.", source="harness")))
        entries += turn(n, f"turn {n}: " + "w " * (text_chars // 2), output=OUTPUT)
    return entries


def test_a_session_that_fits_is_given_whole_in_order_without_reasoning_or_whole_outputs():
    state = json.loads(session_cut(session(10), 25_000))
    assert state["goal"] == "fix the parser\nand keep the old API", "the user's prompts, not the harness's"
    history = state["history"]
    assert [e["i"] for e in history] == sorted(e["i"] for e in history)
    assert sum(e["role"] == "assistant" for e in history) == 10 and all("w w" in e["text"] for e in history[-3:])
    dumped = json.dumps(state)
    assert "LATE-DETAIL" not in dumped and "private thoughts" not in dumped
    note = history[-1]["tool_calls"][0]["result"]
    assert note.startswith(f'ok, {len(OUTPUT)} chars (omitted): starts "first line filler filler') and note.endswith(
        '…"'
    )


def test_a_long_session_is_cut_to_the_budget_keeping_the_first_message_and_the_newest_turns_whole():
    for turns, budget in ((60, 6_000), (400, 6_000)):  # the second is past every fast-jev stage
        cut = session_cut(session(turns), budget)
        assert estimate_tokens(cut) <= budget
        history = json.loads(cut)["history"]
        assert history[0] == {"i": 0, "role": "user", "text": "fix the parser"}
        newest = [e for e in history if e.get("role") == "assistant"][-3:]
        assert [e["text"].split(":")[0] for e in newest] == [f"turn {n}" for n in range(turns - 2, turns + 1)]
        assert all(isinstance(e["tool_calls"][0], dict) for e in newest), "the newest calls keep their form"
    assert any("earlier entries left out" in e["text"] for e in history)


def test_a_checks_note_gives_its_tally_before_the_outputs_first_lines():
    entries = [
        MessageEntry(message=UserMessage(content="make the tests pass")),
        *turn(
            1,
            output="....F\nE   AssertionError: 3 != 4\n1 failed, 4 passed in 0.12s\n",
            status="error",
            command="pytest -q",
        ),
        *turn(2, output=".....\n5 passed in 0.10s\n", command="uv run pytest -q tests"),
        *turn(3, output="a" * 900, command="cat notes.txt"),
    ]
    notes = [e["tool_calls"][0]["result"] for e in json.loads(session_cut(entries, 25_000))["history"][1:]]
    # The tally sits at the end of a long log, past what its first lines show.
    assert (
        notes[0].startswith("error, ") and ': 1 failed, 4 passed in 0.12s; starts "....F E AssertionError' in notes[0]
    )
    assert ": 5 passed in 0.10s; starts " in notes[1]
    assert notes[2].startswith('ok, 900 chars (omitted): starts "aaa'), "a read is not a check: no tally"


def test_the_tuis_view_of_the_cut_is_the_text_system_1_reads_and_says_how_each_message_was_kept():
    seen = set()
    for turns, budget in ((10, 25_000), (30, 6_000), (60, 6_000), (400, 6_000)):
        entries = session(turns)
        view = cut_view(entries, budget)
        assert view.text == session_cut(entries, budget)
        assert len(view.messages) == turns + 3, "every prompt, reply and harness message, kept or not"
        history = json.loads(view.text)["history"]
        shown = {e["i"]: e for e in history if "i" in e}
        calls = {
            c["id"] if isinstance(c, dict) else c.split(" ")[0]: c for e in history for c in e.get("tool_calls", [])
        }
        for m in view.messages:
            seen.add(m.kept)
            if m.kept == "left_out":
                assert m.index not in shown and not any(c.id in calls for c in m.calls)
            elif m.kept == "line":
                assert all(isinstance(calls[c.id], str) for c in m.calls)
            else:
                assert shown[m.index]["text"] == m.text and all(isinstance(calls[c.id], dict) for c in m.calls)
                said = {"whole": len(m.text) == m.chars, "collapsed": m.text == f"[… {m.chars} chars omitted …]"}
                assert said.get(m.kept, m.chars > len(m.text) and "chars omitted" in m.text), m.kept
        newest = view.messages[-3:]
        assert all(m.pinned and m.kept in ("whole", "abridged") for m in newest), "the newest are only abridged"
    assert seen == {"whole", "abridged", "collapsed", "line", "left_out"}
