"""Context projection, token metering and turn-atomic compaction planning."""

from __future__ import annotations

from intern.context import (
    STATE_PROMPT,
    ContextView,
    TokenMeter,
    blank_unseen_numbers,
    build_view,
    dropped_turns,
    estimate_tokens,
    fix_number,
    numbers_in,
    plan_compaction,
    requests_section,
    squeeze,
    state_request,
    uncovered_tokens,
    unseen_numbers,
    with_requests,
    without_code_sections,
)
from intern.instructions import Instructions, PinnedDoc
from intern.journal import CompactionEntry, Journal, MessageEntry, StateEntry
from intern.messages import (
    AssistantMessage,
    JsonObject,
    SystemMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from intern.models.profile import Thresholds
from intern.state_edits import parse

WRITTEN = "## Plan\n1. parse: in progress\n\n## Now\nparsing"


def tool_turn(n: int, size: int) -> list:
    call = ToolCall(id=f"c{n}", name="bash", arguments={"command": f"step {n}"})
    return [
        AssistantMessage(text=f"step {n}", tool_calls=[call], model="p/m"),
        ToolResultMessage(result=ToolResult(call_id=f"c{n}", tool_name="bash", status="ok", content="z" * size)),
    ]


def journal_with(messages: list) -> Journal:
    journal = Journal(cwd="/w")
    journal.append(MessageEntry(message=UserMessage(content="the task")))
    for m in messages:
        journal.append(MessageEntry(message=m))
    return journal


INSTRUCTIONS = Instructions(system="sys", pinned=[PinnedDoc(source="AGENTS.md", content="be careful")])


def test_view_puts_pinned_docs_first_and_the_users_request_in_the_history_and_merges_user_messages():
    journal = journal_with([*tool_turn(1, 10), UserMessage(content="nudge", source="harness")])
    view = build_view(journal.active_path(), INSTRUCTIONS, TokenMeter(), 32_768)
    # No message of the user's is pinned: the first is the history's first, like any later one.
    assert view.pinned == ["Instructions from AGENTS.md:\nbe careful"]
    assert view.history[0] == UserMessage(content="the task") and view.request == "the task"
    view.tail.append("budget note")
    messages = view.messages()
    assert [m.role for m in messages] == ["system", "user", "assistant", "tool", "user"]
    texts = [m.content for m in messages if isinstance(m, SystemMessage | UserMessage)]
    assert texts[0] == "sys"
    assert texts[1] == "Instructions from AGENTS.md:\nbe careful\n\nthe task"
    assert texts[-1] == "nudge\n\nbudget note"


def test_view_after_compaction_has_the_summary_and_only_kept_turns_and_the_first_request_is_one_of_them():
    journal = journal_with([*tool_turn(1, 10), *tool_turn(2, 10)])
    kept = [e for e in journal.active_path() if isinstance(e, MessageEntry)][3]
    journal.append(CompactionEntry(summary="did step 1", first_kept_id=kept.id, tokens_before=1, method="summary"))
    view = build_view(journal.active_path(), INSTRUCTIONS, TokenMeter(), 32_768)
    assert view.summary == "did step 1"
    assert [m.text for m in view.history if isinstance(m, AssistantMessage)] == ["step 2"]
    head = view.messages()[1]
    # The first request was cut with the turn it opened: what is left of it is what the State holds.
    assert isinstance(head, UserMessage) and "the task" not in head.content
    assert view.request == "the task"


def test_meter_calibrates_on_provider_usage():
    meter = TokenMeter()
    assert meter.tokens(3_000) == 1_000
    meter.observe(prompt_chars=8_000, prompt_tokens=2_000)
    assert meter.tokens(3_000) == 750


def test_compaction_plan_cuts_only_at_turn_boundaries():
    messages = [m for n in range(8) for m in tool_turn(n, 3_000)]
    view = build_view(journal_with(messages).active_path(), INSTRUCTIONS, TokenMeter(), 8_192)
    plan = plan_compaction(view, [], TokenMeter(), 8_192, Thresholds())
    assert plan is not None
    kept = view.history[len(plan.older) :]
    assert isinstance(kept[0], AssistantMessage)
    assert isinstance(plan.older[-1], ToolResultMessage)


def test_no_compaction_under_the_threshold():
    view = build_view(journal_with(tool_turn(1, 100)).active_path(), INSTRUCTIONS, TokenMeter(), 32_768)
    assert plan_compaction(view, [], TokenMeter(), 32_768, Thresholds()) is None


def test_squeeze_shortens_old_tool_output_in_the_view_only():
    journal = journal_with([*tool_turn(1, 30_000), *tool_turn(2, 30_000)])
    meter = TokenMeter()
    view = build_view(journal.active_path(), INSTRUCTIONS, meter, 16_384)
    shortened = squeeze(view, [], meter, 12_000)
    assert shortened == 1
    assert estimate_tokens(view, [], meter) <= 12_000
    original = [e.message for e in journal.active_path() if isinstance(e, MessageEntry)][2]
    assert isinstance(original, ToolResultMessage) and len(original.result.content) == 30_000


def test_a_state_entry_covers_the_turns_before_it_for_system_1_but_not_for_system_2():
    journal = journal_with([*tool_turn(1, 2_000), *tool_turn(2, 2_000)])
    last = [e for e in journal.active_path() if isinstance(e, MessageEntry)][-1]
    journal.append(StateEntry(text="## Plan\n- step 2 next", covers_through_id=last.id, tokens_before=1_300))
    for message in tool_turn(3, 2_000):
        journal.append(MessageEntry(message=message))
    view = build_view(journal.active_path(), INSTRUCTIONS, TokenMeter(), 32_768)
    assert view.summary is None and len(view.history) == 7, "System 2 still sees every turn, the request too"
    assert view.state == "## Plan\n- step 2 next" and view.covered == 5
    assert 0 < uncovered_tokens(view, TokenMeter()) < TokenMeter().tokens(4_100)


def test_compaction_that_keeps_nothing_still_shows_what_came_after_it():
    journal = journal_with([*tool_turn(1, 500), *tool_turn(2, 500)])
    journal.append(
        CompactionEntry(summary="## Plan\n- all done so far", first_kept_id=None, tokens_before=1, method="state")
    )
    for message in tool_turn(3, 500):
        journal.append(MessageEntry(message=message))
    view = build_view(journal.active_path(), INSTRUCTIONS, TokenMeter(), 32_768)
    assert view.summary == "## Plan\n- all done so far"
    assert [m.result.call_id for m in view.history if isinstance(m, ToolResultMessage)] == ["c3"]
    assert view.state == view.summary and view.covered == 0


def test_dropped_turns_lists_what_the_dropped_turns_did():
    text = dropped_turns("earlier", [*tool_turn(1, 5), *tool_turn(2, 5)])
    assert text.startswith("earlier")
    assert 'bash({"command": "step 2"})' in text


def quoted(section: str) -> list[str]:
    """The requests in a User requests section: its block quotes, a blank line apart, without the `>` marks."""
    blocks = [block.splitlines() for block in section.split("\n\n")]
    return ["\n".join(line.removeprefix(">").removeprefix(" ") for line in b) for b in blocks if b[0].startswith(">")]


def test_the_states_user_requests_are_the_last_three_as_written_oldest_first():
    asked = ["first ask", "second ask", "third ask:\nfix the parser\n\n## Spec\n- keep the API", "fourth ask"]
    state = with_requests(WRITTEN, asked, 10_000)
    assert state.startswith(f"{WRITTEN}\n\n## User requests\n")
    # Each request whole and as written, one block quote each: a list or a blank line in one cannot split it.
    assert quoted(state) == asked[1:]
    # A heading the user wrote stays in its request: the State parses into the writer's sections and one more.
    assert [s.title for s in parse(state).sections] == ["Plan", "Now", "User requests"]
    assert without_code_sections(state) == WRITTEN and without_code_sections(WRITTEN) is WRITTEN
    # A copy the writer made is replaced.
    replaced = with_requests(f"## User requests\n- my guess\n\n{WRITTEN}", ["the ask"], 99)
    assert replaced.startswith(f"{WRITTEN}\n\n## User requests") and quoted(replaced) == ["the ask"]


def test_the_newest_request_is_always_in_the_state_and_older_ones_only_whole_within_the_budget():
    newest = "n" * 50 + "\n" + "tail of the newest"
    # Over the budget alone: the newest is cut to fit, its end kept, and nothing older is there.
    cut = requests_section(["older", newest], 40)
    assert cut.startswith("## User requests\n") and "older" not in cut
    assert "characters omitted" in cut and cut.endswith("tail of the newest")
    # An older request that does not fit whole is left out, and so is every one before it.
    assert quoted(requests_section(["oldest", "x" * 100, "newest"], 50)) == ["newest"]
    assert quoted(requests_section(["oldest", "middle", "newest"], 50)) == ["oldest", "middle", "newest"]
    assert requests_section(["  ", ""], 50) == "" and with_requests(WRITTEN, [], 50) == WRITTEN


def test_state_request_asks_for_a_rewrite_of_the_previous_state():
    view = ContextView(system="s", pinned=["sum the files"], summary=None, history=[], history_ids=[])
    assert state_request(view).startswith(STATE_PROMPT)
    view.state = "## Plan\n- step 1: done\n\n## User requests\n- sum the files"
    request = state_request(view)
    # The user's requests are code's to keep: the writer gets the rest of the State.
    assert request.startswith(STATE_PROMPT) and request.endswith("## Plan\n- step 1: done")
    # The State the agent already has at the top of its context is referred to, not repeated.
    view.summary = view.state
    assert "## Plan\n- step 1" not in state_request(view)


def test_numbers_the_conversation_never_showed_are_found_and_blanked():
    conversation = [
        UserMessage(content="Review FY2025."),
        AssistantMessage(
            text="Net sales are 4,233,003.", reasoning="margin 36.25%", tool_calls=[ToolCall(id="c", name="read")]
        ),
        ToolResultMessage(
            result=ToolResult(
                call_id="c",
                tool_name="read",
                status="ok",
                content="Receivables,538791,481979\nCapex,(126974)\nRows to update: 120, 126, 131; capex 125; 135 days",
            )
        ),
    ]
    seen = numbers_in(conversation)
    # The same number written another way is seen; a CSV row's comma separates two numbers.
    state = "Sales 4233003 (FY2025); margin 36.250%; AR 538,791/481,979; capex 126,974; FCF 534,199; AR 553,315."
    assert unseen_numbers(state, seen) == ["534,199", "553,315"]
    assert blank_unseen_numbers(state, seen).endswith("capex 126,974; FCF ?; AR ?.")
    # A list the run showed, packed together, is seen (a subset too); numbers from all over the run are not.
    assert unseen_numbers("Rows 120,126,131 and 131,120; capex 125,135", seen) == ["125,135"]


def test_a_list_items_own_number_is_structure_not_a_value_to_check():
    seen = numbers_in([UserMessage(content="Count the rows.")])  # no number at all
    state = "## Plan\n1. count\n7. report 777 rows\n   2) sub-step\n## Facts\n- 3 tables\n12. x"
    assert unseen_numbers(state, seen) == ["3", "777"]  # the items' 1., 7., 2) and 12. are not values
    assert blank_unseen_numbers(state, seen) == state.replace("777", "?").replace("- 3 tables", "- ? tables")


def test_a_number_fix_changes_only_a_line_that_holds_a_flagged_number():
    def edit(**args: str) -> ToolCall:
        arguments: JsonObject = {"path": "STATE.md", **args}
        return ToolCall(id="c", name="edit", arguments=arguments)

    state = "## Facts\n- AR 553,315\n- AR note: 553,315 is unverified\n- cash 308,789"
    fixed, error = fix_number(state, edit(old_string="- AR 553,315", new_string="- AR 538,791"), ["553,315"])
    assert error is None and fixed == state.replace("- AR 553,315", "- AR 538,791")
    for bad in (
        ToolCall(id="c", name="write", arguments={"path": "STATE.md", "content": "## Facts\n- new"}),
        edit(old_string="- cash 308,789", new_string="- cash 1"),  # no flagged number on the line
        edit(old_string="- AR 553,315\n- AR note", new_string="x"),  # more than one line
        edit(old_string="553,315", new_string="538,791"),  # ambiguous
    ):
        text, error = fix_number(state, bad, ["553,315"])
        assert text == state and error


def test_view_type_is_plain_data():
    view = ContextView(system="s", pinned=[], summary=None, history=[], history_ids=[])
    assert [m.role for m in view.messages()] == ["system"]
