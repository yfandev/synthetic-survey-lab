"""The State as sections of numbered items, and the edits a writer sends instead of the whole State."""

from __future__ import annotations

import pytest

from intern.state_edits import (
    EditError,
    apply_edits,
    is_no_change,
    is_whole,
    parse,
    render,
)

STATE = """\
## Plan
1. Reproduce the failure. **Done:** `pytest -q` fails on test_add.
2. Fix `add` in calc.py. **In progress.**
   - the bug: `a - b` instead of `a + b`
3. Run the tests. **Pending.**

## Done when
`pytest -q` passes.

## Now
Editing calc.py line 2.

## Artifacts
- calc.py: the function under test
- test_calc.py: the failing test

## Facts
- add(2, 3) returns -1
- the tests run with `python3 -m pytest -q`"""


def test_a_state_parses_into_sections_of_items_and_renders_back_as_written():
    doc = parse(STATE)
    assert [(s.title, len(s.items)) for s in doc.sections] == [
        ("Plan", 3),
        ("Done when", 1),
        ("Now", 1),
        ("Artifacts", 2),
        ("Facts", 2),
    ]
    plan = doc.section("plan")
    assert plan is not None and plan.items[1].text.endswith("- the bug: `a - b` instead of `a + b`")  # lines under it
    assert render(doc) == STATE and doc.editable()
    numbered = render(doc, numbered=True)
    assert "## Artifacts\n1. calc.py: the function under test\n2. test_calc.py" in numbered
    assert "## Now\n1. Editing calc.py line 2." in numbered


def test_edits_change_only_the_items_they_name():
    reply = """\
## Plan
set 2: Fix `add` in calc.py. **Done:** `a + b` now.
set 3: Run the tests. **In progress.**
add: Report the fix.
## Facts
drop: 1
add: the fix is on line 2
## Now
replace: Running `python3 -m pytest -q`."""
    doc = apply_edits(parse(STATE), reply)
    plan = doc.section("Plan")
    assert plan is not None and [i.text.split(".")[0] for i in plan.items] == [
        "Reproduce the failure",
        "Fix `add` in calc",
        "Run the tests",
        "Report the fix",
    ]
    out = render(doc)
    assert "4. Report the fix." in out and "## Now\nRunning `python3 -m pytest -q`." in out
    assert "## Facts\n- the tests run with `python3 -m pytest -q`\n- the fix is on line 2" in out
    # Untouched sections are exactly as they were.
    for title in ("Done when", "Artifacts"):
        assert render(parse(STATE)).split(f"## {title}")[1].split("##")[0] in out


def test_item_numbers_are_those_the_writer_saw_and_a_numbered_list_is_renumbered():
    doc = apply_edits(parse(STATE), "## Plan\ndrop: 1\nset 3: Run the tests. **Done.**")
    assert render(doc).startswith("## Plan\n1. Fix `add` in calc.py. **In progress.**")
    assert "2. Run the tests. **Done.**" in render(doc)


@pytest.mark.parametrize(
    "reply",
    [
        "```\n## Facts\nadd: a third fact\n```",  # fenced
        "Here are the edits:\n\n## Facts\nadd: a third fact",  # a preamble
        "## facts\n- add: a third fact",  # a bullet before the edit, the heading in lower case
    ],
)
def test_the_usual_wrappings_of_edits_are_read_as_edits(reply):
    out = render(apply_edits(parse(STATE), reply))
    assert out.endswith("- the tests run with `python3 -m pytest -q`\n- a third fact")


def test_an_edits_text_goes_on_over_the_next_lines_and_add_alone_takes_a_list():
    reply = "## Plan\nset 3: Run the tests. **Done.**\n   - 4 passed\n## Facts\nadd:\n- one\n- two\ndrop: 1-2"
    out = render(apply_edits(parse(STATE), reply))
    assert "3. Run the tests. **Done.**\n   - 4 passed" in out
    assert out.endswith("## Facts\n- one\n- two")


def test_a_section_written_out_without_edits_is_its_new_text_unless_it_says_unchanged():
    out = render(apply_edits(parse(STATE), "## Now\n1. Running the tests.\n2. Then the report.\n## Facts\n(unchanged)"))
    assert "## Now\n1. Running the tests.\n2. Then the report.\n\n## Artifacts" in out
    assert out.endswith(STATE.split("## Facts")[1].join(["## Facts", ""]))


def test_slips_seen_in_replays_are_read_as_the_edit_meant():
    out = render(apply_edits(parse(STATE), "## Plan\nreplace 3: Run the tests. **Done.**\nset 1. Reproduced."))
    assert out.startswith("## Plan\n1. Reproduced.\n2. Fix") and "3. Run the tests. **Done.**" in out


def test_replace_all_without_a_heading_of_its_own_is_nows_when_the_reply_has_no_now():
    now = "## Now\nRunning the tests.\n\n## Artifacts"
    assert now in render(apply_edits(parse(STATE), "replace all: Running the tests."))  # before any heading
    after_items = "## Plan\nset 2: Fix `add`. **Done.**\nreplace all: Running the tests."  # Now's heading left out
    out = render(apply_edits(parse(STATE), after_items))
    assert "2. Fix `add`. **Done.**" in out and now in out
    with pytest.raises(EditError, match="Plan replaced and edited at once"):  # the reply has a Now: it is Plan's
        apply_edits(parse(STATE), after_items + "\n## Now\nreplace all: Then the report.")


def test_an_add_with_a_number_is_an_add():
    assert render(apply_edits(parse(STATE), "## Facts\nadd 3: a third fact")).endswith("\n- a third fact")


def test_a_section_the_state_lacks_is_added_in_its_place():
    doc = apply_edits(parse("## Plan\n1. a\n\n## Facts\n- f"), "## Now\nreplace: doing a")
    assert [s.title for s in doc.sections] == ["Plan", "Now", "Facts"]


@pytest.mark.parametrize(
    ("reply", "why"),
    [
        ("## Plan\nset 7: nothing there", "Plan has no item 7"),
        ("## Facts\ndrop: 3", "Facts has no item 3"),
        ("## Plan\nThe parser is fixed.\nset 1: a", "Plan replaced and edited at once"),
        ("set 1: before any heading", "an edit before any section heading"),
        ("## Risks\nadd: one", "no section `Risks`"),
        ("## Plan\nset 1: a\ndrop: 1", "item 1 of Plan edited twice"),
        ("## Now\nreplace: x\n## Now\nreplace: y", "Now replaced twice"),
        ("## Now\nreplace: x\nadd: y", "Now replaced and edited at once"),
        ("## Plan\nmodify 2: done", "not an edit this format knows"),
        ("## Plan\nset 1: a\nreplace all 8: b", "not an edit this format knows"),  # garbled, not item 1's text
        ("## Artifacts\nadd: x\nset: 4. y", "not an edit this format knows"),
        ("## Plan\nset 2:", "item 2 of Plan set to nothing"),
        ("nothing to say", "no section heading"),
    ],
)
def test_edits_that_cannot_be_applied_are_refused_as_a_whole(reply, why):
    with pytest.raises(EditError, match=why):
        apply_edits(parse(STATE), reply)


def test_a_whole_state_and_no_change_are_told_from_edits():
    assert is_whole(STATE) and not is_whole("## Plan\nset 1: x\n## Now\nreplace: y")
    assert not is_whole("## Plan\n1. a\n## Now\nb")  # two sections given in full: neither edits nor a State
    assert is_no_change("No change.") and is_no_change("no changes") and not is_no_change("## Now\nno change")


def test_a_state_without_its_sections_is_not_edited():
    assert not parse("Earlier turns were removed to fit the window. Tool calls they made:\n- bash(...)").editable()
    assert not parse("## Plan\n1. a").editable() and parse("## Plan\n1. a\n## Now\nb").editable()
