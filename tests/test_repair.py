from __future__ import annotations

import pytest

from intern.providers.repair import ThinkSplitter, extract_text_tool_calls, parse_arguments, split_unopened_think


@pytest.mark.parametrize(
    "raw",
    [
        '{"path": "a.py"}',
        '```json\n{"path": "a.py"}\n```',
        'Here you go: {"path": "a.py"} hope that helps',
        '{"path": "a.py",}',
        '"{\\"path\\": \\"a.py\\"}"',
    ],
)
def test_damaged_arguments_are_repaired(raw: str):
    assert parse_arguments(raw) == ({"path": "a.py"}, None)


def test_python_literals_are_repaired():
    assert parse_arguments('{"all": True, "limit": None}') == ({"all": True, "limit": None}, None)


def test_empty_arguments_mean_no_arguments():
    assert parse_arguments("  ") == ({}, None)


@pytest.mark.parametrize("raw", ['{"path": "a.py"', "[1, 2]", "not json"])
def test_unrepairable_arguments_report_an_error(raw: str):
    arguments, error = parse_arguments(raw)
    assert arguments is None
    assert error


def test_json_tool_calls_written_as_text_are_recovered():
    text, calls = extract_text_tool_calls(
        'Let me look.\n<tool_call>\n{"name": "read", "arguments": {"path": "x.py"}}\n</tool_call>'
    )
    assert text == "Let me look."
    assert [(c.name, c.arguments) for c in calls] == [("read", {"path": "x.py"})]


def test_xml_tool_calls_written_as_text_are_recovered_with_raw_string_values():
    _, calls = extract_text_tool_calls(
        "<tool_call>\n<function=write>\n<parameter=path>\nconf.json\n</parameter>\n"
        '<parameter=content>\n{"a": 1}\n</parameter>\n</function>\n</tool_call>'
    )
    assert [(c.name, c.arguments) for c in calls] == [("write", {"path": "conf.json", "content": '{"a": 1}'})]


def test_repairs_never_change_text_inside_strings():
    raw = '{"content": "flag = True\\nx = None, ]", "all": True,}'
    assert parse_arguments(raw) == ({"content": "flag = True\nx = None, ]", "all": True}, None)


def test_think_tags_are_split_across_chunk_boundaries():
    splitter = ThinkSplitter()
    pieces = []
    for chunk in ["<thi", "nk>plan it</th", "ink>The answer", " is 4.<", "b>"]:
        pieces += splitter.feed(chunk)
    pieces += splitter.finish()
    reasoning = "".join(t for r, t in pieces if r)
    text = "".join(t for r, t in pieces if not r)
    assert (reasoning, text) == ("plan it", "The answer is 4.<b>")


def test_reply_with_only_a_closing_think_tag():
    assert split_unopened_think("thinking...</think>\n\nAnswer") == ("thinking...", "Answer")
    assert split_unopened_think("plain") == (None, "plain")
