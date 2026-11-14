"""Repairs for what small models get wrong on the wire: malformed tool arguments, tool calls written
as text, and reasoning inlined in content between think tags."""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass

from intern.messages import JsonObject

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_PY_LITERALS = {"True": "true", "False": "false", "None": "null"}


def _fix_outside_strings(text: str) -> str:
    """Drop trailing commas and turn Python literals into JSON ones, never touching string contents."""
    out: list[str] = []
    i, in_string = 0, False
    while i < len(text):
        char = text[i]
        if in_string:
            out.append(char)
            if char == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 1
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            out.append(char)
        elif char == ",":
            rest = text[i + 1 :].lstrip()
            if not rest.startswith(("}", "]")):
                out.append(char)
        else:
            literal = next((lit for lit in _PY_LITERALS if text.startswith(lit, i)), None)
            boundary = literal and not (i and (text[i - 1].isalnum() or text[i - 1] == "_"))
            if literal and boundary:
                out.append(_PY_LITERALS[literal])
                i += len(literal)
                continue
            out.append(char)
        i += 1
    return "".join(out)


def _candidates(text: str) -> list[str]:
    found = [text]
    fenced = _FENCE.match(text)
    if fenced:
        found.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        found.append(text[start : end + 1])
    return found + [_fix_outside_strings(candidate) for candidate in found]


def parse_arguments(raw: str) -> tuple[JsonObject | None, str | None]:
    """Parse tool-call arguments, repairing common damage. Returns (arguments, error)."""
    text = raw.strip()
    if not text:
        return {}, None
    first_error = ""
    for candidate in _candidates(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as e:
            first_error = first_error or str(e)
            continue
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(value)
        if isinstance(value, dict):
            return value, None  # pyright: ignore[reportUnknownVariableType]
        return None, f"arguments must be a JSON object, got {type(value).__name__}"
    return None, f"arguments are not valid JSON ({first_error})"


@dataclass
class TextToolCall:
    name: str
    arguments: JsonObject | None
    raw: str
    error: str | None


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_XML_FUNCTION = re.compile(r"<function=([\w.\-]+)>(.*?)</function>", re.DOTALL)
_XML_PARAMETER = re.compile(r"<parameter=([\w.\-]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _parse_block(block: str) -> TextToolCall | None:
    xml = _XML_FUNCTION.search(block)
    if xml:
        # Values stay strings; each tool's schema coerces them (e.g. "30" to an int) when it parses.
        params: JsonObject = dict(_XML_PARAMETER.findall(xml.group(2)))
        return TextToolCall(name=xml.group(1), arguments=params, raw=block, error=None)
    payload, error = parse_arguments(block)
    if payload is None:
        return TextToolCall(name="", arguments=None, raw=block, error=error)
    name = payload.get("name")
    if not isinstance(name, str):
        return None
    arguments = payload.get("arguments", payload.get("parameters", {}))
    if isinstance(arguments, str):
        parsed, error = parse_arguments(arguments)
        return TextToolCall(name=name, arguments=parsed, raw=block, error=error)
    if not isinstance(arguments, dict):
        return TextToolCall(name=name, arguments=None, raw=block, error="arguments must be a JSON object")
    return TextToolCall(name=name, arguments=arguments, raw=block, error=None)  # pyright: ignore[reportUnknownArgumentType]


def extract_text_tool_calls(text: str) -> tuple[str, list[TextToolCall]]:
    """Pull `<tool_call>` blocks out of reply text. Returns the remaining text and the calls found."""
    calls = [call for block in _TOOL_CALL_BLOCK.findall(text) if (call := _parse_block(block))]
    if not calls:
        return text, []
    return _TOOL_CALL_BLOCK.sub("", text).strip(), calls


class ThinkSplitter:
    """Routes streamed content between `<think>` and `</think>` to reasoning, across chunk boundaries."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._thinking = False

    def feed(self, chunk: str) -> list[tuple[bool, str]]:
        """Returns (is_reasoning, text) pieces that are safe to emit now."""
        self._buffer += chunk
        pieces: list[tuple[bool, str]] = []
        while True:
            tag = self.CLOSE if self._thinking else self.OPEN
            index = self._buffer.find(tag)
            if index >= 0:
                if index:
                    pieces.append((self._thinking, self._buffer[:index]))
                self._buffer = self._buffer[index + len(tag) :]
                self._thinking = not self._thinking
                continue
            keep = _partial_suffix(self._buffer, tag)
            emit = self._buffer[: len(self._buffer) - keep]
            if emit:
                pieces.append((self._thinking, emit))
            self._buffer = self._buffer[len(emit) :]
            return pieces

    def finish(self) -> list[tuple[bool, str]]:
        rest, self._buffer = self._buffer, ""
        return [(self._thinking, rest)] if rest else []


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `tag`."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


def split_unopened_think(text: str) -> tuple[str | None, str]:
    """Some chat templates open the think block in the prompt, so the reply has only `</think>`."""
    head, sep, tail = text.partition(ThinkSplitter.CLOSE)
    if not sep:
        return None, text
    return head.strip() or None, tail.lstrip()
