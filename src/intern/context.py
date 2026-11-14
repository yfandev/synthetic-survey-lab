"""Builds each request from the journal.

A request has four segments, in this order so the provider's prompt cache keeps hitting:

- system: the system prompt; byte-stable for the whole run.
- pinned: instruction files and the recalled memory block; never pruned.
- history: everything since the last compaction, the user's messages included; pruned only in whole turns.
  Beside it stands the State, which System 2 writes at a compaction for the turns the cut drops.
- tail: per-turn additions from the engine and extensions; always last.

Token counts come from provider usage. Text the provider has not seen yet is converted with the
characters-per-token ratio measured on the previous request, never with a fixed guess once a request
has been answered.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from intern.events import ProcessInfo, now_ms
from intern.instructions import Instructions
from intern.journal import CompactionEntry, Entry, Kept, MemoryRecallEntry, MessageEntry, StateEntry
from intern.messages import (
    AssistantMessage,
    ImagePart,
    Message,
    SystemMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from intern.models.profile import Thresholds
from intern.processes import duration_text
from intern.providers.base import ProviderRequest, ToolSpec
from intern.state_edits import (
    MAX_EDITED_WORDS,
    EditError,
    StateDoc,
    apply_edits,
    heading,
    is_no_change,
    is_whole,
    words,
)
from intern.state_edits import parse as parse_state
from intern.state_edits import render as render_state

INITIAL_CHARS_PER_TOKEN = 3.0
MESSAGE_OVERHEAD_CHARS = 12
PINNED_SHARE = 0.10
SQUEEZED_RESULT_CHARS = 400
BOUNDED_HEAD_SHARE = 0.3
PIXELS_PER_IMAGE_TOKEN = 750

# The State is a brief for further execution (arXiv 2611.05306, appendix A): it replaces the compacted
# turns in the agent's context, so it says what to do next rather than what happened. No user message is pinned;
# the latest requests are a section code keeps (`REQUESTS_SECTION`). The agent's own session writes it: the
# request it would send next, with this appended, so the writer sees all the agent sees, reasoning included.
# Seeing all of it, Qwen3.8 wrote States of 15-18K characters unbounded; with the word bound 6-10K, as before.
# What each section holds, for a State written whole and for edits alike (the edit request shows only the State).
# Artifacts names what the task produced or changed, not what was read or what a file contains: blind judges found
# the invented details (helper names, line numbers) there; Plan is a line a step, its results in Facts, which
# Plan had restated; a section of the writer's own drifted from these.
STATE_SECTIONS = (
    ("Plan", "The task broken into steps, one line each, marked done, in progress or pending."),
    ("Done when", "How to tell the task is complete: what to check or run."),
    ("Now", "What is in progress, why, and the next action to take."),
    (
        "Artifacts",
        "What this task produced or changed, one line each: a file, branch, commit, PR, issue or URL, and in a "
        "few words what it is for. Not files it only read, and not what a file contains.",
    ),
    (
        "Facts",
        "Values, errors, decisions and constraints found so far that will matter later, what each step found or "
        "produced included.",
    ),
)
_SECTIONS_TEXT = "\n".join(f"## {title}\n{what}" for title, what in STATE_SECTIONS)
_SECTIONS_LIST = "\n".join(f"- {title}: {what}" for title, what in STATE_SECTIONS)
# The last section of every State, which code fills and the writer never sees to edit (`with_requests`). Codex's
# compaction keeps the latest user messages verbatim within a budget; Hermes keeps the latest one raw since a summary
# that held it made agents stall or drop the request, and stopped pinning the first ones after a compaction because
# they grew without end. A first message pinned for good (as here before) left a later request to the writer's
# paraphrase.
REQUESTS_SECTION = ("User requests", "The user's latest messages as they wrote them, oldest first.")
MAX_REQUESTS = 3
# Code also sets the commands still running when the State is written, before the requests: the turn that started a
# job may be among those the State replaces, and the agent would otherwise lose the job's handle.
PROCESSES_SECTION = ("Running processes", "Commands still running when this state was written.")
MAX_PROCESSES = 8

STATE_PROMPT = f"""\
Pause the task: do not call any tool and do not continue the work in this reply. Write the state of the \
work so far, so that an agent that sees only this state can carry on. Reply with the state \
only, in these sections and no others:

{_SECTIONS_TEXT}

The harness adds sections of its own; leave them out. {REQUESTS_SECTION[0]}: {REQUESTS_SECTION[1]} And while \
commands run, {PROCESSES_SECTION[0]}: {PROCESSES_SECTION[1]}

When a previous state is given, rewrite it: update what changed, drop what no longer matters, do not \
append. Be specific and brief: at most about 1,000 words in all. Do not invent anything that is not in \
this conversation. Copy every number exactly as it appears in this conversation; do not compute, round, \
convert or estimate one yourself: a figure the work has not produced yet is a step to take, not a fact."""

# Once there is a State, it is brought up to date with edits, not written again: a whole State is
# 2-4K output tokens and 16-35 s a write, mostly re-emitting what did not change, and rewording it each time lost
# details no one dropped. Items are named by their number in the section, so no edit copies a line exactly
# (`state_edits.py` parses and applies them). The State it edits is given here, numbered, at the end of the request,
# so the prompt prefix the provider caches stays the agent's; edit tools would change it from the first token.
STATE_EDIT_PROMPT = f"""\
Pause the task: do not call any tool and do not continue the work in this reply. Bring the state of the work \
below up to date with what happened since it was written, so that an agent that sees only the state can \
carry on. Do not write the state again: reply with edits only, under the heading of each section \
that changes, one edit per line:

## <section>
set <n>: <item n's new text>
add: <a new item>
drop: <n>
replace all: <the whole section's new text>

<n> is the item's number in the state below. Mark a plan step done or in progress with set; add to Facts what \
the turns found or produced that later steps will need, and to Artifacts what they produced or changed; drop \
what no longer matters; replace all of Now with what is in progress and the next action. Start the edits of \
every section with its heading. A section gets either replace all or set, add and drop, never both: one item \
changes with set. An item's text may go on over the next lines. Leave out every section that does not change. \
Do not invent anything that is not in this conversation. Copy every number exactly as it appears in this \
conversation; do not compute, round, convert or estimate one yourself. If nothing changed, reply: no change.

What each section holds:
{_SECTIONS_LIST}

{{size}}

State, every item numbered within its section:
{{state}}"""

# The State's size and its limit, in every edit request, so the edits keep it under.
STATE_SIZE = """\
The state has {words} words. Keep it under {limit}: when it would grow past that, drop or shorten items the rest \
of the task no longer needs."""

# An edit that leaves the State past the limit gets one more reply in the same fork: edits that only trim it. Naming
# how much to cut matters: asked only to get under, Opus cut 3-14 times what was over.
STATE_TRIM_PROMPT = """\
These edits leave the State at {words} words, over the limit of {limit}. Reply with edits on the State below, in \
the same format, that bring it to about {goal} words: cut about {cut} words, not more. Drop or shorten (with `set`) \
the items the rest of the task needs least, such as details of finished steps and facts nothing ahead uses. Keep \
everything the remaining work needs. Change nothing else.

The State after your edits, every item numbered within its section:
{state}"""

# When the State gives numbers the conversation never showed, it becomes this file for the writer to fix them.
STATE_FILE = "STATE.md"
STATE_NUMBERS_PROMPT = """\
These numbers in your state do not appear anywhere in this conversation: {numbers}. Your state is now the file \
STATE.md, exactly as you wrote it above, so there is no need to read it. Fix each number with one edit call on \
STATE.md, all in this reply: old_string is the line that holds it, new_string that line with the number copied \
exactly from where it appears in this conversation, or without it if it does not appear. Change nothing else."""
# The same, after edits: the writer never wrote the State out, so it is shown.
STATE_NUMBERS_EDITED_PROMPT = """\
Your edits are applied. These numbers in the state do not appear anywhere in this conversation: {numbers}. The \
state is now the file STATE.md, as follows, so there is no need to read it:

{state}

Fix each number with one edit call on STATE.md, all in this reply: old_string is the line that holds it, \
new_string that line with the number copied exactly from where it appears in this conversation, or without it \
if it does not appear. Change nothing else."""
# The fix, then one retry for numbers left (it often fixes one per reply); a third reply added only latency.
MAX_NUMBER_FIX_REPLIES = 2

# A number as written: 1,234,567.8 or 1234.5. A comma only groups thousands, so a CSV row's 538791,481979 is two.
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?")
# A list of numbers as written: 120, 126, 131. Packed into a State it reads 120,126 and looks like one number.
# A list item's number at the start of a line (`2. report`): structure, not a value the run showed, so the number
# check leaves it; otherwise a plan's `2.` was blanked to `?.` when no `2` stood alone in the conversation.
LIST_MARK = r"^[ \t]*\d{1,3}[.)](?=[ \t])"
_MARK_OR_NUMBER = re.compile(rf"(?P<mark>{LIST_MARK})|{NUMBER.pattern}", re.MULTILINE)
NUMBER_LIST = re.compile(r"\d+(?:\.\d+)?(?:,\s+\d+(?:\.\d+)?)+")


class TokenMeter:
    """Converts characters to tokens using the ratio the provider last reported."""

    def __init__(self) -> None:
        self.chars_per_token = INITIAL_CHARS_PER_TOKEN
        self.calibrated = False

    def observe(self, prompt_chars: int, prompt_tokens: int) -> None:
        if prompt_tokens > 0 and prompt_chars > 0:
            self.chars_per_token = prompt_chars / prompt_tokens
            self.calibrated = True

    def tokens(self, chars: int) -> int:
        return math.ceil(chars / self.chars_per_token)


def image_chars(image: ImagePart) -> int:
    """An image's prompt cost in character units: about one token per 750 pixels on Claude and GPT
    vision models, at the starting characters-per-token ratio."""
    return math.ceil(image.width * image.height / PIXELS_PER_IMAGE_TOKEN * INITIAL_CHARS_PER_TOKEN)


def message_chars(message: Message) -> int:
    match message:
        case SystemMessage(content=text) | UserMessage(content=text):
            size = len(text)
        case AssistantMessage():
            size = len(message.text) + sum(len(c.name) + len(json.dumps(c.arguments)) for c in message.tool_calls)
        case ToolResultMessage(result=result):
            size = len(result.content) + sum(image_chars(image) for image in result.images)
    return size + MESSAGE_OVERHEAD_CHARS


def reasoning_chars(message: Message, model: str | None, replay: bool) -> int:
    """The reasoning a request carries back to the model that wrote ``message``: signed blocks always, the text
    when the profile replays it. Not in `message_chars`: it rides along only for the model's own messages."""
    if not isinstance(message, AssistantMessage) or model is None or message.model != model:
        return 0
    blocks = len(json.dumps(message.reasoning_blocks)) if message.reasoning_blocks else 0
    return blocks + (len(message.reasoning or "") if replay else 0)


def tools_chars(tools: list[ToolSpec]) -> int:
    return len(json.dumps([t.model_dump() for t in tools]))


@dataclass
class ContextView:
    """The request under construction. The `context` hook may change any field."""

    system: str
    pinned: list[str]
    summary: str | None
    history: list[Message]
    history_ids: list[str]
    tail: list[str] = field(default_factory=list)
    # The user's latest message: what the current run answers.
    request: str | None = None
    # The latest State, written at a compaction (then it is `summary` too); `covered` leading history messages are
    # in it. System 2 reads `summary`, and the two agree once compaction has cut the covered turns.
    state: str | None = None
    covered: int = 0
    # Reads the last compaction kept verbatim (System 1's choice): shown after the State.
    kept: list[Kept] = field(default_factory=list)
    # The model whose own messages carry their reasoning back (`reasoning_chars`), and whether the text replays:
    # measured with the history, so compaction plans for what the request will send.
    model: str | None = None
    replay: bool = False

    def messages(self) -> list[Message]:
        """Render the segments, merging adjacent user messages so strict chat templates accept them."""
        head = "\n\n".join(self.pinned)
        if self.summary:
            head = f"{head}\n\nState of the work so far (earlier turns were compacted):\n{self.summary}"
        if self.kept:
            files = "\n\n".join(f"{k.path}:\n{k.content}" for k in self.kept)
            head = f"{head}\n\nFiles read earlier, kept verbatim because the current step works on them:\n{files}"
        rendered: list[Message] = [SystemMessage(content=self.system)]
        if head:
            rendered.append(UserMessage(content=head))
        rendered += self.history
        if self.tail:
            rendered.append(UserMessage(content="\n\n".join(self.tail), source="harness"))

        merged: list[Message] = []
        for message in rendered:
            previous = merged[-1] if merged else None
            if isinstance(message, UserMessage) and isinstance(previous, UserMessage):
                merged[-1] = UserMessage(content=f"{previous.content}\n\n{message.content}", source=previous.source)
            else:
                merged.append(message)
        return merged

    def segment_chars(self) -> dict[str, int]:
        summary = len(self.summary or "")
        return {
            "system": len(self.system) + MESSAGE_OVERHEAD_CHARS,
            "pinned": sum(len(p) for p in self.pinned) + MESSAGE_OVERHEAD_CHARS,
            "history": summary
            + sum(len(k.path) + len(k.content) for k in self.kept)
            + sum(message_chars(m) + reasoning_chars(m, self.model, self.replay) for m in self.history),
            "tail": sum(len(t) for t in self.tail),
        }

    def total_chars(self) -> int:
        return sum(self.segment_chars().values())


def _bounded(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[{label} cut to fit the window; {len(text) - limit} characters omitted]"


def build_view(path: list[Entry], instructions: Instructions, meter: TokenMeter, working_window: int) -> ContextView:
    """Project the journal's active path into a view: the last compaction plus what followed it."""
    messages: list[tuple[str, Message]] = [(e.id, e.message) for e in path if isinstance(e, MessageEntry)]
    request = next((m for _, m in reversed(messages) if isinstance(m, UserMessage) and m.source == "user"), None)

    compaction = next((e for e in reversed(path) if isinstance(e, CompactionEntry)), None)
    summary, kept = None, []
    if compaction is not None:
        summary, kept = compaction.summary, compaction.kept
        ids = [entry_id for entry_id, _ in messages]
        if compaction.first_kept_id in ids:
            start = ids.index(compaction.first_kept_id)
        else:  # nothing kept: history restarts after the compaction
            start = sum(isinstance(e, MessageEntry) for e in path[: path.index(compaction)])
        messages = messages[start:]
    messages = [(i, m) for i, m in messages if not isinstance(m, SystemMessage)]
    history_ids = [i for i, _ in messages]
    state, covered = summary, 0
    latest = next((e for e in reversed(path) if isinstance(e, StateEntry | CompactionEntry)), None)
    if isinstance(latest, CompactionEntry) and latest.method == "state":
        # The compaction cut to a standing State; that State may still cover some of the kept turns.
        latest = next((e for e in reversed(path) if isinstance(e, StateEntry)), None)
    if isinstance(latest, StateEntry):
        state = latest.text
        covered = history_ids.index(latest.covers_through_id) + 1 if latest.covers_through_id in history_ids else 0

    pinned_limit = int(meter.chars_per_token * working_window * PINNED_SHARE)
    pinned = [
        _bounded(f"Instructions from {doc.source}:\n{doc.content}", pinned_limit, doc.source)
        for doc in instructions.pinned
    ]
    memory = next((e for e in reversed(path) if isinstance(e, MemoryRecallEntry)), None)
    if memory is not None and memory.text:
        pinned.append(memory.text)
    return ContextView(
        system=instructions.system,
        pinned=pinned,
        summary=summary,
        kept=kept,
        history=[m for _, m in messages],
        history_ids=history_ids,
        request=request.content if request is not None else None,
        state=state,
        covered=covered,
    )


def uncovered_tokens(view: ContextView, meter: TokenMeter) -> int:
    """Tokens of history the State does not describe yet."""
    return meter.tokens(sum(message_chars(m) for m in view.history[view.covered :]))


@dataclass
class CompactionPlan:
    older: list[Message]
    first_kept_id: str | None
    tokens_before: int


def estimate_tokens(view: ContextView, tools: list[ToolSpec], meter: TokenMeter) -> int:
    return meter.tokens(view.total_chars() + tools_chars(tools))


def plan_compaction(
    view: ContextView,
    tools: list[ToolSpec],
    meter: TokenMeter,
    working_window: int,
    thresholds: Thresholds,
    *,
    now: bool = False,
) -> CompactionPlan | None:
    """Choose which whole turns to compact, or None when the request still fits under the threshold (`now`: at
    any size, as a client asks) or when every turn fits in the recent share kept as it is."""
    tokens = estimate_tokens(view, tools, meter)
    if not now and tokens <= thresholds.compact_at * working_window:
        return None
    # A turn starts at any message that is not a tool result, so a cut never separates a call from its result.
    cuts = [i for i, m in enumerate(view.history) if not isinstance(m, ToolResultMessage)]
    if not cuts:
        return None
    keep_budget = thresholds.keep_recent * working_window
    suffix = [0] * (len(view.history) + 1)
    for i in range(len(view.history) - 1, -1, -1):
        chars = message_chars(view.history[i]) + reasoning_chars(view.history[i], view.model, view.replay)
        suffix[i] = suffix[i + 1] + meter.tokens(chars)
    cut = next((i for i in cuts if suffix[i] <= keep_budget), cuts[-1])
    if cut == 0:
        return None
    return CompactionPlan(older=view.history[:cut], first_kept_id=view.history_ids[cut], tokens_before=tokens)


def bound_output(text: str, limit: int) -> tuple[str, bool]:
    """Keep the head and, with priority, the tail of `text` within `limit` characters, at line breaks."""
    if len(text) <= limit:
        return text, False
    head_size = int(limit * BOUNDED_HEAD_SHARE)
    head = text[:head_size].rsplit("\n", 1)[0]
    tail = text[len(text) - (limit - head_size) :]
    tail = tail.split("\n", 1)[1] if "\n" in tail else tail
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n[... {omitted} characters omitted ...]\n{tail}", True


def _value(number: str) -> str:
    """1,234.50 and 1234.5 are one number."""
    plain = number.replace(",", "")
    if "." in plain:
        plain = plain.rstrip("0").rstrip(".")
    return plain.lstrip("0") or "0"


@dataclass
class SeenNumbers:
    """The numbers a request shows, and its lists of numbers, to tell a number in the State the run showed."""

    values: set[str] = field(default_factory=set)
    lists: list[frozenset[str]] = field(default_factory=list)

    def __contains__(self, number: object) -> bool:
        if not isinstance(number, str):
            return False
        if _value(number) in self.values:
            return True
        # 120,126 is a list packed together when its items are all in one list the run showed (a subset will do);
        # items scattered over the run do not count, or a made-up 125,135 would pass.
        items = {_value(item) for item in number.split(",")}
        return "," in number and any(items <= run for run in self.lists)


def numbers_in(messages: list[Message]) -> SeenNumbers:
    """Every number the messages show: text, reasoning, tool call arguments and tool results."""
    found = SeenNumbers()
    for message in messages:
        match message:
            case AssistantMessage():
                parts = [message.text, message.reasoning or ""]
                parts += [json.dumps(c.arguments, ensure_ascii=False) for c in message.tool_calls]
            case ToolResultMessage():
                parts = [message.result.content]
            case _:
                parts = [message.content]
        for part in parts:
            found.values.update(_value(n) for n in NUMBER.findall(part))
            found.lists += [frozenset(_value(n.strip()) for n in run.split(",")) for run in NUMBER_LIST.findall(part)]
    return found


def unseen_numbers(text: str, seen: SeenNumbers) -> list[str]:
    """The numbers in `text` that are not in `seen`, as `text` writes them; list items' own numbers aside."""
    return sorted({m.group() for m in _MARK_OR_NUMBER.finditer(text) if not m.group("mark") and m.group() not in seen})


def blank_unseen_numbers(text: str, seen: SeenNumbers) -> str:
    """`text` with every number not in `seen` replaced by `?`; list items' own numbers stay."""
    return _MARK_OR_NUMBER.sub(lambda m: m.group() if m.group("mark") or m.group() in seen else "?", text)


def fix_number(text: str, call: ToolCall, numbers: list[str]) -> tuple[str, str | None]:
    """The State after one number-fixing edit, and why it was not applied (None if it was).

    Only an edit on STATE.md whose old_string is one line holding one of `numbers` applies, so a fix cannot
    rewrite the State: everything but the lines with those numbers stays as the writer first wrote it."""
    args = call.arguments
    old, new = args.get("old_string"), args.get("new_string")
    if call.name != "edit" or str(args.get("path", "")).rsplit("/", 1)[-1] != STATE_FILE:
        return text, f"Not run: only edit on {STATE_FILE} fixes the numbers."
    if not isinstance(old, str) or not isinstance(new, str) or not any(n in old for n in numbers):
        return text, f"old_string must be the line that holds one of: {', '.join(numbers)}."
    if "\n" in old.strip("\n"):
        return text, "old_string must be one line: fix one number per edit."
    count = text.count(old)
    if count != 1:
        found = "was not found" if count == 0 else f"occurs {count} times"
        return text, f"old_string {found} in {STATE_FILE}: copy the whole line exactly."
    return text.replace(old, new, 1), None


@dataclass
class StateWrite:
    """What one State write produced: the State (None when there was none) and what it cost."""

    text: str | None
    usage: Usage
    unseen_numbers: list[str]  # numbers the State gave that the conversation never showed
    blanked: list[str]  # those still there after the fix, replaced by `?`
    method: Literal["edit", "rewrite"] = "rewrite"  # edits on the previous State, or the State written whole
    fallback: str | None = None  # why edits were asked for and the State was written whole instead


@dataclass
class StateEdit:
    """A State write as edits: the State they apply to, and the request for the whole State, should they fail."""

    previous: StateDoc
    rewrite: ProviderRequest


def _edited(previous: StateDoc, message: AssistantMessage) -> tuple[str | None, Literal["edit", "rewrite"], str | None]:
    """The State a reply to the edit request gives, how, and why there is none (then it is written whole)."""
    reply = message.text.strip()
    if not reply:
        return None, "edit", "the reply called a tool" if message.tool_calls else "the reply was empty"
    # Edits that parse are edits even beside a tool call (Qwen3.8 did so in a replay); the fork runs no tool.
    if is_no_change(reply):
        return render_state(previous), "edit", None
    if is_whole(reply):  # the writer wrote the State out after all: take it as written
        return reply, "rewrite", None
    try:
        text = render_state(apply_edits(previous, reply))
    except EditError as e:
        return None, "edit", "the reply called a tool" if message.tool_calls else str(e)
    return text, "edit", None


async def write_state(
    request: ProviderRequest,
    reply: Callable[[ProviderRequest], Awaitable[AssistantMessage | None]],
    edit: StateEdit | None = None,
) -> StateWrite:
    """One State write: one reply with the whole State, or with ``edit``, one reply with edits on the previous
    State, applied by code (when they cannot be, a second reply with the whole State; when they leave it past
    `MAX_EDITED_WORDS`, a second reply with edits that trim it). Numbers in it that appear
    nowhere in the request (a miscopy, or the writer's own sum) go back to the writer in the same fork, as a file
    to fix with line edits; those still there are blanked. The State is all the agent keeps of what it found, so
    it must not hold a number the run never showed."""
    message = await reply(request)
    usage = message.usage if message is not None else Usage()
    history: list[Message] | None = None  # the fork so far, when it went past the request and its reply
    method: Literal["edit", "rewrite"] = "rewrite"
    fallback: str | None = None
    text: str | None = None
    if edit is not None:
        if message is None:
            fallback = "the edit request failed"
        else:
            text, method, fallback = _edited(edit.previous, message)
        if text is not None and message is not None and words(text) > MAX_EDITED_WORDS:
            text, history, trim_usage = await _trimmed(text, request, message, reply)
            usage += trim_usage
        if text is None:
            request, method = edit.rewrite, "rewrite"
            message = await reply(request)
            usage += message.usage if message is not None else Usage()
    if text is None:
        if message is None or message.tool_calls or not message.text.strip():
            return StateWrite(text=None, usage=usage, unseen_numbers=[], blanked=[], fallback=fallback)
        text = message.text.strip()
        if not parse_state(text).is_state():  # no section of a State: not one; the previous State stands
            return StateWrite(text=None, usage=usage, unseen_numbers=[], blanked=[], fallback=fallback)
    seen = numbers_in(request.messages)
    flagged = unseen_numbers(text, seen)
    left = flagged
    fix_prompt = (
        STATE_NUMBERS_EDITED_PROMPT.format(numbers=", ".join(flagged), state=text)
        if method == "edit"
        else STATE_NUMBERS_PROMPT.format(numbers=", ".join(flagged))
    )
    assert message is not None
    messages: list[Message] = [
        *(history or [*request.messages, message]),
        UserMessage(content=fix_prompt, source="harness"),
    ]
    for _ in range(MAX_NUMBER_FIX_REPLIES if flagged else 0):
        fix = await reply(request.model_copy(update={"messages": messages}))
        if fix is None:
            break
        usage += fix.usage
        if not fix.tool_calls:
            break
        results: list[Message] = []
        for call in fix.tool_calls:
            if call.name == "read" and str(call.arguments.get("path", "")).rsplit("/", 1)[-1] == STATE_FILE:
                status, content = "ok", text  # the writer's habit before an edit; the file is the State
            else:
                text, error = fix_number(text, call, left)
                status = "ok" if error is None else "error"
                content = error or f"{STATE_FILE} updated."
            results.append(
                ToolResultMessage(
                    result=ToolResult(call_id=call.id, tool_name=call.name, status=status, content=content)
                )
            )
        left = unseen_numbers(text, seen)
        if not left:
            break
        note = f"Still not in this conversation: {', '.join(left)}. Fix each with one edit, or remove it."
        messages = [*messages, fix, *results, UserMessage(content=note, source="harness")]
    return StateWrite(
        text=blank_unseen_numbers(text, seen).strip() or None,
        usage=usage,
        unseen_numbers=flagged,
        blanked=unseen_numbers(text, seen),
        method=method,
        fallback=fallback,
    )


async def _trimmed(
    text: str,
    request: ProviderRequest,
    message: AssistantMessage,
    reply: Callable[[ProviderRequest], Awaitable[AssistantMessage | None]],
) -> tuple[str, list[Message] | None, Usage]:
    """An edited State past `MAX_EDITED_WORDS`, trimmed by one more reply in the same fork: edits that drop or
    shorten items, to some room under the limit. Kept as edited, over it, when that reply fails, cannot be applied
    or does not shorten it (a writer may keep what the task still needs; a capped window may leave no room)."""
    doc = parse_state(text)
    goal = MAX_EDITED_WORDS * 9 // 10  # some room, so the next edit does not go straight back over
    ask = UserMessage(
        content=STATE_TRIM_PROMPT.format(
            words=f"{words(text):,}",
            limit=f"{MAX_EDITED_WORDS:,}",
            goal=f"{goal:,}",
            cut=f"{words(text) - goal:,}",
            state=render_state(doc, numbered=True),
        ),
        source="harness",
    )
    history: list[Message] = [*request.messages, message, ask]
    trim = await reply(request.model_copy(update={"messages": history}))
    if trim is None:
        return text, None, Usage()
    trimmed, _, _ = _edited(doc, trim)
    if trimmed is None or words(trimmed) >= words(text):
        return text, None, trim.usage
    return trimmed, [*history, trim], trim.usage


def state_edit_request(previous: StateDoc) -> str:
    """The message that turns the agent's next request into edits on the State it has."""
    size = STATE_SIZE.format(words=f"{words(render_state(previous)):,}", limit=f"{MAX_EDITED_WORDS:,}")
    return STATE_EDIT_PROMPT.format(size=size, state=render_state(previous, numbered=True))


def state_request(view: ContextView) -> str:
    """The message that turns the agent's next request into a State write."""
    if view.state and view.state == view.summary:
        previous = "The previous state is the one given at the start of this conversation."
    elif view.state:
        previous = f"Previous state:\n{without_code_sections(view.state)}"
    else:
        previous = "There is no previous state."
    return f"{STATE_PROMPT}\n\n{previous}"


def requests_section(requests: list[str], limit: int) -> str:
    """The last `MAX_REQUESTS` requests as written, oldest first, within `limit` characters counted from the newest,
    which is always kept (cut to fit). Each is a block quote, a blank line apart, so where one ends is plain however
    many lines or lists it has; the order is said once, not as labels, since a request newer than the State can
    follow it in the transcript."""
    asked = [text.strip() for text in requests if text.strip()][-MAX_REQUESTS:]
    kept: list[str] = []
    room = limit
    for text in reversed(asked):
        if len(text) > room:
            if not kept:
                kept.append(bound_output(text, max(room, 0))[0])
            break
        kept.append(text)
        room -= len(text)
    if not kept:
        return ""
    quotes = ("\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines()) for text in reversed(kept))
    return "\n\n".join([f"## {REQUESTS_SECTION[0]}\n{REQUESTS_SECTION[1]}", *quotes])


def processes_section(running: Sequence[ProcessInfo], now: int) -> str:
    """The jobs and what commands left running (not `bash` calls, which end before the State is written), the oldest
    `MAX_PROCESSES` first, each with what the agent needs to reach it again; empty when none runs."""
    lines = []
    for info in [p for p in running if p.kind != "call"][:MAX_PROCESSES]:
        took = duration_text((now - info.started_at) // 1000)
        if info.kind == "job":
            lines.append(f"- job {info.id} (`process` tool): `{info.command}`, running {took}")
        else:
            group = info.id.removeprefix("group-")
            lines.append(f"- left running by a command, process group {group}: `{info.command}`, running {took}")
    return "\n".join([f"## {PROCESSES_SECTION[0]}\n{PROCESSES_SECTION[1]}", *lines]) if lines else ""


def without_code_sections(state: str) -> str:
    """`state` without the sections code sets: its `## User requests` and `## Running processes`."""
    titles = {REQUESTS_SECTION[0].casefold(), PROCESSES_SECTION[0].casefold()}
    lines = state.splitlines()
    kept: list[str] = []
    inside = False
    for line in lines:
        title = heading(line)
        if title is not None:
            inside = title.casefold() in titles
        if not inside:
            kept.append(line)
    return state if len(kept) == len(lines) else "\n".join(kept).strip()


def with_requests(state: str, requests: list[str], limit: int, running: Sequence[ProcessInfo] = ()) -> str:
    """`state` with the commands still `running` and then the user's latest requests as its last sections, replacing
    any it had (the writer's copies too)."""
    sections = [processes_section(running, now_ms()), requests_section(requests, limit)]
    return "\n\n".join(part for part in [without_code_sections(state), *sections] if part)


def dropped_turns(previous: str | None, older: list[Message]) -> str:
    """Deterministic fallback when no State could be written: list what the dropped turns did."""
    calls = [
        f"- {c.name}({json.dumps(c.arguments)[:160]})"
        for m in older
        if isinstance(m, AssistantMessage)
        for c in m.tool_calls
    ]
    listing = "\n".join(calls[-40:]) or "- (no tool calls)"
    head = f"{previous}\n\n" if previous else ""
    return f"{head}Earlier turns were removed to fit the window. Tool calls they made:\n{listing}"


def squeeze(view: ContextView, tools: list[ToolSpec], meter: TokenMeter, limit_tokens: int) -> int:
    """Last resort when whole-turn compaction is not enough: shorten tool results in the view, oldest
    first. The journal keeps the full text. Returns how many results were shortened."""
    shortened = 0
    for index, message in enumerate(view.history):
        if estimate_tokens(view, tools, meter) <= limit_tokens:
            break
        if isinstance(message, ToolResultMessage) and len(message.result.content) > SQUEEZED_RESULT_CHARS:
            result = message.result.model_copy(
                update={"content": _bounded(message.result.content, SQUEEZED_RESULT_CHARS, "old tool output")}
            )
            view.history[index] = ToolResultMessage(result=result)
            shortened += 1
    return shortened
