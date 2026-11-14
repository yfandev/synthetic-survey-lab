from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

import httpx
from pydantic import SecretStr

from intern.memory.steward.extract import INJECTION, Extract, file_pages
from intern.messages import Usage
from intern.models.registry import ConfigError, Endpoint, resolve_endpoint
from intern.providers.openai_compat import reply_usage

GENERAL = "general"
INSTRUCTIONS = "instructions"
ME = "user/me"
NEW = "new"
GROUPS = ("people", "organisations", "projects", "places", "things")
KINDS = ("where", "how", "fails", "state", "event", "rule", "preference", "told")
# Which kinds each source may give: a user's words, the assistant's words, a tool's result, the agent's own change.
FROM_SOURCE = {
    "user": {"where", "how", "fails", "state", "event", "rule", "preference"},
    "assistant": {"told"},
    "tool": {"where", "how", "fails", "state", "rule"},
    "action": {"where", "how", "fails", "state", "event", "rule"},
}
MAX_ITEMS = 24
DRAFT_ATTEMPTS = 2
ITEM_CHARS = 300
ALIAS_CHARS = 60
ORIGIN_CHARS = 60
MAX_ALIASES = 6
MAX_ALIAS_WORDS = 4
LESSON_KINDS = ("how", "fails", "rule")
MAX_LESSONS = 4
SUCCEEDED = "succeeded"
MIN_LESSON_SESSIONS = 2
TITLE_CHARS = 80
MAX_STORY = 4
TRACE_CHARS = 60_000
TIMEOUT_S = 120.0
RETRIES = 4
RETRY_WAIT_S = 5.0
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# Word overlap at which a new item repeats one already in memory.
SIMILAR = 0.6

SYSTEM = """You keep the long-term memory of an AI agent that works for one user, session after session, in the
same world: the user's services, projects, documents and tools, and the people and things they deal with.
You get one finished session in order. Each line has an id: u<n> is a message from the user, a<n> a reply from
the assistant, s<n> a tool call of the agent with the start of its result.

Write memory items: what will still be true and useful in a later session. Values such as names, ids, amounts,
dates and addresses are welcome when they stay true. Leave out what only served this one task, such as a draft,
a lookup or an intermediate result.

Kinds of item:
- "where": where something is or how to reach it: a service and the call it accepts, a folder, a file, an address.
- "how": how something is done or used, as it worked here.
- "fails": what does not work here: a call that errors, a missing program, a rejected option.
- "state": a fact about a subject that holds until it changes; "attribute" is the one or two words the fact
  answers, such as "role", "email", "plan", "setting" or "status", and empty when none fits. What someone asked for
  or wrote in one message is not their role.
- "event": something that happened or will happen; "when" is its date as YYYY-MM-DD, YYYY-MM or YYYY, worked out
  from the line's date, and "phrase" keeps the original wording such as "last Tuesday" or "next spring".
- "rule": a standing instruction on how to act; "trigger" is the action and the scope it applies to,
  such as "sending messages to a client", "booking anything on a Friday" or "every reply".
- "preference": what the user likes or dislikes, with no instruction to act.
- "told": what the assistant told or recommended to the user that they may ask about again.

Where items come from:
- A user line can give state, events, preferences and rules, often several in one message. A request for this
  task alone ("use the short version here") is not a rule. Write a rule only when the user says it holds beyond
  this task, for example with "always", "never", "from now on", "whenever" or a standing setup.
- An s line can give where, how, fails and state from what the tool returned, and an event when the agent changed
  something. A change that worked also gives a "how" on the page of the tool or service that made it: the call and
  the fields it needed, and the document it followed when one said to make it, without this task's values.
  What a tool returned is never a rule. A rule from s lines is a working rule the agent learned: it goes
  on "general" or a place page, and only when the outcome is succeeded.
- What a document, record or page the agent read says (terms, prices, dates, limits, policies, the step to take
  in a given case) holds for later tasks whatever this session's outcome: it is state or how on the page of what
  it is about, such as a product, offer or service (a subject page or "{new}"), with the document's title or id as
  "from". The document also gives a "where" on the page of the place that holds it: its title or id and what it
  covers.
- An assistant line gives only "told".

Pages (a page name says where memory files an item, never where a thing is: "text" names a file by its path as
the session used it):
- a file page listed below, for what holds about that one file: its layout, sheets, columns, units, which copy
  to use and its quirks. What a file holds goes on its file page, never on a subject page;
- a place page listed below (a folder, project, service, tool or the environment), for what holds about that place;
- "{general}": working rules the agent learned that hold everywhere;
- "{instructions}": the user's standing rules that hold everywhere, not tied to one subject;
- "{me}": facts about the user themself, from what the user said. A name, email or role that a tool result or
  document shows (a contact, a signature, a directory entry) is about that person and goes on their page, even
  when it matches the user;
- a subject page listed below, or "{new}" with "subject" (the name as written) and "group" ({groups}) for a person,
  organisation, project, place or thing the user deals with. A rule about one subject goes on its page.
  The people who write messages the agent reads (emails, chats, notes) are subjects too. So is the project or
  organisation the work is for: what holds about it as a whole (its goal, client, team, deadlines, its standing
  rules and methods) goes on its page; what one file holds stays on the file's page, and a person's role in it on
  the person's page.

Writing:
- "text" is one self-contained sentence under {item_chars} characters that names its subject.
  Every value in it (name, number, date, id, address, quoted text) must appear in the cited line.
  Keep negations, limits and scope words as written ("never", "only on weekdays", "except"), and keep numbers as
  given ("3 of 68").
- "aliases": other names the line uses for the same subject, such as nicknames, handles, emails or ids, as written.
  A part, file, document or topic of the subject is not another name for it.
- "from": for an s line, what the fact was read from, as the line names it: a file, a message and who sent it,
  a record or a page, such as "Survey Questions.docx" or "email from Anna"; empty for u and a lines.
- "topic": the part of the page's place or subject the item is about, in 1 to 3 words as the session names it,
  such as a sheet, tab, folder, section, account or feature ("Calc_Core", "Pricing Module"); empty when it is
  about the whole place or subject. Items about the same part share one topic.
- "step" cites the one line the item comes from.
- Skip greetings, what a capable agent knows anyway, and what memory already holds (listed below) unless it changed.
- Up to {max_items} items, the most lasting first.

Also describe the session for a person reading the memory later:
- "title": the kind of work, under {title_chars} characters, like "Fix a validation error in a config parser" or
  "Plan a weekend trip"; it names no person, company, project, file or record;
- "story": 1 to {max_story} sentences, each citing a line: what was asked, what was done and how it ended;
- "work": the kind of work, 1 to 3 lowercase words joined by hyphens, like "bug-fixes" or "travel-planning",
  general enough that other sessions doing the same work share it, and specific enough to tell it from the user's
  other work: "invoice-reconciliation" or "pricing-models", not "data-analysis" or "coding". Reuse a kind already
  in memory when it names the same work, even if the details differ.

Answer with JSON only:
{{"title": "<title>", "work": "<kind of work>", "story": [{{"step": "<id>", "text": "<sentence>"}}],
 "items": [{{"kind": "<kind>", "page": "<page>", "subject": "", "group": "", "aliases": [], "attribute": "",
 "when": "", "phrase": "", "trigger": "", "from": "", "topic": "", "step": "<id>", "text": "<sentence>"}}]}}"""

LESSONS = """You keep the working lessons of an AI agent that works for one user, session after session, in the
same world. You get the stories of past sessions, each with how it ended, and the skills memory already holds.

Write lessons: ways of working that more than one session shows and that will help later tasks here.
- "how": a step of a way of working that helped, such as where to look first or which check settled the answer.
- "fails": a pitfall: an approach that went wrong or wasted steps.
- "rule": a check to always do for this kind of work.
A "how" or "rule" cites at least one succeeded session; a failed, partial or unverified session can only show what
to avoid. A lesson cites story lines of at least two different sessions in "steps". It may name the places, files
and people it concerns, never one task's results or numbers. Skip what a skill already holds and what a capable
agent does anyway, such as reading the documents or checking the facts before answering. A lesson names a concrete
step: the place to look, the call or argument to use, or the check that settled the answer. One that only asks for
more searching, checking or care adds steps to every later task: skip it. Up to {max_lessons}
lessons, the most useful first; none when the sessions share no way of working.

Each lesson belongs to a skill: the kind of work it helps with. Give "skill" as the name of a skill listed below,
or name a new one in 2 to 5 words, such as "Survey data analysis", with "when": one sentence on the tasks it fits.
Prefer a listed skill when the lesson fits it.

Answer with JSON only:
{{"lessons": [{{"kind": "<kind>", "skill": "<name>", "when": "", "steps": ["<id>", "<id>"],
 "text": "<one sentence under {item_chars} characters>"}}]}}"""

CONDENSE = """You keep the long-term memory of an AI agent. One memory page is full: rewrite it to free at least
{need} characters, so new lines fit without moving old ones out. The page's lines are listed with ids, by section.

- "merged": lines of one section that state related facts about the same thing, rewritten as one self-contained
  sentence under {merged_chars} characters. Keep every value, name, negation, limit and scope word of the lines it
  replaces, and add nothing they do not say.
- "dropped": a line that another line on the page repeats, or that a newer line shows no longer holds.
Leave every other line as it is. Never merge lines of different sections, and never merge a line with itself.

Answer with JSON only:
{{"merged": [{{"lines": ["<id>", "<id>"], "text": "<sentence>"}}], "dropped": ["<id>"]}}"""
MERGED_CHARS = 2 * ITEM_CHARS
TOPIC_CHARS = 40
CONDENSE_EFFORT = "low"
SKILL_NAME_CHARS = 60

_VALUE = re.compile(
    r"(?i)"
    r"[\w.+-]+@[\w-]+\.[\w.]+"  # email
    r"|[$€£]\s?\d"  # amount
    r"|\b\d{4}-\d{2}-\d{2}\b"  # date
    r"|\"[^\"]{2,}\"|'[^']{2,}'"  # quoted text
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d"  # date in words
    # a token with a digit, other than path versions (v2, 1.0) and HTTP status codes
    r"|(?<![\w.])(?!v\d+(?:\.\d+)*\b)(?!\d+\.\d+\b)(?![1-5]\d\d\b)[A-Za-z_]*\d[\w-]*"
)
# What an item must take from its source: emails, amounts, tokens with a digit, quoted and backticked text, names.
_GROUND = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|[$€£]\s?[\d.,]+\d|(?<![\w])[\w./:+-]*\d[\w./:+-]*|\"([^\"]{2,})\"|`([^`]+)`"
    r"|(?<!\w)'([^'\n]{2,80})'(?!\w)"
)
_NAME = re.compile(r"(?<![\w'’-])[A-Z][\w&'’-]*")
_SENTENCE_START = re.compile(r"(?:^|[.!?:;]\s+|\(\s*)$")
_WORDS = re.compile(r"[a-z0-9{}/._-]+")
_NUMBER = re.compile(r"(?<![\w.-])\d+(?:\.\d+)*(?![\w-])")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_WORK = re.compile(r"[a-z]+(?:-[a-z]+){0,2}")
_COMPOUND = re.compile(r"(?<![\w{])(?:[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+|[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*)(?![\w}])")
_KIND_SUFFIX = re.compile(r"\s*\([a-z]+\)\s*$")
_LINE_ID = re.compile(r"([usa])(\d+)")
_WHEN = re.compile(r"\d{4}(-\d{2}(-\d{2})?)?")
# A placeholder stands for a value ("<b64>", "<message id>"); it is not one.
_PLACEHOLDER = re.compile(r"<[\w .-]{1,30}>")
_COMMON_NAMES = frozenset({"I", "I'm", "I've", "I'd", "I'll", "OK"})
_FILE_NAME = re.compile(r"\.[A-Za-z][A-Za-z0-9]{1,4}$")


def name_like(alias: str, name: str, files: Iterable[str] = ()) -> bool:
    # Another name for a subject (a nickname, handle, email or id), not a part, file or topic of it.
    words = alias.split()
    lowered, base = alias.lower(), name.lower().strip()
    if not words or len(words) > MAX_ALIAS_WORDS or lowered == base:
        return False
    if base and lowered.startswith(f"{base} "):
        return False
    if "@" not in alias and (" " in alias or "/" in alias) and _FILE_NAME.search(alias):
        return False
    if any(lowered == f.lower() for f in files):
        return False
    return any(c.isupper() or c.isdigit() or c == "@" for c in alias)


@dataclass
class Draft:
    page: str
    pointer: str
    text: str


@dataclass
class Item:
    kind: str
    page: str
    text: str
    pointer: str
    source: str
    said: str
    order: int
    subject: str = ""
    group: str = ""
    aliases: list[str] = field(default_factory=list)
    attribute: str = ""
    when: str = ""
    phrase: str = ""
    trigger: str = ""
    origin: str = ""
    topic: str = ""


@dataclass
class Story:
    title: str
    lines: list[Draft]
    work: str = ""


@dataclass
class Past:
    # One session as its work page tells it, for lessons drawn across sessions.
    session: str
    title: str
    outcome: str
    lines: list[Draft]


@dataclass
class Learned:
    kind: str
    text: str
    pointers: list[str]
    sessions: list[str]
    skill: str = ""
    when: str = ""
    succeeded: int = 0
    # The line memory already holds that says the same, which this lesson confirms, and the page it is on.
    old: str = ""
    old_page: str = ""


@dataclass
class Skill:
    # A skill memory holds, as the lessons writer sees it.
    name: str
    when: str
    lines: list[str]


@dataclass
class Merge:
    ids: list[str]
    text: str


@dataclass
class Rejected:
    pointer: str
    reason: str


class Writer:
    def __init__(self, endpoint: Endpoint, transport: httpx.AsyncBaseTransport | None = None) -> None:
        if endpoint.api == "anthropic":
            raise ConfigError(f"the steward writer needs an OpenAI-compatible provider, not {endpoint.provider}")
        self.endpoint = endpoint
        self.name = endpoint.ref
        self._transport = transport
        self.usage = Usage()

    async def complete(self, system: str, user: str, session: str = "", effort: str = "") -> str:
        key = self.endpoint.api_key
        headers = {"Authorization": f"Bearer {key.get_secret_value()}"} if key else {}
        payload: dict[str, object] = {
            "model": self.endpoint.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.endpoint.provider == "vllm":
            # Qwen thinks by default; notes need no reasoning trace and come back much faster without it.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif effort:
            # An open-ended rewrite can think until the output limit and return no answer at all.
            payload["reasoning_effort"] = effort
        url = f"{self.endpoint.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=TIMEOUT_S, transport=self._transport) as client:
            response = await client.post(url, headers=headers, json=payload)
            for attempt in range(RETRIES):
                if response.status_code not in RETRY_STATUS:
                    break
                await asyncio.sleep(RETRY_WAIT_S * 2**attempt)
                response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
            self.usage += reply_usage(body.get("usage"), self.endpoint.ref)
            return body["choices"][0]["message"].get("content") or ""


class ReplayWriter:
    def __init__(self, folder: Path, record: Writer | None = None) -> None:
        self.folder = folder
        self.record = record
        self.name = f"replay:{folder}" if record is None else record.name

    @property
    def usage(self) -> Usage:
        return self.record.usage if self.record is not None else Usage()

    async def complete(self, system: str, user: str, session: str = "", effort: str = "") -> str:
        saved = self.folder / f"{session}.json"
        if self.record is None:
            if not saved.is_file():
                raise ValueError(f"no saved draft for session {session} in {self.folder}")
            return json.loads(saved.read_text())["completion"]
        text = await self.record.complete(system, user, session, effort)
        saved.write_text(json.dumps({"completion": text}))
        return text


def writer_from_ref(ref: str, env: Mapping[str, str]) -> Writer | ReplayWriter:
    if ref.startswith("replay:"):
        return ReplayWriter(Path(ref.removeprefix("replay:")).expanduser())
    endpoint = resolve_endpoint(ref, env)
    # Local servers need no key by default; a key in $<PROVIDER>_API_KEY is sent when one is set.
    if endpoint.api_key is None and (key := env.get(f"{endpoint.provider.upper()}_API_KEY")):
        endpoint = endpoint.model_copy(update={"api_key": SecretStr(key)})
    return Writer(endpoint)


@dataclass
class Line:
    # One citable line of the session: its id in the trace, the text an item is checked against, and its source.
    ident: str
    pointer: str
    source: str
    text: str
    said: str
    order: int
    change: bool | None = None


def lines(ex: Extract) -> dict[str, Line]:
    found: dict[str, Line] = {}
    users = [t for t in ex.turns if t.role == "user"]
    replies = [t for t in ex.turns if t.role == "assistant"]
    for prefix, turns, source in (("u", users, "user"), ("a", replies, "assistant")):
        for i, turn in enumerate(turns):
            ident = f"{prefix}{i}"
            found[ident] = Line(ident, turn.pointer, source, turn.text, turn.at[:10] or ex.date, turn.order)
    for i, step in enumerate(ex.steps):
        source = "action" if step.change else "tool"
        text = f"{step.call}\n{step.output or step.result}"
        found[f"s{i}"] = Line(f"s{i}", step.pointer, source, text, step.at[:10] or ex.date, step.order, step.change)
    return found


def trace(ex: Extract, outcome: str) -> str:
    rows: list[tuple[int, str]] = []
    for line in lines(ex).values():
        if line.ident.startswith("s"):
            step = ex.steps[int(line.ident[1:])]
            status = "ok" if step.ok else "error"
            rows.append((line.order, f"{line.ident} [{status}] $ {step.call}\n    -> {step.result}"))
        else:
            who = "user" if line.source == "user" else "assistant"
            rows.append((line.order, f"{line.ident} [{who}, {line.said}] {line.text}"))
    rows.sort(key=lambda row: row[0])
    head = [f"Session date: {ex.date}", f"Outcome (judged): {outcome}"]
    if not ex.turns:
        head.append(f"Task:\n{ex.task}")
    text = "\n".join([*head, "Session:", *(row for _, row in rows)])
    if len(text) <= TRACE_CHARS:
        return text
    half = TRACE_CHARS // 2
    return f"{text[:half]}\n[...]\n{text[-half:]}"


def _items_json(raw: str) -> dict | None:
    try:
        answer = json.loads(_FENCE.sub("", raw.strip()))
    except json.JSONDecodeError:
        return None
    return answer if isinstance(answer, dict) and "items" in answer else None


def numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def similar(a: str, b: str) -> bool:
    # Lines that differ in a number (a port, a limit, a version) say different things, however alike their words.
    if numbers(a) != numbers(b):
        return False
    x, y = set(_WORDS.findall(a.lower())), set(_WORDS.findall(b.lower()))
    return bool(x and y) and len(x & y) / len(x | y) >= SIMILAR


def has_value(text: str, values: Iterable[str] = (), names: Iterable[str] = ()) -> bool:
    plain = re.sub(r"\{[^{}]+\}", "", text)
    if _VALUE.search(plain):
        return True
    lowered = plain.lower()
    values = list(values)
    if any(v.lower() in lowered and re.search(rf"(?<![\w-]){re.escape(v.lower())}(?![\w-])", lowered) for v in values):
        return True
    # A compound name in a note that is part of a longer name in the session's data is still that data.
    seen = "\n".join(v.lower() for v in values)
    allowed = set(names)
    return any(n.lower() in seen and n.lower() not in allowed for n in _COMPOUND.findall(plain))


# A record id or a name in code: SKU-6999, INV_204, R2438, year_offset.
_ID = re.compile(
    r"(?<![\w.@/])(?:[A-Z]{1,8}-?\d{3,}|[A-Z][A-Za-z]{0,7}-\d{3,}|[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)(?!\w)"
)


def _values(text: str) -> list[str]:
    found: list[str] = []
    for match in _GROUND.finditer(text):
        found.append((match[1] or match[2] or match[3] or match[0]).strip().rstrip(".,;:!?)"))
    found += _COMPOUND.findall(text)
    for match in _NAME.finditer(text):
        if match[0] not in _COMMON_NAMES and not _SENTENCE_START.search(text[: match.start()]):
            found.append(match[0].rstrip(".,;:!?'’"))
    return [v for v in dict.fromkeys(found) if v and not _PLACEHOLDER.fullmatch(v)]


def _flat(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def _file_key(name: str) -> str:
    # An origin loses its brackets and commas, so file names compare by letters and digits only.
    return re.sub(r"[\W_]+", "", name.lower())


def ungrounded(text: str, source: str) -> list[str]:
    # Every value an item states must be in the session; a value worked out from it belongs in a field such
    # as `when`, never in the text. A lesson joins steps (a failure and its fix), so one cited line is too narrow.
    seen = _flat(source)
    return [v for v in _values(text) if _flat(v) not in seen]


def grounded(text: str, source: str) -> bool:
    return not ungrounded(text, source)


def _text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else ""


def _page(raw: object, kind: str, line: Line, pages: set[str], subjects: set[str]) -> str | None:
    page = _KIND_SUFFIX.sub("", raw) if isinstance(raw, str) else ""
    if page == GENERAL:
        return page if line.source in ("tool", "action") and kind in ("rule", "how", "fails") else None
    if page == INSTRUCTIONS:
        return page if line.source == "user" and kind == "rule" else None
    # A contact card or signature the agent read names whoever it is about; it does not say who the user is.
    if page == ME and line.source in ("tool", "action"):
        return None
    if page in pages:
        return page
    if page in (ME, NEW) or page in subjects:
        return page
    # A writer that names a new subject as its page means a new subject page.
    return NEW if page.strip() and "/" not in page and len(page) <= ALIAS_CHARS else None


def _item(
    raw: object, by_id: dict[str, Line], pages: set[str], subjects: set[str], session: str, files: set[str]
) -> Item | Rejected | None:
    if not isinstance(raw, dict):
        return None
    step = str(raw.get("step", "")).strip()
    line = by_id.get(step) if _LINE_ID.fullmatch(step) else None
    if line is None:
        return None
    kind = str(raw.get("kind", "")).strip().lower()
    text = _text(raw.get("text"), ITEM_CHARS)
    if kind not in KINDS or not text:
        return Rejected(line.pointer, f"an item with kind {kind!r} or text that does not fit")
    if kind not in FROM_SOURCE[line.source]:
        return Rejected(line.pointer, f"a {kind} item cannot come from a {line.source} line: {text}")
    page = _page(raw.get("page"), kind, line, pages, subjects)
    if page is None:
        return Rejected(line.pointer, f"a {kind} item on page {raw.get('page')!r}, which it cannot go on: {text}")
    if kind == "rule" and line.source in ("tool", "action") and (page.startswith("user/") or page == NEW):
        return Rejected(line.pointer, f"a rule from a tool result on a user page: {text}")
    subject = _text(raw.get("subject"), ALIAS_CHARS)
    if page == NEW and not subject and raw.get("page") != NEW:
        subject = _text(raw.get("page"), ALIAS_CHARS)
    group = str(raw.get("group", "")).strip().lower()
    group = group if group in GROUPS else next((g for g in GROUPS if g.startswith(group[:4] or "-")), "things")
    if page == NEW and not subject:
        return Rejected(line.pointer, f"a new subject without a name: {text}")
    cited = _flat(line.text)
    aliases = [
        a
        for alias in (raw.get("aliases") or [])[:MAX_ALIASES]
        if (a := _text(alias, ALIAS_CHARS)) and name_like(a, subject, files) and _flat(a) in cited
    ]
    # The origin sits inside the line's "(from <origin>, date)" label, so it holds no commas or brackets.
    origin = _text(raw.get("from"), ORIGIN_CHARS) if line.source in ("tool", "action") else ""
    origin = " ".join(re.sub(r"[,()\[\]]", " ", origin).split())
    attribute = " ".join(re.sub(r"[_\W]+", " ", _text(raw.get("attribute"), ALIAS_CHARS)).lower().split())
    if any(c.isdigit() for c in attribute):
        attribute = ""
    trigger = _text(raw.get("trigger"), ITEM_CHARS)
    when = str(raw.get("when", "")).strip()
    # Grounding is off until it shows value: its literal match dropped mostly correct facts. An id is never worked
    # out, though: one the session never showed (the end of a range, the next record) was made up.
    seen = _flat(session)
    if made_up := [v for v in _ID.findall(text) if v.lower() not in seen]:
        return Rejected(line.pointer, f"an item with ids not in its session ({', '.join(made_up[:3])}): {text}")
    if any(INJECTION.search(t) for t in (text, trigger, subject, origin, *aliases)):
        return Rejected(line.pointer, "an item matched a prompt-injection pattern; it is not kept")
    return Item(
        kind=kind,
        page=page,
        text=text,
        pointer=line.pointer,
        source=line.source,
        said=line.said,
        order=line.order,
        subject=subject,
        group=group,
        aliases=aliases,
        attribute=attribute,
        when=when if _WHEN.fullmatch(when) else "",
        phrase=_text(raw.get("phrase"), ALIAS_CHARS),
        trigger=trigger if kind == "rule" else "",
        origin=origin,
        topic=" ".join(re.sub(r"[,:()\[\]{}#]", " ", _text(raw.get("topic"), TOPIC_CHARS)).split()),
    )


def _cited(by_id: dict[str, Line], item: object, session: str) -> Draft | None:
    if not isinstance(item, dict):
        return None
    step, text = str(item.get("step", "")).strip(), " ".join(str(item.get("text", "")).split())
    line = by_id.get(step)
    if line is None or not text or len(text) > ITEM_CHARS:  # or not grounded(text, session): grounding is off
        return None
    return Draft(page="", pointer=line.pointer, text=text)


async def draft(
    writer: Writer | ReplayWriter,
    ex: Extract,
    outcome: str,
    pages: list[str],
    subjects: list[str],
    known: list[str],
    kinds: list[str] | None = None,
) -> tuple[Story | None, list[Item], list[Rejected]]:
    if not ex.steps and not ex.turns:
        return None, [], []
    system = SYSTEM.format(
        max_items=MAX_ITEMS,
        item_chars=ITEM_CHARS,
        title_chars=TITLE_CHARS,
        max_story=MAX_STORY,
        general=GENERAL,
        instructions=INSTRUCTIONS,
        me=ME,
        new=NEW,
        groups=", ".join(GROUPS),
    )
    kinds_of = {o.place: o.place_kind for o in ex.observations}
    files_of = file_pages(ex.observations)
    places = ", ".join(f"{p} ({kinds_of[p]})" if p in kinds_of else p for p in pages if p not in files_of) or "(none)"
    files = "\n".join(f"- {p}: {files_of[p]}" for p in pages if p in files_of) or "(none)"
    already = "\n".join(f"- {line}" for line in known) or "(nothing yet)"
    work = ", ".join(kinds or []) or "(none yet)"
    user = (
        f"File pages:\n{files}\n\nPlace pages: {places}\n\nSubject pages: {', '.join(subjects) or '(none yet)'}\n\n"
        f"Kinds of work already in memory: {work}\n\nAlready in memory:\n{already}\n\n{trace(ex, outcome)}"
    )
    # The writer now and then answers with JSON that does not parse; asked again, it answers well.
    raw, answer = "", None
    for _ in range(DRAFT_ATTEMPTS):
        raw = await writer.complete(system, user, ex.session_id)
        if (answer := _items_json(raw)) is not None:
            break
    if answer is None:
        raise ValueError(f"the writer did not answer with items JSON: {raw[:200]!r}")
    found = answer["items"]
    by_id = lines(ex)
    session = "\n".join(line.text for line in by_id.values())
    subject_paths = {s.split(" (", 1)[0] for s in subjects}
    names = {PurePosixPath(f).name for f in files_of.values()} | set(files_of.values())
    # What a file holds goes on its file page even when the writer puts it on a subject page; what a message in a
    # file says about a person stays on the person's page.
    file_homes = {_file_key(PurePosixPath(f).name): p for p, f in files_of.items()}
    items: list[Item] = []
    rejected: list[Rejected] = []
    for raw_item in found[:MAX_ITEMS] if isinstance(found, list) else []:
        if isinstance(raw_item, dict) and isinstance(raw_item.get("text"), str):
            # A file page's name says where memory files a line, not where the file is: an agent would read it.
            for page in sorted(files_of, key=len, reverse=True):
                raw_item["text"] = re.sub(rf"(?<![\w/]){re.escape(page)}(?![\w/-])", files_of[page], raw_item["text"])
        result = _item(raw_item, by_id, set(pages), subject_paths, session, names)
        if isinstance(result, Item):
            on_subject = (
                result.kind == "state"
                and (result.group != "people" if result.page == NEW else result.page in subject_paths)
                and not result.page.startswith("user/people/")
            )
            # A general lesson read from one file is about that file, and general lessons load in every task.
            if (on_subject or result.page == GENERAL) and (home := file_homes.get(_file_key(result.origin))):
                result = replace(result, page=home, subject="", group="", aliases=[])
        if isinstance(result, Item):
            items.append(result)
        elif isinstance(result, Rejected):
            rejected.append(result)
    title = " ".join(str(answer.get("title", "")).split())
    story = [d for item in (answer.get("story") or [])[:MAX_STORY] if (d := _cited(by_id, item, session)) is not None]
    asked = "\n".join(line.text for line in by_id.values() if line.source == "user")
    # A title may repeat what the user asked for ("a 30-day forecast"), never a value only the data holds.
    fits = (
        title
        and len(title) <= TITLE_CHARS
        and (not has_value(title, ex.values, ex.names) or (asked and grounded(title.replace("-", " "), asked)))
    )
    work = str(answer.get("work", "")).strip().lower()
    if not _WORK.fullmatch(work) or has_value(work.replace("-", " "), ex.values, ex.names):
        work = ""
    if not fits and not story and not work:
        return None, items, rejected
    return Story(title=title if fits else "", lines=story, work=work), items, rejected


async def learn(writer: Writer | ReplayWriter, sessions: list[Past], skills: list[Skill], key: str) -> list[Learned]:
    # Lessons across sessions: each one must cite story lines of at least two different sessions.
    if len(sessions) < MIN_LESSON_SESSIONS:
        return []
    by_id: dict[str, tuple[str, Draft]] = {}
    outcomes = {past.session: past.outcome for past in sessions}
    parts: list[str] = []
    for i, past in enumerate(sessions, 1):
        parts.append(f"w{i} [{past.outcome}] {past.title}")
        for j, line in enumerate(past.lines, 1):
            by_id[f"w{i}.{j}"] = (past.session, line)
            parts.append(f"  w{i}.{j} {line.text}")
    held = [f"- {s.name}: {s.when}" + "".join(f"\n    - {line}" for line in s.lines) for s in skills]
    system = LESSONS.format(max_lessons=MAX_LESSONS, item_chars=ITEM_CHARS)
    user = f"Skills already in memory:\n{chr(10).join(held) or '(none yet)'}\n\nSessions:\n" + "\n".join(parts)
    names = {s.name.lower(): s.name for s in skills}
    raw = await writer.complete(system, user, key)
    try:
        found = json.loads(_FENCE.sub("", raw.strip()))["lessons"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"the writer did not answer with lessons JSON: {raw[:200]!r}") from e
    learned: list[Learned] = []
    for raw_lesson in found[:MAX_LESSONS] if isinstance(found, list) else []:
        if not isinstance(raw_lesson, dict):
            continue
        kind = str(raw_lesson.get("kind", "")).strip().lower()
        text = _text(raw_lesson.get("text"), ITEM_CHARS)
        cited = [by_id[s] for s in dict.fromkeys(str(s).strip() for s in raw_lesson.get("steps") or []) if s in by_id]
        sessions_cited = list(dict.fromkeys(sid for sid, _ in cited))
        if kind not in LESSON_KINDS or not text or len(sessions_cited) < MIN_LESSON_SESSIONS or INJECTION.search(text):
            continue
        # A way of working to follow needs a session it worked in; one nothing confirmed only shows what to avoid.
        if kind != "fails" and not any(outcomes.get(sid) == SUCCEEDED for sid in sessions_cited):
            continue
        skill = _text(raw_lesson.get("skill"), SKILL_NAME_CHARS)
        when = _text(raw_lesson.get("when"), ITEM_CHARS)
        skill = names.get(skill.lower(), skill)
        # A new skill needs a name and a sentence on when it fits; a listed one has both already.
        if not skill or (skill.lower() not in names and not when) or INJECTION.search(f"{skill} {when}"):
            continue
        succeeded = sum(outcomes.get(sid) == SUCCEEDED for sid in sessions_cited)
        learned.append(Learned(kind, text, [d.pointer for _, d in cited], sessions_cited, skill, when, succeeded))
    return learned


async def condense(
    writer: Writer | ReplayWriter, sections: dict[str, list[tuple[str, str]]], need: int, key: str
) -> tuple[list[Merge], list[str]]:
    # A full page rewritten: merges stay inside one section, and every id named must be on the page.
    section_of = {ident: name for name, lines in sections.items() for ident, _ in lines}
    listed = "\n\n".join(
        f"{name}:\n" + "\n".join(f"{ident}. {text}" for ident, text in lines) for name, lines in sections.items()
    )
    system = CONDENSE.format(need=need, merged_chars=MERGED_CHARS)
    raw = await writer.complete(system, f"Page lines:\n\n{listed}", key, effort=CONDENSE_EFFORT)
    try:
        answer = json.loads(_FENCE.sub("", raw.strip()))
    except json.JSONDecodeError as e:
        raise ValueError(f"the writer did not answer with merge JSON: {raw[:200]!r}") from e
    if not isinstance(answer, dict):
        return [], []
    used: set[str] = set()
    merges: list[Merge] = []
    for raw_merge in answer.get("merged") or []:
        if not isinstance(raw_merge, dict):
            continue
        ids = [i for i in dict.fromkeys(str(x).strip() for x in raw_merge.get("lines") or []) if i in section_of]
        text = _text(raw_merge.get("text"), MERGED_CHARS)
        if len(ids) < 2 or not text or used & set(ids) or len({section_of[i] for i in ids}) > 1:
            continue
        if INJECTION.search(text):
            continue
        used |= set(ids)
        merges.append(Merge(ids, text))
    named = dict.fromkeys(str(x).strip() for x in answer.get("dropped") or [])
    dropped = [i for i in named if i in section_of and i not in used]
    return merges, dropped
