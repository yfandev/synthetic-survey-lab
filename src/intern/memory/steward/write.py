from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from pydantic import BaseModel, Field

from intern.memory.format import (
    BLOCK_TOKENS,
    PAGE_TOKENS,
    RELATIONS_ID,
    Block,
    FormatError,
    Page,
    has_source,
    inert,
    is_page_file,
    links,
    page_file,
    parse_page,
    source_of,
    strip_source,
    tokens,
    with_seen,
)
from intern.memory.steward.extract import Extract, Observation, UserNote, describe, file_pages
from intern.memory.steward.place import file_page, is_folder_key, lesson_for, name_of
from intern.memory.steward.writer import GENERAL, INSTRUCTIONS, ME, Item, Learned, Story, name_like

GENERAL_LESSONS = f"lessons/{GENERAL}"
USER_FOLDER = "user/"
USER_INSTRUCTIONS = f"{USER_FOLDER}{INSTRUCTIONS}"
TIMELINE = f"{USER_FOLDER}timeline"
HISTORY = "history"
# Pages every task loads are never archived for lack of use.
NEVER_STALE = frozenset({GENERAL_LESSONS, USER_INSTRUCTIONS, ME, "map/environment"})
# Each kind of item has its block; a state that changed moves to Earlier, and a contradiction to Conflicts.
BLOCKS = {
    "where": ("b-where", "Where"),
    "how": ("b-how", "How"),
    "fails": ("b-fails", "Fails"),
    "state": ("b-state", "State"),
    "preference": ("b-state", "State"),
    "rule": ("b-rules", "Rules"),
    "event": ("b-events", "Events"),
    "told": ("b-told", "Told"),
}
EARLIER = ("b-earlier", "Earlier")
CONFLICTS = ("b-conflicts", "Conflicts")
FILES = ("b-files", "Files")
PROGRAMS = ("b-programs", "Programs")
TASK = ("b-task", "For this task only")
# What the session computed or concluded for its task: a later task recomputes it rather than reusing it.
RESULTS = ("b-results", "Results of this task")
TOPIC_SLUG_CHARS = 30
FALLBACK_TITLE_WORDS = 10
_ASKED = re.compile(r"^(?:the )?user (?:asked|wanted|requested)(?: the agent)?(?: to)? ", re.IGNORECASE)
_ATTRIBUTE = re.compile(r"^[a-z][a-z ]{0,24}: ")
# The "(from …, date)" label that ends a line; brackets inside the line stay.
_LINE_LABEL = re.compile(r" \([^()]*\d{4}-\d{2}-\d{2}[^()]*\)$")
LEAD_BLOCKS = ("b-where", "b-state", "b-how", "b-rules", "b-fails")
# Blocks whose oldest lines move to the page's history when the page is full, least useful first;
# where things are, how they work, what fails and rules always stay.
MOVABLE = ("b-told", "b-events", "b-earlier", "b-conflicts")
# What still holds never goes to history, which recall does not load: a full page moves its largest topic to a
# part page of its own, which recall can pick.
MIN_PART_LINES = 2
PAGE_KINDS = ("map", "lesson", "event", "subject", "timeline", "skill")
SKILL_FOLDER = "skills/"
GRAVEYARD = "graveyard/"
USAGE_FILE = ".wiki/usage.json"
# One plain-text transcript per session, which the agent's memory search reads and recall never loads.
SESSIONS_FOLDER = "sessions/"
# Sessions a page may go unused and unwritten before it is archived.
STALE_SESSIONS = 100
_RELATION_LINE = re.compile(r"^- [a-z_]+ \[\[([^\]]+)\]\]$")
# A skill page is a procedure: the steps that worked, the pitfalls to avoid and the checks to always run.
SKILL_BLOCKS = {
    "how": ("b-steps", "Steps"),
    "fails": ("b-pitfalls", "Pitfalls"),
    "rule": ("b-checks", "Checks"),
}
RELATIONS = (
    "about",
    "corrected_by",
    "corrects",
    "has_history",
    "has_part",
    "history_of",
    "part_of",
    "seen_in",
    "touched",
)
SUCCEEDED = "succeeded"
# Said of a work page whose result nothing confirmed, so a later session checks before reusing it.
OUTCOME_NOTES = {
    "unverified": "Nothing in the session checked its result; check before reusing its method or numbers.",
    "partial": "Only part of the task was done; check before reusing its method or numbers.",
    "failed": "The task was not done; its method and numbers are not to be reused as they are.",
    "abandoned": "The session stopped before the task was really attempted.",
}
_UNCONFIRMED = re.compile(r", (?:unverified|partial|failed|abandoned) session\)")
MAX_MOVES = 4
MAX_HISTORY = 20
MAX_ALIASES = 8
LEAD_CHARS = 360
SAID_BY = {"user": "user said", "assistant": "assistant told", "tool": "from", "action": "agent did on"}
GROUP_NOUNS = {
    "people": "a person",
    "organisations": "an organisation",
    "projects": "a project",
    "places": "a place",
    "things": "a thing",
}
LEAD_TRIGGERS = 4
LEAD_ATTRIBUTES = 6
LEAD_DATES = 3
_SENTENCE_BREAK = re.compile(r"[.!?]\s+")

# Room kept on a page for Contents descriptions that System 1 picks after the blocks are written.
CONTENTS_MARGIN_TOKENS = 80
MAX_SEEN_IN = 5
# A map page's lead names what its entries reach, so recall can tell pages apart from the index alone.
MAP_LEAD_WORDS = 12
FILE_LEAD_CHARS = 160
_SEGMENT = re.compile(r"[/.:]")
_VERSION = re.compile(r"v?\d+(\.\d+)*")
_SLUG_WORD = re.compile(r"[a-z0-9]+")
_WORDLIKE = re.compile(r"[A-Za-z]")
# A path segment that names a resource (`messages`, `postMessage`), not a value or a variable (`$id`, `Q1 Sales`).
_RESOURCE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_ON_DATE = re.compile(r"(?i)on \d{4}-\d{2}-\d{2},?\s+")
SAYS = 0.8
SLUG_WORDS = 6
UNTITLED = "Untitled session"
WORK_FOLDER = "work/"
# Front matter of a work page: the session's user turns when it was stewarded, so a later prompt shows.
USER_TURNS = "user_turns"
OTHER_WORK = "other"
PLACE_NOUNS = {
    "service": "service reached over the network",
    "project": "code or package project",
    "documents": "folder of documents and data",
    "environment": "set of programs installed where the agent runs",
    "tool": "connected tool",
    "file": "file",
}


class StewardError(RuntimeError):
    pass


@dataclass
class Gap:
    text: str
    pointer: str


@dataclass
class Written:
    event: str = ""
    title: str = ""
    pages: dict[str, Page] = field(default_factory=dict)
    gaps: list[Gap] = field(default_factory=list)
    # Pages that had no room for a line, when written without spilling to history.
    full: set[str] = field(default_factory=set)


@dataclass
class Placed:
    # An item with the page it belongs to and how it relates to what the page already holds.
    item: Item
    path: str
    # new, same (a restatement or "still holds"), replaces, conflicts, or retires (another line stating the value
    # the item replaced, moved to Earlier)
    change: str = "new"
    old: str = ""
    old_path: str = ""
    # Holds only for this session's task, so it stays on the session's work page.
    task_only: bool = False
    # A value the session computed or concluded for its task, which also stays on the work page.
    result: bool = False


def _base(block_id: str) -> str:
    return "-".join(block_id.split("-")[:2])


def subject_path(group: str, name: str) -> str:
    return f"{USER_FOLDER}{group}/{slug(name)}"


def skill_path(name: str) -> str:
    return f"{SKILL_FOLDER}{slug(name)}"


def is_history(path: str) -> bool:
    last = path.rsplit("/", 1)[-1]
    return last == HISTORY or last.startswith(f"{HISTORY}-")


def source_label(item: Item, outcome: str = "", fallback: str = "") -> str:
    # Who said it, or for a tool result or an action, what it was read from: "from email from Anna".
    said_by = SAID_BY[item.source]
    if item.source in ("tool", "action"):
        # The label is "(from <name>, date)", so a file name's own brackets and commas cannot stay in it.
        name = " ".join(re.sub(r"[,()\[\]]", " ", item.origin or fallback).split())
        said_by = f"{said_by} {name}" if name else ("from a tool" if item.source == "tool" else "agent did")
    mark = f", {outcome} session" if outcome and outcome != SUCCEEDED else ""
    return f"{said_by}, {item.said}{mark}"


def confirmed(line: str) -> str:
    return _UNCONFIRMED.sub(")", line, count=1)


def _from_session(line: str, session_id: str) -> bool:
    return f"[source: evt#{session_id}:" in line


def item_body(item: Item, outcome: str = "", fallback: str = "") -> str:
    text = inert(item.text)
    if item.kind in ("state", "preference") and item.attribute and not text.lower().startswith(item.attribute.lower()):
        text = f"{inert(item.attribute)}: {text}"
    if item.kind == "preference":
        text = f"(preference) {text}"
    if item.kind == "rule" and item.trigger and not _says(text, item.trigger):
        text = f"{text} (applies to: {inert(item.trigger)})"
    if item.kind == "event":
        date = item.when or item.said
        text = _ON_DATE.sub("", text, count=1) if _ON_DATE.match(text) and date in text[:20] else text
        text = text[:1].upper() + text[1:]
        # How the session referred to the date ("this week"), unless it is the date or the line already says it.
        said = item.phrase and item.phrase != date and item.phrase.lower() not in text.lower()
        phrase = f', "{inert(item.phrase)}"' if said else ""
        text = f"{date}{phrase}: {text}"
    return f"{text} ({source_label(item, outcome, fallback)})"


def _says(text: str, words: str) -> bool:
    # A trigger the rule already states ("preserve values ... in records" / "values in records") adds nothing.
    wanted = set(_SLUG_WORD.findall(words.lower()))
    return bool(wanted) and len(wanted & set(_SLUG_WORD.findall(text.lower()))) >= SAYS * len(wanted)


def item_line(item: Item, date: str, outcome: str = "", fallback: str = "") -> str:
    return bullet(item_body(item, outcome, fallback), item.pointer, date)


def earlier(line: str, date: str) -> str:
    source = source_of(line)
    return f"{line[: len(line) - len(source)]} (until {date}){source}"


def bullet(text: str, pointer: str, date: str) -> str:
    return f"- {text} [source: {pointer}; added: {date}]"


def quote(text: str) -> str:
    cleaned = inert(text).replace('"', "'")
    return f'"{cleaned.replace("<", "&lt;")}"'


def slug(title: str) -> str:
    return "-".join(_SLUG_WORD.findall(title.lower())[:SLUG_WORDS]) or "session"


def work_title(work: str) -> str:
    return UNTITLED if work == OTHER_WORK else work.replace("-", " ").capitalize()


def event_path(files: dict[str, str], extract: Extract, title: str, work: str) -> str:
    base = f"{WORK_FOLDER}{work}/{slug(title)}"
    path, n = base, 1
    while page_file(path) in files:
        n += 1
        path = f"{base}-{n}"
    return path


def stewarded(files: dict[str, str], session_id: str) -> bool:
    marker = f"\nsession: {session_id}\n"
    folders = (WORK_FOLDER, f"{GRAVEYARD}{WORK_FOLDER}")
    return any(marker in text for name, text in files.items() if name.startswith(folders))


def user_turns(extract: Extract) -> int:
    return sum(t.role == "user" for t in extract.turns)


def stewarded_turns(files: dict[str, str], session_id: str) -> int | None:
    """How many user turns the session had when its work page was written; none on a page that predates the count.
    A session resumed after its page moved to graveyard/ has a newer page too, which counts more."""
    marker = f"\nsession: {session_id}\n"
    folders = (WORK_FOLDER, f"{GRAVEYARD}{WORK_FOLDER}")
    counts: list[int] = []
    for name, text in files.items():
        if name.startswith(folders) and name.endswith(".md") and marker in text:
            try:
                value = parse_page(name[:-3], text).meta.get(USER_TURNS, "")
            except FormatError:
                continue
            if value.isdigit():
                counts.append(int(value))
    return max(counts, default=None)


def forget(files: dict[str, str], session_id: str) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """The drive without a session's own record (its work page, its transcript and the links to them), the pages
    that lost links and the files removed, so the session can be stewarded again whole. What it taught stays: a
    second pass can drop an item the first kept, and the same item found again confirms the line."""
    marker = f"\nsession: {session_id}\n"
    removed = [name for name, text in files.items() if name.startswith(WORK_FOLDER) and marker in text]
    removed += [name for name in (f"{SESSIONS_FOLDER}{session_id}.txt",) if name in files]
    gone = {name[:-3] for name in removed if is_page_file(name)}
    changed: dict[str, str] = {}
    for name, text in files.items():
        if name in removed or not is_page_file(name) or not any(f"[[{page}]]" in text for page in gone):
            continue
        page = parse_page(name[:-3], text)
        for block in page.blocks:
            block.lines = [line for line in block.lines if not gone.intersection(links(line))]
        page.blocks = [b for b in page.blocks if b.lines or b.id == RELATIONS_ID]
        page.refresh_contents()
        changed[name] = page.render()
    left = {name: text for name, text in files.items() if name not in removed} | changed
    return left, changed, removed


def with_values(schema: str, field: str, wanted: list[str]) -> str:
    lines = schema.splitlines()
    for i, line in enumerate(lines):
        key, sep, value = line.partition(":")
        if sep and key.strip() == field:
            have = [v.strip() for v in value.split(",") if v.strip()]
            if all(w in have for w in wanted):
                return schema
            lines[i] = f"{field}: {', '.join(sorted({*have, *wanted}))}"
            return "\n".join(lines) + "\n"
    return schema.rstrip("\n") + f"\n{field}: {', '.join(sorted(wanted))}\n"


def with_work_kind(schema: str, work: str) -> str:
    return with_values(schema, "work kinds", [work])


def with_vocabulary(schema: str) -> str:
    return with_values(with_values(schema, "kinds", list(PAGE_KINDS)), "relations", list(RELATIONS))


def _load(files: dict[str, str], path: str, kind: str, lead: str) -> Page:
    text = files.get(page_file(path))
    if text is None:
        return Page(path=path, meta={"id": path, "kind": kind}, title=name_of(path), lead=[lead])
    try:
        return parse_page(path, text)
    except FormatError as e:
        raise StewardError(f"{e}; fix the page on main before running the steward") from e


def _fits(page: Page) -> bool:
    page.refresh_contents()
    return tokens(page.render()) <= PAGE_TOKENS - CONTENTS_MARGIN_TOKENS


def _relations(page: Page) -> Block:
    block = page.block(RELATIONS_ID)
    if block is None:
        block = Block(id=RELATIONS_ID, heading="Relations")
        page.blocks.append(block)
    return block


def _identity(line: str) -> str:
    text = strip_source(line)
    return text.split("`")[1] if text.startswith("- `") and text.count("`") >= 2 else text


def topic_of(block: Block) -> str:
    # A block about one part of its page is headed "Where: Calc_Core"; a numbered one goes on with the same part.
    head = block.heading.split(" (", 1)[0]
    return head.split(": ", 1)[1] if ": " in head else ""


def _topic_slug(topic: str) -> str:
    return slug(topic)[:TOPIC_SLUG_CHARS].strip("-") if topic else ""


def append(page: Page, base_id: str, heading: str, bullets: list[str], topic: str = "") -> list[str]:
    existing = {_identity(b) for block in page.blocks for b in block.bullets}
    overflow: list[str] = []
    topic = " ".join(re.sub(r"[:()\[\]{}#]", " ", topic).split())
    tag = _topic_slug(topic)
    for line in bullets:
        if _identity(line) in existing:
            continue
        blocks = [b for b in page.blocks if _base(b.id) == base_id]
        same = [b for b in blocks if _topic_slug(topic_of(b)) == tag]
        target = same[-1] if same else None
        created = target is None or tokens(f"{target.render()}\n{line}") > BLOCK_TOKENS
        if created:
            n = len(same) + 1
            named = f"{heading}: {topic}" if tag else heading
            stem = f"{base_id}-{tag}" if tag else base_id
            while page.block(block_id := stem if n == 1 else f"{stem}-{n}") is not None:
                n += 1
            target = Block(id=block_id, heading=named if n == 1 else f"{named} ({n})")
            # Blocks of one kind stay together, so a page reads by kind and then by part.
            rel = page.block(RELATIONS_ID)
            at = page.blocks.index(blocks[-1]) + 1 if blocks else (page.blocks.index(rel) if rel else len(page.blocks))
            page.blocks.insert(at, target)
        assert target is not None
        target.lines.append(line)
        if not _fits(page):
            target.lines.pop()
            if created:
                page.blocks.remove(target)
            overflow.append(line)
            continue
        existing.add(_identity(line))
    return overflow


def _no_breaks(text: str) -> str:
    # The index shows a lead's first sentence, so a name like "03. Model" must not end it.
    return _SENTENCE_BREAK.sub(lambda m: m[0].replace(".", "").replace("!", "").replace("?", ""), text)


def _first_clause(page: Page) -> str:
    # What the page holds describes it; an event or a reply about it does not.
    ranked = sorted(
        (b for b in page.blocks if _base(b.id) in LEAD_BLOCKS), key=lambda b: LEAD_BLOCKS.index(_base(b.id))
    )
    first = next((strip_source(line)[2:] for block in ranked for line in block.bullets), "")
    clause = _clause(_LINE_LABEL.sub("", first))
    if len(clause) > FILE_LEAD_CHARS:
        clause = clause[:FILE_LEAD_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + " …"
    return clause


def file_lead(page: Page) -> str:
    # The file's path and the start of what is known about it, so recall can pick the page from the index.
    relative = page.meta.get("file") or name_of(page.path)
    owner = page.path.rsplit("/", 1)[0] if page.meta.get("part") else page.path
    folder = name_of(owner.rsplit("/", 1)[0])
    clause = _first_clause(page)
    about = f": {clause}" if clause else ""
    lead = _no_breaks(f"{relative}, a file in {folder}{about}")
    return lead if lead.endswith(("…", ".")) else lead + "."


def _held(page: Page, depth: int) -> list[str]:
    # A folder is told apart by what it holds: its subfolders and files, most used first; a subfolder's part page
    # by the names one level further in.
    held: Counter[str] = Counter()
    for block in page.blocks:
        if _base(block.id) == FILES[0]:
            for line in block.bullets:
                key = "/".join(_identity(line).split("/")[depth:]) or _identity(line)
                held[key.split("/", 1)[0] + ("/" if "/" in key else "")] += 1
    return [_no_breaks(name) for name, _ in held.most_common(MAP_LEAD_WORDS)]


def map_lead(page: Page) -> str:
    kind = page.meta.get("place", "")
    if kind == "file":
        return file_lead(page)
    if page.meta.get("part"):
        # A part of a folder or service is named by its owner and told apart by what it holds.
        clause = _first_clause(page) or ", ".join(_held(page, 1))
        intro = map_intro(page.path.rsplit("/", 1)[0], kind)
        lead = _no_breaks(intro + (f": {clause}" if clause else ""))
        return lead if lead.endswith(("…", ".")) else lead + "."
    if kind in ("documents", "project"):
        names = _held(page, 0)
        return map_intro(page.path, kind) + (f": {', '.join(names)}." if names else ".")
    counts: Counter[str] = Counter()
    for block in page.blocks:
        if block.id == RELATIONS_ID:
            continue
        for line in block.bullets:
            text = strip_source(line)
            if not text.startswith("- `") or text.count("`") < 2:
                continue
            key = text.split("`")[1]
            # A call is "METHOD /path"; a program is named by the command's first word, not its arguments.
            first, _, rest = key.partition(" ")
            target = first if kind == "environment" else rest or key
            host = set(name_of(page.path).lower().split("."))
            for segment in _SEGMENT.split(target):
                if (
                    _RESOURCE.fullmatch(segment)
                    and not _VERSION.fullmatch(segment)
                    and segment.lower() not in host
                    and segment != "api"
                ):
                    counts[segment] += 1
    words = [word for word, _ in counts.most_common(MAP_LEAD_WORDS)]
    return map_intro(page.path, page.meta.get("place", "")) + (f": {', '.join(words)}." if words else ".")


def map_intro(path: str, kind: str) -> str:
    if kind == "environment":
        return "The programs installed where the agent runs"
    return f"{name_of(path)}, a {PLACE_NOUNS.get(kind, 'place')}"


def _clause(text: str) -> str:
    return _SENTENCE_BREAK.sub(" ", text).replace(";", ",").rstrip(".!? ")


def _rule_triggers(page: Page) -> list[str]:
    found: list[str] = []
    for block in page.blocks:
        if block.id.startswith("b-rules"):
            for line in block.bullets:
                text = strip_source(line)
                trigger = text.rsplit("(applies to: ", 1)[1].split(")", 1)[0] if "(applies to: " in text else ""
                found.append(trigger or text[2:].split(" (", 1)[0][:60])
    return list(dict.fromkeys(found))


def subject_lead(page: Page) -> str:
    # One sentence, since the index line is the lead's first sentence: name, aliases, what it holds, rules, dates.
    name = page.meta.get("name") or name_of(page.path)
    aliases = [a.strip() for a in page.meta.get("aliases", "").split(",") if a.strip()]
    noun = GROUP_NOUNS.get(page.meta.get("group", ""), "")
    parts = [name + (f" (also {', '.join(aliases)})" if aliases else "") + (f", {noun}" if noun else "")]
    attributes: list[str] = []
    dates: list[str] = []
    for block in page.blocks:
        for line in block.bullets:
            text = strip_source(line)[2:]
            if block.id.startswith("b-state") and ": " in text:
                attributes.append(text.removeprefix("(preference) ").split(": ", 1)[0])
            if block.id.startswith("b-events"):
                dates.append(text.split(":", 1)[0].split(",", 1)[0])
    # What is known about the subject first, so recall can tell two people of one group apart.
    if clause := _ATTRIBUTE.sub("", _first_clause(page).removeprefix("(preference) ")):
        parts.append(clause)
    if attributes:
        parts.append("known: " + ", ".join(list(dict.fromkeys(attributes))[:LEAD_ATTRIBUTES]))
    if triggers := _rule_triggers(page):
        parts.append("rules on " + ", ".join(triggers[:LEAD_TRIGGERS]))
    if dates:
        parts.append("events on " + ", ".join(sorted(set(dates))[-LEAD_DATES:]))
    return "; ".join(_clause(p) for p in parts)[:LEAD_CHARS] + "."


def with_rules(lead: str, page: Page) -> str:
    triggers = _rule_triggers(page)
    if not triggers:
        return lead
    rules = ", ".join(_clause(t) for t in triggers[:LEAD_TRIGGERS])
    return f"{lead.rstrip('.')}; rules on {rules}"[:LEAD_CHARS] + "."


def relate(page: Page, relation: str, target: str) -> None:
    block = _relations(page)
    line = f"- {relation} [[{target}]]"
    if line in block.lines:
        return
    block.lines.append(line)
    if relation == "seen_in":
        seen = [i for i, existing in enumerate(block.lines) if existing.startswith("- seen_in ")]
        for index in reversed(seen[:-MAX_SEEN_IN]):
            del block.lines[index]
    while not _fits(page) and len(block.lines) > 1:
        del block.lines[0 if block.lines[0] != line else 1]


def write(
    files: dict[str, str],
    extract: Extract,
    outcome: str,
    kept: list[Observation],
    lessons: list[UserNote],
    items: list[Placed] | None = None,
    story: Story | None = None,
    work: str = OTHER_WORK,
    learned: list[Learned] | None = None,
    spill: bool = True,
) -> Written:
    date = extract.date
    title = inert(story.title) if story is not None and story.title else ""
    if not title and story is not None and story.lines:
        # Without a title, the first line of the story says what was asked.
        asked = _ASKED.sub("", story.lines[0].text.strip()).rstrip(".")
        title = inert(" ".join(asked.split()[:FALLBACK_TITLE_WORDS]))
        title = title[:1].upper() + title[1:]
    title = title or work_title(work)
    event = event_path(files, extract, title, work)
    out = Written(event=event, title=title)

    def page(path: str, kind: str, lead: str) -> Page:
        if path not in out.pages:
            out.pages[path] = _load(files, path, kind, lead)
        return out.pages[path]

    place_kinds = {o.place: o.place_kind for o in extract.observations}
    files_of = file_pages(extract.observations)

    def map_page(path: str) -> Page:
        kind = "file" if path in files_of else place_kinds.get(path, "")
        target = page(path, "map", f"{map_intro(path, kind)}.")
        if kind and "place" not in target.meta:
            target.meta["place"] = kind
        if path in files_of:
            target.meta.setdefault("file", files_of[path])
            target.title = PurePosixPath(target.meta["file"]).name
        return target

    def part_path(owner: Page, topic: str) -> str:
        return f"{owner.path}/{_topic_slug(topic)}"

    def existing_part(owner: Page, topic: str) -> Page | None:
        if not topic or owner.meta.get("part") or is_history(owner.path):
            return None
        path = part_path(owner, topic)
        return out.pages.get(path) or (page(path, owner.kind, "") if page_file(path) in files else None)

    def new_part(owner: Page, topic: str) -> Page:
        target = page(part_path(owner, topic), owner.kind, f"{topic}, part of [[{owner.path}]].")
        for key in ("name", "group", "place", "file"):
            if key in owner.meta and key not in target.meta:
                target.meta[key] = owner.meta[key]
        target.meta["part"] = topic
        target.title = f"{owner.title}: {topic}"
        return target

    def files_part(owner: Page, topic: str, left: list[str]) -> list[str]:
        # A subfolder whose files no longer fit on its folder's page moves whole to the subfolder's part page.
        if not topic or owner.meta.get("part") or is_history(owner.path):
            return left
        part = new_part(owner, topic)
        moved = [b for b in owner.blocks if _base(b.id) == FILES[0] and topic_of(b) == topic]
        for block in moved:
            for line in list(block.bullets):
                if not append(part, FILES[0], FILES[1], [line]):
                    block.lines.remove(line)
            if not block.bullets:
                owner.blocks.remove(block)
        left = append(part, FILES[0], FILES[1], left)
        if part.blocks:
            relate(owner, "has_part", part.path)
            relate(part, "part_of", owner.path)
        return left

    by_place: dict[str, list[Observation]] = defaultdict(list)
    for observation in kept:
        # A folder entered on the way to a file says nothing the file's line does not.
        if not (observation.kind == "file" and is_folder_key(observation.key)):
            by_place[observation.place].append(observation)
    for place, observations in by_place.items():
        target = map_page(place)
        groups = (
            (FILES, [o for o in observations if o.kind == "file"]),
            (PROGRAMS, [o for o in observations if o.kind == "shell"]),
            (("b-where", "Where"), [o for o in observations if o.kind not in ("file", "shell")]),
        )
        for (base_id, heading), group in groups:
            # A folder's files are listed by subfolder.
            by_topic: dict[str, list[Observation]] = defaultdict(list)
            for o in group:
                by_topic[o.key.split("/", 1)[0].strip() if o.kind == "file" and "/" in o.key else ""].append(o)
            for topic, observed in by_topic.items():
                lines = {bullet(describe(o), o.pointer, date): o for o in observed}
                holder = (existing_part(target, topic) if base_id == FILES[0] else None) or target
                left = append(holder, base_id, heading, list(lines), topic if holder is target else "")
                if left and spill and base_id == FILES[0] and holder is target:
                    left = files_part(target, topic, left)
                for line in left:
                    out.full.add(place)
                    out.gaps.append(Gap(f"no room on [[{place}]] for {describe(lines[line])}", lines[line].pointer))
        target.lead = [map_lead(target)]
        relate(target, "seen_in", event)

    for note in lessons:
        place = note.place
        if place is None or (place not in out.pages and page_file(place) not in files):
            out.gaps.append(Gap("a correction with no map page to attach it to", note.pointer))
            continue
        lesson = page(lesson_for(place), "lesson", f"Corrections users gave about {name_of(place)}.")
        if append(lesson, "b-corr", "Corrections", [bullet(quote(note.text), note.pointer, date)]):
            out.gaps.append(Gap(f"no room on [[{lesson.path}]] for a correction", note.pointer))
            continue
        relate(lesson, "corrects", place)
        relate(lesson, "seen_in", event)
        relate(map_page(place), "corrected_by", lesson.path)

    def history(owner: Page) -> Page | None:
        n = 1
        while n <= MAX_HISTORY:
            path = f"{owner.path}/{HISTORY}" if n == 1 else f"{owner.path}/{HISTORY}-{n}"
            target = page(path, owner.kind, f"Older lines moved from [[{owner.path}]] when it was full.")
            for key in ("name", "group", "place"):
                if key in owner.meta and key not in target.meta:
                    target.meta[key] = owner.meta[key]
            if _fits(target) and tokens(target.render()) < PAGE_TOKENS - BLOCK_TOKENS:
                return target
            n += 1
        return None

    def split_topic(target: Page) -> bool:
        # The topic with the most text moves whole, so one part of a file or subject reads in one place.
        sizes: Counter[str] = Counter()
        for block in target.blocks:
            if _base(block.id) in LEAD_BLOCKS and (topic := topic_of(block)):
                sizes[topic] += tokens(block.render())
        for topic, _ in sizes.most_common():
            blocks = [b for b in target.blocks if _base(b.id) in LEAD_BLOCKS and topic_of(b) == topic]
            if sum(len(b.bullets) for b in blocks) < MIN_PART_LINES:
                continue
            part = new_part(target, topic)
            moved = False
            for block in blocks:
                heading = block.heading.split(" (", 1)[0].split(": ", 1)[0]
                for line in list(block.bullets):
                    if append(part, _base(block.id), heading, [line]):
                        continue
                    block.lines.remove(line)
                    moved = True
                if not block.bullets:
                    target.blocks.remove(block)
            if moved:
                relate(target, "has_part", part.path)
                relate(part, "part_of", target.path)
                return True
        return False

    def make_room(target: Page) -> bool:
        if is_history(target.path):
            return False
        movable = [b for base in MOVABLE for b in target.blocks if _base(b.id) == base and b.bullets]
        if not movable and not target.meta.get("part") and split_topic(target):
            return True
        block = movable[0] if movable else None
        archive = history(target) if block is not None else None
        if block is None or archive is None:
            return False
        line = block.bullets[0]
        heading = block.heading.split(" (", 1)[0].split(": ", 1)[0]
        if append(archive, _base(block.id), heading, [line], topic_of(block)):
            return False
        block.lines.remove(line)
        if not block.bullets:
            target.blocks.remove(block)
        relate(target, "has_history", archive.path)
        relate(archive, "history_of", target.path)
        return True

    def put(target: Page, base_id: str, heading: str, line: str, topic: str = "") -> bool:
        if base_id in LEAD_BLOCKS and (part := existing_part(target, topic)) is not None:
            target, topic = part, ""
        if not spill:
            # The page is merged before anything moves out of it; the steward writes it again after.
            if not append(target, base_id, heading, [line], topic):
                return True
            out.full.add(target.path)
            return False
        for _ in range(MAX_MOVES + 1):
            if not append(target, base_id, heading, [line], topic):
                return True
            if not make_room(target):
                break
        # Nothing older can move out: a line about one topic starts that topic's part page; any other line goes
        # to the page's history, linked from the page.
        if base_id in LEAD_BLOCKS and topic and not target.meta.get("part") and not is_history(target.path):
            part = new_part(target, topic)
            if not append(part, base_id, heading, [line]):
                relate(target, "has_part", part.path)
                relate(part, "part_of", target.path)
                return True
        archive = None if is_history(target.path) else history(target)
        if archive is None or append(archive, base_id, heading, [line], topic):
            return False
        relate(target, "has_history", archive.path)
        relate(archive, "history_of", target.path)
        return True

    def remove_in_place(target: Page, old: str, new: str) -> bool:
        for block in target.blocks:
            if old in block.lines:
                block.lines[block.lines.index(old)] = new
                return True
        return False

    def remove(target: Page, line: str) -> bool:
        for block in target.blocks:
            if line in block.lines:
                block.lines.remove(line)
                if not block.bullets and block.id != RELATIONS_ID:
                    target.blocks.remove(block)
                return True
        return False

    def item_page(placed: Placed) -> Page:
        it, path = placed.item, placed.path
        if path == GENERAL_LESSONS:
            return page(path, "lesson", "Working rules the agent learned that hold everywhere.")
        if path == USER_INSTRUCTIONS:
            return page(path, "lesson", "Standing instructions the user gave that hold everywhere.")
        if not path.startswith(USER_FOLDER):
            return map_page(path)
        target = page(path, "subject", "")
        if path == ME:
            target.meta.setdefault("name", "the user")
        else:
            target.meta.setdefault("name", inert(it.subject) if it.subject else name_of(path))
            target.meta.setdefault("group", it.group or path.split("/")[1])
        target.title = target.meta["name"]
        known = [a.strip() for a in target.meta.get("aliases", "").split(",") if a.strip()]
        names = {target.meta["name"].lower(), *(a.lower() for a in known), "user", "the user", "me", "i"}
        own = [] if path == ME else [it.subject]
        added = [
            inert(a)
            for a in [*own, *it.aliases]
            if a and a.lower() not in names and "," not in a and name_like(a, target.meta["name"])
        ]
        if added:
            target.meta["aliases"] = ", ".join(list(dict.fromkeys([*known, *added]))[:MAX_ALIASES])
        target.lead = [subject_lead(target)]
        return target

    def old_page(path: str) -> Page | None:
        return page(path, "", "") if path in out.pages or page_file(path) in files else None

    def fallback(target: Page | None, pointer: str) -> str:
        # What a tool's line was read from when the writer did not say: the file of its page, or the service.
        if target is not None and target.meta.get("place") == "file" and target.meta.get("file"):
            return PurePosixPath(target.meta["file"]).name
        cited = extract.places.get(pointer, [])
        service = next((name_of(p) for p in cited if place_kinds.get(p) in ("service", "tool")), "")
        shells = [o for o in extract.observations if o.pointer == pointer and o.kind == "shell"]
        program = shells[0].key.split(" ", 1)[0] if shells else ""
        return service or program

    task_lines: dict[str, str] = {}
    result_lines: dict[str, str] = {}
    used = set(extract.used)
    for placed in items or []:
        it = placed.item
        if placed.change == "retires":
            holder = old_page(placed.old_path)
            if holder is not None and remove(holder, placed.old):
                put(holder, *EARLIER, earlier(placed.old, it.said))
            continue
        if placed.task_only or placed.result:
            into = result_lines if placed.result else task_lines
            into[item_line(it, date, outcome, fallback(None, it.pointer))] = it.pointer
            continue
        target = item_page(placed)
        line = item_line(it, date, outcome, fallback(target, it.pointer))
        base_id, heading = BLOCKS[it.kind]
        # A line is confirmed by a succeeded session that finds it still true, or by a session that found it
        # again on its own: one that did not read it from the page first.
        independent = target.path not in used and not _from_session(placed.old, extract.session_id)
        seen = with_seen(confirmed(placed.old) if outcome == SUCCEEDED or independent else placed.old, date)
        holder = target if placed.old_path in ("", target.path) else old_page(placed.old_path)
        if placed.change == "same" and holder is not None and remove_in_place(holder, placed.old, seen):
            relate(holder, "seen_in", event)
            continue
        if placed.change == "replaces" and holder is not None and remove(holder, placed.old):
            if not put(holder, *EARLIER, earlier(placed.old, it.said)):
                out.gaps.append(Gap(f"no room on [[{holder.path}]] for an earlier value", it.pointer))
        elif placed.change == "conflicts" and holder is not None and remove(holder, placed.old):
            if not put(holder, *CONFLICTS, placed.old):
                out.gaps.append(Gap(f"no room on [[{holder.path}]] for a conflicting value", it.pointer))
            if holder is target:
                base_id, heading = CONFLICTS
        if not put(target, base_id, heading, line, it.topic):
            out.gaps.append(Gap(f"no room on [[{target.path}]] for: {inert(it.text)}", it.pointer))
            continue
        relate(target, "seen_in", event)
        if it.kind == "event":
            month = (it.when or it.said)[:7]
            timeline = page(f"{TIMELINE}/{month}", "timeline", f"Dated events of {month}, one line each.")
            timeline.title = f"Events of {month}"
            entry = bullet(
                f"{item_body(it, outcome, fallback(target, it.pointer))} on [[{target.path}]]", it.pointer, date
            )
            if not put(timeline, "b-events", "Events", entry):
                out.gaps.append(Gap(f"no room on [[{timeline.path}]] for an event", it.pointer))

    for lesson in learned or []:
        name = inert(lesson.skill)
        target = page(skill_path(name), "skill", f"{name}: {_no_breaks(_clause(inert(lesson.when)))}.")
        target.meta.setdefault("name", name)
        target.title = target.meta["name"]
        # A general lesson a skill now holds moves to the skill, under the wording drawn across sessions.
        if lesson.old and lesson.old_page and lesson.old_page != target.path:
            if (holder := old_page(lesson.old_page)) is not None:
                remove(holder, lesson.old)
        elif lesson.old and remove_in_place(target, lesson.old, with_seen(lesson.old, date)):
            relate(target, "seen_in", event)
            continue
        body = (
            f"{inert(lesson.text)} (learned from {len(lesson.sessions)} sessions, {lesson.succeeded} succeeded, {date})"
        )
        if not put(target, *SKILL_BLOCKS[lesson.kind], bullet(body, lesson.pointers[-1], date)):
            out.gaps.append(
                Gap(f"no room on [[{target.path}]] for a lesson: {inert(lesson.text)}", lesson.pointers[-1])
            )
            continue
        relate(target, "seen_in", event)

    # A folder's file line links the file's page, and the page links back to its folder.
    for path, target in list(out.pages.items()):
        if target.meta.get("place") == "file":
            folder = path.rsplit("/", 1)[0]
            if folder in out.pages or page_file(folder) in files:
                relate(target, "part_of", folder)
            continue
        folder = path.rsplit("/", 1)[0] if target.meta.get("part") else path
        for block in target.blocks:
            if _base(block.id) != FILES[0]:
                continue
            for i, line in enumerate(block.lines):
                linked = file_page(folder, _identity(line)) if has_source(line) and "[[" not in line else ""
                if not linked or (linked not in out.pages and page_file(linked) not in files):
                    continue
                block.lines[i] = f"{strip_source(line)} → [[{linked}]]{source_of(line)}"
                if tokens(block.render()) > BLOCK_TOKENS or not _fits(target):
                    block.lines[i] = line

    for path, target in list(out.pages.items()):
        if is_history(path) and not target.blocks:
            del out.pages[path]
            continue
        if is_history(path) or target.kind not in ("subject", "map"):
            continue
        old = target.lead
        lead = subject_lead(target) if target.kind == "subject" else with_rules(map_lead(target), target)
        target.lead = [f"{target.meta['part']}, part of {lead}" if target.meta.get("part") else lead]
        if not _fits(target):
            target.lead = old

    note = f" {OUTCOME_NOTES[outcome]}" if outcome in OUTCOME_NOTES else ""
    record = page(event, "event", f"{title}: `{outcome}` (session `{extract.session_id}`).{note}")
    record.meta["session"] = extract.session_id
    record.meta["date"] = extract.turns[0].at if extract.turns else extract.date
    record.meta[USER_TURNS] = str(user_turns(extract))
    record.title = title
    if story is not None:
        lines = [bullet(inert(line.text), line.pointer, date) for line in story.lines]
        append(record, "b-story", "What happened", lines)
    for line in append(record, *RESULTS, list(result_lines)):
        out.gaps.append(Gap(f"no room on [[{event}]] for a result: {strip_source(line)}", result_lines[line]))
    for line in append(record, *TASK, list(task_lines)):
        out.gaps.append(
            Gap(f"no room on [[{event}]] for a line about this task: {strip_source(line)}", task_lines[line])
        )
    append(record, "b-corr", "Corrections", [bullet(quote(n.text), n.pointer, date) for n in lessons])
    touched = [p for p in out.pages if p.startswith("map/") and not is_history(p)]
    for place in dict.fromkeys([*by_place, *touched]):
        relate(record, "touched", place)
    for path in dict.fromkeys(p.path for p in items or []):
        if path.startswith(USER_FOLDER) and path in out.pages:
            relate(record, "about", path)
    return out


class PageUse(BaseModel):
    # Counted in steward sessions: how often a page was recalled or read, and when it was last used or written.
    used: int = 0
    last_used: int = 0
    written: int = 0


class Usage(BaseModel):
    sessions: int = 0
    pages: dict[str, PageUse] = Field(default_factory=dict)


def read_usage(text: str | None) -> Usage:
    if not text:
        return Usage()
    try:
        return Usage.model_validate_json(text)
    except ValueError:
        return Usage()


def tally(usage: Usage, live: list[str], used: list[str], written: list[str]) -> Usage:
    # A page first seen gets a full grace period; one never used and never written again goes stale.
    seq = usage.sessions + 1
    pages = {p: usage.pages.get(p) or PageUse(written=seq) for p in live}
    for path in written:
        if path in pages:
            pages[path].written = seq
    for path in used:
        if path in pages:
            pages[path].used += 1
            pages[path].last_used = seq
    return Usage(sessions=seq, pages=pages)


def stale(usage: Usage) -> list[str]:
    def idle(path: str) -> bool:
        use = usage.pages.get(path)
        return use is None or usage.sessions - max(use.last_used, use.written) > STALE_SESSIONS

    # A history page is read through its page, so it goes only with it.
    return [
        path
        for path in usage.pages
        if path not in NEVER_STALE and idle(path) and not (is_history(path) and not idle(path.rsplit("/", 1)[0]))
    ]


def _unlinked(line: str, gone: set[str]) -> str | None:
    # A link to an archived page: a relation line goes, a file link goes, any other link reads as plain text.
    found = _RELATION_LINE.match(line)
    if found and found[1] in gone:
        return None
    for path in gone:
        line = line.replace(f" → [[{path}]]", "").replace(f"[[{path}]]", f"{path} (archived)")
    return line


def archive(files: dict[str, str], paths: list[str]) -> tuple[dict[str, str], list[str]]:
    # Unused pages move to graveyard/, where recall never looks; the pages that linked them drop the links.
    gone = {p for p in paths if page_file(p) in files}
    changed = {f"{GRAVEYARD}{page_file(p)}": files[page_file(p)] for p in gone}
    for name, text in files.items():
        if not is_page_file(name) or name[:-3] in gone or "[[" not in text:
            continue
        try:
            page = parse_page(name[:-3], text)
        except FormatError:
            continue
        for block in page.blocks:
            block.lines = [new for line in block.lines if (new := _unlinked(line, gone)) is not None]
        page.blocks = [b for b in page.blocks if b.lines or b.id == RELATIONS_ID]
        page.refresh_contents()
        if page.render() != text:
            changed[name] = page.render()
    return changed, sorted(page_file(p) for p in gone)
