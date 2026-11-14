from __future__ import annotations

import math
import re

from pydantic import BaseModel, Field

CHARS_PER_TOKEN = 3
BLOCK_TOKENS = 400
PAGE_TOKENS = 1_500
MAX_TOC_ENTRIES = 255

ROOT_INDEX = "MEMORY.md"
FOLDER_INDEX = "index.md"
SCHEMA_FILE = ".wiki/schema.md"
# The one world the drive's sessions ran in; written by the first stewarded session that names one.
WORLD_FILE = ".wiki/world.md"
UNREAD_FOLDERS = (".wiki/", "graveyard/")

LEAD_ID = "b-lead"
TOC_ID = "b-toc"
RELATIONS_ID = "b-rel"

# What each top folder is for, in one sentence; its index line adds what it holds now.
DEFAULT_LEADS = {
    "": "What earlier sessions learned here.",
    "map/": "Where things are and how to use them, one page per place.",
    "lessons/": "Working rules that hold everywhere, and corrections users gave.",
    "work/": "Past sessions, one folder per kind of work.",
    "user/": "The user's world, one page per subject.",
    "skills/": "Ways of working learned across sessions, one page per kind of work.",
}
# Leads earlier stewards wrote; an index that still has one gets the current lead.
_OLD_LEADS = frozenset(
    {
        "What earlier sessions learned in this workspace: where things are and how they are used.",
        "Where things are and how to use them, one page per place the agent works with: "
        "a service, a project, a folder of documents, a connected tool, or the environment.",
        "Kinds of work the agent did, one folder per kind: each session, how it went and what was left out.",
        "Kinds of work the agent did, one folder per kind, one page per session: what was asked and done.",
        "What the user said and what holds about them and the people, organisations, projects, places and things "
        "they deal with, one page per subject, with their standing instructions and a dated timeline.",
    }
)

_HEADING = re.compile(r"^(#{1,2}) (.+?) \{#(b-[a-z0-9-]+)\}\s*$")
_LINK = re.compile(r"\[\[([^\]]+)\]\]")
_SOURCE = re.compile(r" \[source: evt#[^;\]]+; added: \d{4}-\d{2}-\d{2}(?:; seen: \d{4}-\d{2}-\d{2})?\]$")
_SEEN = re.compile(r"(; added: \d{4}-\d{2}-\d{2})(?:; seen: \d{4}-\d{2}-\d{2})?\]$")
_CONTENTS_LINE = re.compile(r"^- (b-[a-z0-9-]+): (.+) · (\d+) tok$")
_RELATION_LINE = re.compile(r"^- ([a-z_]+) \[\[([^\]]+)\]\]$")
_INDEX_PAGE = re.compile(r"^- \[\[([^\]]+)\]\]: (.+) · (\d+) tok$")
_INDEX_FOLDER = re.compile(r"^- ([^\s\[\]]+/): (.+)$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_BACKTICKED = re.compile(r"`([a-z_]+)`")
# A folder's index line names this many of its pages, so a reader can tell folders apart.
SUMMARY_TITLES = 3


class FormatError(ValueError):
    pass


def tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def links(text: str) -> list[str]:
    return _LINK.findall(text)


def drive_world(files: dict[str, str]) -> str | None:
    found = re.search(r"^world: (.+)$", files.get(WORLD_FILE, ""), re.MULTILINE)
    return found[1].strip() if found else None


def world_file(world: str) -> str:
    note = "Every session on this drive ran in one world; a session of another world needs its own drive."
    return f"# World\n\n{note}\n\nworld: {world}\n"


def inert(text: str) -> str:
    # Session text (code, tool output) must not read as a link or a source tag on the page.
    return text.replace("[[", "[ [").replace("]]", "] ]").replace("[source:", "(source:")


def clip(text: str, limit: int) -> str:
    # A client may put its own instructions before the user's words (T3 Code's run about 5 KB), so a long text
    # keeps its end as well as its start.
    if len(text) <= limit:
        return text
    head = limit // 5
    return f"{text[:head]} [...] {text[-(limit - head) :]}"


def has_source(line: str) -> bool:
    return bool(_SOURCE.search(line))


def strip_source(line: str) -> str:
    return _SOURCE.sub("", line)


def source_of(line: str) -> str:
    found = _SOURCE.search(line)
    return found[0] if found else ""


def with_seen(line: str, date: str) -> str:
    return _SEEN.sub(lambda m: f"{m[1]}; seen: {date}]", line)


def first_sentence(text: str) -> str:
    return _SENTENCE_END.split(text.strip(), maxsplit=1)[0]


class Block(BaseModel):
    id: str
    heading: str
    lines: list[str] = Field(default_factory=list)

    def render(self) -> str:
        body = "\n".join(self.lines)
        return f"## {self.heading} {{#{self.id}}}" + (f"\n{body}" if body else "")

    @property
    def bullets(self) -> list[str]:
        return [line for line in self.lines if line.startswith("- ")]


class ContentsLine(BaseModel):
    block_id: str
    description: str
    tokens: int

    def render(self) -> str:
        return f"- {self.block_id}: {self.description} · {self.tokens} tok"


class Relation(BaseModel):
    relation: str
    target: str


class Page(BaseModel):
    path: str
    meta: dict[str, str]
    title: str
    lead: list[str]
    contents: list[ContentsLine] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)

    @property
    def kind(self) -> str:
        return self.meta.get("kind", "")

    @property
    def description(self) -> str:
        return first_sentence(" ".join(self.lead)) or self.title

    def block(self, block_id: str) -> Block | None:
        return next((b for b in self.blocks if b.id == block_id), None)

    def relations(self) -> list[Relation]:
        block = self.block(RELATIONS_ID)
        found: list[Relation] = []
        for line in block.lines if block else []:
            if match := _RELATION_LINE.match(line):
                found.append(Relation(relation=match[1], target=match[2]))
        return found

    def lead_text(self) -> str:
        return "\n".join(self.lead)

    def refresh_contents(self, descriptions: dict[str, str] | None = None) -> None:
        old = {line.block_id: line.description for line in self.contents}
        given = descriptions or {}
        self.contents = [
            ContentsLine(
                block_id=b.id,
                description=given.get(b.id) or old.get(b.id) or b.heading.lower(),
                tokens=tokens(b.render()),
            )
            for b in self.blocks
        ]

    def render(self) -> str:
        meta = "\n".join(f"{k}: {v}" for k, v in self.meta.items())
        parts = [
            f"# {self.title} {{#{LEAD_ID}}}\n" + "\n".join(self.lead),
            f"## Contents {{#{TOC_ID}}}\n" + "\n".join(line.render() for line in self.contents),
            *(b.render() for b in self.blocks),
        ]
        return f"---\n{meta}\n---\n" + "\n\n".join(p.rstrip() for p in parts) + "\n"


def page_file(path: str) -> str:
    return f"{path}.md"


def is_page_file(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return (
        name.endswith(".md")
        and name != ROOT_INDEX
        and base != FOLDER_INDEX
        and not name.startswith(UNREAD_FOLDERS)
        and "/" in name
    )


def folder_of(path: str) -> str:
    return path.rsplit("/", 1)[0] + "/" if "/" in path else ""


def parent_folder(folder: str) -> str:
    return folder_of(folder.rstrip("/"))


def index_file(folder: str) -> str:
    return ROOT_INDEX if not folder else f"{folder}{FOLDER_INDEX}"


def parse_page(path: str, text: str) -> Page:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---" or "---" not in [line.strip() for line in lines[1:]]:
        raise FormatError(f"{path}: no front matter")
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if not sep:
            raise FormatError(f"{path}: front matter line {line!r} is not `key: value`")
        meta[key.strip()] = value.strip()

    sections: list[tuple[str, str, list[str]]] = []
    for line in lines[end + 1 :]:
        if match := _HEADING.match(line):
            sections.append((match[3], match[2], []))
        elif sections:
            sections[-1][2].append(line)
        elif line.strip():
            raise FormatError(f"{path}: text before the lead heading")
    if not sections or sections[0][0] != LEAD_ID:
        raise FormatError(f"{path}: the first heading must be `# <title> {{#{LEAD_ID}}}`")

    def trimmed(body: list[str]) -> list[str]:
        while body and not body[-1].strip():
            body = body[:-1]
        while body and not body[0].strip():
            body = body[1:]
        return body

    _, title, lead = sections[0]
    contents: list[ContentsLine] = []
    blocks: list[Block] = []
    for block_id, heading, body in sections[1:]:
        body = trimmed(body)
        if block_id == TOC_ID:
            for line in body:
                match = _CONTENTS_LINE.match(line)
                if not match:
                    raise FormatError(f"{path}: Contents line {line!r} is not `- <block-id>: <description> · <N> tok`")
                contents.append(ContentsLine(block_id=match[1], description=match[2], tokens=int(match[3])))
        else:
            blocks.append(Block(id=block_id, heading=heading, lines=body))
    return Page(path=path, meta=meta, title=title, lead=trimmed(lead), contents=contents, blocks=blocks)


class IndexEntry(BaseModel):
    target: str = Field(description="A page path, or a folder ending in `/`.")
    description: str
    tokens: int | None = None

    @property
    def folder(self) -> bool:
        return self.target.endswith("/")

    def render(self) -> str:
        if self.folder:
            return f"- {self.target}: {self.description}"
        return f"- [[{self.target}]]: {self.description} · {self.tokens} tok"


class Index(BaseModel):
    title: str
    lead: str
    entries: list[IndexEntry] = Field(default_factory=list)

    def render(self) -> str:
        entries = "\n".join(e.render() for e in self.entries)
        return f"# {self.title}\n{self.lead}\n" + (f"\n{entries}\n" if entries else "")


def parse_index(name: str, text: str) -> Index:
    lines = [line for line in text.splitlines()]
    if not lines or not lines[0].startswith("# "):
        raise FormatError(f"{name}: the first line must be `# <title>`")
    lead: list[str] = []
    entries: list[IndexEntry] = []
    for line in lines[1:]:
        if page := _INDEX_PAGE.match(line):
            entries.append(IndexEntry(target=page[1], description=page[2], tokens=int(page[3])))
        elif folder := _INDEX_FOLDER.match(line):
            entries.append(IndexEntry(target=folder[1], description=folder[2]))
        elif line.startswith("- "):
            raise FormatError(f"{name}: entry {line!r} is not `- [[path]]: ... · N tok` or `- folder/: ...`")
        elif line.strip() and not entries:
            lead.append(line.strip())
    return Index(title=lines[0][2:].strip(), lead=" ".join(lead), entries=entries)


class Schema(BaseModel):
    kinds: list[str]
    relations: list[str]
    event_kinds: list[str] = Field(default_factory=list)
    work_kinds: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)


def parse_schema(text: str) -> Schema:
    fields: dict[str, list[str]] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and not line.startswith(("#", "-")):
            fields[key.strip().replace(" ", "_")] = [v.strip() for v in value.split(",") if v.strip()]
    try:
        return Schema.model_validate(fields)
    except ValueError as e:
        raise FormatError(f"{SCHEMA_FILE}: {e}") from e


def build_indexes(files: dict[str, str]) -> dict[str, str]:
    pages: dict[str, Page] = {}
    sizes: dict[str, int] = {}
    for name, text in files.items():
        if is_page_file(name):
            try:
                pages[name[:-3]] = parse_page(name[:-3], text)
            except FormatError:
                continue
            sizes[name[:-3]] = tokens(text)
    folders = {""}
    for path in pages:
        folder = folder_of(path)
        while folder:
            folders.add(folder)
            folder = parent_folder(folder)

    outcomes: list[str] = []
    if SCHEMA_FILE in files:
        try:
            outcomes = parse_schema(files[SCHEMA_FILE]).outcomes
        except FormatError:
            outcomes = []

    def counts(folder: str) -> str:
        inside = [p for p in sorted(pages) if p.startswith(folder)]
        text = f"{len(inside)} page{'' if len(inside) == 1 else 's'}"
        found = [o for p in inside for o in _BACKTICKED.findall(" ".join(pages[p].lead)) if o in outcomes]
        if found:
            text += ": " + ", ".join(f"{found.count(o)} {o}" for o in outcomes if o in found)
        return text

    def names(inside: list[str]) -> str:
        # "and 1 more" says less than the one name it stands for.
        shown = inside if len(inside) <= SUMMARY_TITLES + 1 else inside[:SUMMARY_TITLES]
        more = len(inside) - len(shown)
        return ", ".join(pages[p].title for p in shown) + (f" and {more} more" if more else "")

    def summary(folder: str) -> str:
        parts = names([p for p in sorted(pages) if folder_of(p) == folder])
        subfolders = sorted(f for f in folders if f and parent_folder(f) == folder)
        for sub in subfolders:
            inside = [p for p in sorted(pages) if p.startswith(sub)]
            parts += ("; " if parts else "") + f"{sub.removeprefix(folder).rstrip('/')} ({names(inside)})"
        return f"{counts(folder)}. {parts}."

    leads: dict[str, str] = {}
    titles: dict[str, str] = {}
    for folder in folders:
        name = index_file(folder)
        existing = None
        if name in files:
            try:
                existing = parse_index(name, files[name])
            except FormatError:
                existing = None
        leads[folder] = DEFAULT_LEADS.get(folder) or f"{counts(folder)}."
        if folder in DEFAULT_LEADS and existing and existing.lead and existing.lead not in _OLD_LEADS:
            leads[folder] = existing.lead
        titles[folder] = (existing.title if existing else "") or (folder.rstrip("/") or "Memory")

    out: dict[str, str] = {}
    for folder in folders:
        children = sorted(f for f in folders if f and parent_folder(f) == folder)
        entries = [
            IndexEntry(
                target=f,
                description=(f"{first_sentence(leads[f])} " if f in DEFAULT_LEADS else "") + summary(f),
            )
            for f in children
        ]
        entries += [
            IndexEntry(target=path, description=page.description, tokens=sizes[path])
            for path, page in sorted(pages.items())
            if folder_of(path) == folder
        ]
        out[index_file(folder)] = Index(title=titles[folder], lead=leads[folder], entries=entries).render()
    return out
