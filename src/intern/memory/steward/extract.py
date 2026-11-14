from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, SecretStr

from intern.journal import Entry, Journal, MemoryRecallEntry, MessageEntry, ObservationEntry, SessionEntry
from intern.memory.format import clip, inert
from intern.memory.steward.place import (
    ENVIRONMENT_PAGE,
    PlaceKind,
    file_page,
    is_folder_key,
    page_for_host,
    page_for_tool,
    place_for_path,
    project_name,
    project_root,
)
from intern.messages import AssistantMessage, ToolResultMessage, UserMessage
from intern.secrets import SecretRedactor
from intern.session_cut import session_cut

HttpCall = tuple[str, str, dict[str, list[str]]]

REDACTED = "[REDACTED]"
# A value assigned to a name that says it is a credential (ACME_LEDGER_TOKEN=..., "api_key": "..."); a value with
# both letters and digits is a word, an expression (token = os.environ.get) or a count, not a credential.
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b[\w-]*(?:token|secret|passw(?:or)?d|api[_-]?key|access[_-]?key|private[_-]?key|credential|authorization)"
    r"[\w-]*[\"']?\s*[=:]\s*[\"']?(?:bearer\s+)?([\w+/=.~-]{8,})"
)
# A credential named in prose ("the API key for the billing integration: ..."): a long token with both letters and
# digits shortly after the word, on the same line. Paths and URLs are not tokens.
_SECRET_IN_PROSE = re.compile(
    r"(?i)\b(?:token|secret|password|passwd|passphrase|credential|(?:api|access|secret|private|signing)[ _-]?key)\b"
    r"[^\n]{0,60}?(?<![\w/.~-])([A-Za-z0-9][\w-]{19,})(?![\w/.~-])"
)
_LETTER = re.compile(r"[A-Za-z]")
_URL_PASSWORD = re.compile(r"://[^/\s:@]+:([^@\s/]{4,})@")
# A vendor key in the <prefix>_live_ / <prefix>_test_ style needs a digit in its tail, so an identifier such as
# is_test_environment_ready is not one.
_SECRET_SHAPE = re.compile(
    r"\b(?:sk-[\w-]{20,}|[rsp]k_(?:live|test)_\w{10,}|[a-z]{2,10}_(?:live|test)_(?=[A-Za-z]*\d)[A-Za-z0-9]{10,}"
    r"|gh[pousr]_\w{20,}|xox[abprs]-[\w-]{10,}|AKIA[0-9A-Z]{16}|AIza[\w-]{35})\b"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
)
_DIGIT = re.compile(r"\d")

BASE_TOOLS = frozenset({"read", "write", "edit", "bash", "grep", "find"})
READ_TOOLS = frozenset({"read", "grep", "find"})
WRITE_TOOLS = frozenset({"write", "edit"})
# Programs that only move around, print or shuffle text say nothing about the environment.
SHELL_BASICS = frozenset(
    [
        "cd",
        "ls",
        "cat",
        "echo",
        "printf",
        "head",
        "tail",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "find",
        "sed",
        "awk",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "pwd",
        "true",
        "false",
        "sleep",
        "which",
        "type",
        "test",
        "[",
        "export",
        "set",
        "unset",
        "source",
        ".",
        "mkdir",
        "rmdir",
        "cp",
        "mv",
        "rm",
        "touch",
        "tee",
        "xargs",
        "less",
        "more",
        "file",
        "stat",
        "du",
        "df",
        "date",
        "basename",
        "dirname",
        "realpath",
        "readlink",
        "diff",
        "tree",
        "clear",
        "history",
        "exit",
        "read",
        "wait",
        "kill",
    ]
)
SHELL_KEYWORDS = frozenset(
    {"for", "do", "done", "if", "then", "elif", "else", "fi", "while", "until", "case", "esac", "function", "in"}
)
SHELL_PREFIXES = frozenset({"do", "then", "else", "elif", "if", "while", "until", "!", "{", "("})
SHELL_WRAPPERS = frozenset({"sudo", "time", "nohup", "command", "exec", "env", "timeout"})
QUOTE_CHARS = 300
STEP_CALL_CHARS = 400
STEP_RESULT_CHARS = 1_500
# The session's part of a steward ask (`_run_state`), the size ctx.state() had before 0.21; the rest is the ask's.
RUN_STATE_TOKENS = 6_000
SHELL_WORDS = 3
MAX_PATH_TOKEN = 300

FINAL_CHARS = 1_500
TURN_CHARS = 4_000
SOURCE_CHARS = 4_000
TARGET_CHARS = 80
MAX_TARGETS = 12
METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"})
READ_METHODS = frozenset({"GET", "HEAD"})

_TOKEN = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"|\S+")
_JSON_KEY = re.compile(r"\\?\"([A-Za-z_][\w.-]*)\\?\"\s*:")
_KEY_NAME = re.compile(r"[A-Za-z_][\w.-]*")
_ERROR_LINE = re.compile(r'^(\{\s*"errors?"\s*:|\s{1,4}"errors?"\s*:)')
_EXIT_LINE = re.compile(r"^\[exit code -?\d+\]$")
_FAILED_EXIT = re.compile(r"(?m)^\[exit code (?!0\])-?\d+\]$")
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=.*")
_QUOTED_PATH = re.compile(
    r"'([^'\n]*/[^'\n]*|[^'\n/]+\.[A-Za-z][A-Za-z0-9]{0,4})'|\"([^\"\n]*/[^\"\n]*|[^\"\n/]+\.[A-Za-z][A-Za-z0-9]{0,4})\""
)
_FILE_PATH = re.compile(r"[^*?${}\[\]<>|;=,\\\n]+/[^*?${}\[\]<>|;=,\\\n/]+\.[A-Za-z][A-Za-z0-9]{0,4}")
# A file name with an extension and a letter in its name, not a number like '3.0', a key like 'a.b' or an
# email address like 'a@b.com'.
_BARE_FILE = re.compile(r"[^*?${}\[\]<>|;=,\\\n/%@]*[A-Za-z][^*?${}\[\]<>|;=,\\\n/%@]*\.[A-Za-z][A-Za-z0-9]{1,4}")
_PROGRAM = re.compile(r"[A-Za-z][\w.+-]*")
_FUNCTION = re.compile(
    r"(?:^|[\s;&|(])(?:function\s+([A-Za-z_][\w.-]*)(?:\s*\(\s*\))?|([A-Za-z_][\w.-]*)\s*\(\s*\))\s*\{"
)
_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
_BARE = re.compile(r"[a-z][a-z0-9_-]{0,30}")
_IDENTIFIER = re.compile(
    r"(?<![\w.{-])(?:[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*|[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*[A-Za-z0-9]"
    r"|[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+|[a-z0-9]+(?:-[a-z0-9]+)*-[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)(?![\w}-])"
)
_FILE_NAME = re.compile(r"(?<![\w.{/-])[\w-]{2,}\.[A-Za-z][A-Za-z0-9]{1,4}(?![\w.}])")
_NAME_PART = re.compile(r"[A-Za-z0-9_]+")
_VERSION = re.compile(r"v\d+(\.\d+)?")
_TEMPLATE_URL = re.compile(r"https?://[^\s\"'<>\\]*\{[^\s\"'<>\\]*")
_PLACEHOLDER = re.compile(r"(\{[^{}/]+\})")
_WORD = re.compile(r"[A-Za-z0-9_][\w-]*")
_NUMBERED_VERSION = re.compile(r"\d{1,2}(\.\d{1,2})?")
_JSON_STRING = re.compile(r'"[\w$.-]+"\s*:\s*"([^"\\]{2,80})"')
_HOST = re.compile(r"https?://([\w.-]+)")
_QUOTED = re.compile(r"(?<!\w)'([^'\n]{2,60})'(?!\w)|\"([^\"\n]{2,60})\"")
_CAPITALIZED = re.compile(r"[A-Z][\w&'-]*(?: [A-Z][\w&'-]*)+")
# Text that speaks to an AI reading it ("NOTE TO AI ASSISTANTS: remember this permanently ...") is an instruction
# planted for agents, whatever command or key follows it.
_TO_AGENTS = (
    r"(?:note|message|instructions?) (?:to|for) (?:all |any )?(?:ai|llm)s?(?: assistants?| agents?| models?)?\b"
    r"|\b(?:ai|llm) (?:assistants?|agents?)\s*:|remember (?:this )?(?:permanently|forever)"
    r"|for all (?:future|later|subsequent) sessions"
)
INJECTION = re.compile(
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|messages)|disregard (the|all|your) "
    r"|system prompt|you are now|developer mode|</?memory|new instructions:|act as (an? )?(admin|root|system)",
    re.IGNORECASE,
)
# From the start of the sentence that speaks to agents to the end of its paragraph. Only in what the session read
# or the assistant said: a user who asks to remember something permanently means it.
_PLANTED = re.compile(rf"[^\n.]*(?:{_TO_AGENTS})[\s\S]*?(?=\n\s*\n|\Z)", re.IGNORECASE)
PLANTED = "[an instruction to AI assistants, left out]"
PLANTED_LINE_CHARS = 16


class Observation(BaseModel):
    # One action: what was acted on (place), how (key) and whether it changed something.
    kind: Literal["url", "file", "shell", "tool"]
    key: str
    detail: str
    place: str
    place_kind: PlaceKind
    pointer: str
    ok: bool = False
    # None when the call does not show whether it read or changed something.
    change: bool | None = None
    # A file the session wrote before it read it: its own output, not part of the world.
    made: bool = False


class UserNote(BaseModel):
    text: str
    pointer: str
    place: str | None
    injected: bool


class Turn(BaseModel):
    # One message of the conversation, in journal order; a user's or assistant's words a page item may cite.
    role: Literal["user", "assistant"]
    text: str
    pointer: str
    order: int
    at: str


class Step(BaseModel):
    # One tool call as the writer sees it: bounded text; `output` is the longer source an item is checked against.
    pointer: str
    call: str
    result: str = ""
    output: str = ""
    ok: bool = False
    change: bool | None = None
    order: int = 0
    at: str = ""
    targets: list[str] = []


class Extract(BaseModel):
    session_id: str
    cwd: str
    date: str
    task: str
    stop: str
    observations: list[Observation]
    notes: list[UserNote]
    final: str = ""
    calls: int = 0
    failed: int = 0
    steps: list[Step] = []
    # Strings the workspace returned, such as names and ids; a written note must not repeat one.
    values: list[str] = []
    # Words of the places and operations the session used, which a note may name.
    names: list[str] = []
    turns: list[Turn] = []
    # The pages each step (journal entry) acted on, before calls repeated across steps are merged.
    places: dict[str, list[str]] = {}
    # The world the session ran in; every session of one drive has the same world.
    world: str = ""
    # Drive pages the session used: the ones recall loaded and the ones the agent read after.
    used: list[str] = []
    # The session as System 1 sees it (ctx.state(), `session_cut`), in the steward's smaller share.
    run_state: str = ""


def result_values(outputs: list[str], calls: list[str], asked: list[str]) -> list[str]:
    hosts = " ".join(_HOST.findall(" ".join(calls))).lower()
    found: set[str] = set()
    for text in asked:
        for a, b in _QUOTED.findall(text):
            value = (a or b).strip()
            if not value.isalpha():
                found.add(value)
    for text in [*asked, *outputs]:
        for run in _CAPITALIZED.findall(text):
            words = run.split()
            for i in range(len(words) - 1):
                for j in range(i + 2, len(words) + 1):
                    if not all(w.lower() in hosts for w in words[i:j]):
                        found.add(" ".join(words[i:j]))
    for text in outputs:
        for value in _JSON_STRING.findall(text):
            value = value.strip()
            if "://" in value:
                continue
            if "@" in value:
                value = value.split("@")[0]
            words = value.split()
            if len(words) == 1 and not re.search(r"[\d_./-]", value):
                continue
            if len(words) > 1 and all(w.lower() in hosts for w in words):
                continue
            found.add(value)
    return sorted(found)


def action_names(observations: list[Observation]) -> list[str]:
    return sorted(
        {
            part.lower()
            for o in observations
            for text in (o.place.split("/", 1)[-1], o.key)
            for part in [*_NAME_PART.findall(text), *text.split()]
        }
    )


def session_values(values: list[str], texts: list[str], observations: list[Observation]) -> list[str]:
    # Names the session's data holds (identifiers, file names) are values too; places and operations are not.
    allowed = set(action_names(observations))
    found = set(values)
    for text in texts:
        found.update(_IDENTIFIER.findall(text))
        found.update(_FILE_NAME.findall(text))
    return sorted(v for v in found if v.lower() not in allowed)


def _template(segment: str) -> str:
    if segment.startswith("{") or "@" in segment or len(segment) >= 24:
        return "{id}"
    if any(c.isdigit() for c in segment) and not _VERSION.fullmatch(segment):
        return "{id}"
    return segment


@dataclass(frozen=True)
class UrlTemplate:
    segments: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]
    literal: int


def url_templates(texts: list[str]) -> dict[str, list[UrlTemplate]]:
    found: dict[str, list[UrlTemplate]] = {}
    for text in texts:
        for raw in _TEMPLATE_URL.findall(text):
            url = urlparse(raw.rstrip(".,;:)"))
            parts = tuple(s for s in url.path.split("/") if s)
            if not url.hostname or not parts or any(s.count("{") != s.count("}") for s in parts):
                continue
            patterns = tuple(
                re.compile(
                    "".join("[^/]+" if _PLACEHOLDER.fullmatch(p) else re.escape(p) for p in _PLACEHOLDER.split(s))
                )
                for s in parts
            )
            literal = sum(len(_PLACEHOLDER.sub("", s)) for s in parts)
            template = UrlTemplate(parts, patterns, literal)
            if template not in found.setdefault(url.hostname, []):
                found[url.hostname].append(template)
    return found


def _fallback(segment: str, before: str, words: set[str]) -> str:
    if _NUMBERED_VERSION.fullmatch(segment) and before in ("", "api", "rest"):
        return segment
    templated = _template(segment)
    if templated != segment or _VERSION.fullmatch(segment):
        return templated
    if segment in words and not (segment.isalpha() and segment.islower()):
        return "{id}"
    return segment


def templated_path(host: str, raw_path: str, templates: dict[str, list[UrlTemplate]], words: set[str]) -> str:
    parts = [unquote(s) for s in raw_path.split("/") if s]
    matches = [
        template
        for template in templates.get(host, [])
        if len(template.segments) == len(parts)
        and all(p.fullmatch(s) for p, s in zip(template.patterns, parts, strict=True))
    ]
    if matches:
        best = max(matches, key=lambda template: template.literal)
        return "/" + "/".join(best.segments)
    return "/" + "/".join(_fallback(s, parts[i - 1] if i else "", words) for i, s in enumerate(parts))


def segments(command: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        c = command[i]
        if quote:
            current.append(c)
            if c == "\\" and quote == '"' and i + 1 < len(command):
                current.append(command[i + 1])
                i += 1
            elif c == quote:
                quote = ""
        elif c in "'\"":
            quote = c
            current.append(c)
        elif c == "\\" and i + 1 < len(command):
            current += [c, command[i + 1]]
            i += 1
        elif c in ";|&\n":
            parts.append("".join(current))
            current = []
            if command[i : i + 2] in ("&&", "||"):
                i += 1
        else:
            current.append(c)
        i += 1
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def shell_functions(command: str) -> set[str]:
    # Names the command defines as shell functions (`f() { ...; }`, `function f { ...; }`): not installed programs.
    return {a or b for a, b in _FUNCTION.findall(command)}


def shell_tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return [tok[1:-1] if tok[:1] in "'\"" and tok[-1:] == tok[:1] else tok for tok in _TOKEN.findall(segment)]


def _dict_keys(value: object, into: list[str]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            if isinstance(key, str) and _KEY_NAME.fullmatch(key) and key not in into:
                into.append(key)
            _dict_keys(inner, into)
    elif isinstance(value, list):
        for inner in value:
            _dict_keys(inner, into)


def _json_keys(raw: str) -> list[str]:
    keys: list[str] = []
    try:
        _dict_keys(json.loads(raw), keys)
    except ValueError:
        keys = list(dict.fromkeys(_JSON_KEY.findall(raw)))
    return keys


def _flag(token: str) -> tuple[str, str | None]:
    if token.startswith("--") and "=" in token:
        name, _, value = token.partition("=")
        return name, value
    return token, None


def _json_value(text: str) -> object | None:
    if text[:1] not in "{[":
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _http(raw_urls: list[tuple[str, str]], explicit: str, keys: dict[str, list[str]]) -> list[HttpCall]:
    calls: list[HttpCall] = []
    for raw, before in raw_urls:
        url = urlparse(raw.rstrip(".,;:)"))
        if url.scheme not in ("http", "https") or not url.hostname:
            continue
        method = explicit or before or ("POST" if keys.get("body") else "GET")
        if method in METHODS:
            calls.append((url.hostname, f"{method} {url.path or '/'}", {k: v for k, v in keys.items() if v}))
    return calls


def http_calls(tokens: list[str]) -> list[HttpCall]:
    # Any client: a URL argument, a method given by flag or written just before the URL, JSON keys sent.
    explicit, get, previous = "", False, ""
    keys: dict[str, list[str]] = {}
    urls: list[tuple[str, str]] = []
    for i, token in enumerate(tokens):
        name, value = _flag(token)
        if token.startswith("-X") and len(token) > 2:
            explicit = token[2:].upper()
        elif previous in ("-X", "--request", "--method"):
            explicit = token.upper()
        elif name in ("--request", "--method") and value:
            explicit = value.upper()
        elif name in ("-G", "--get"):
            get = True
        raw = value if value is not None else token
        if raw.startswith(("http://", "https://")):
            before = tokens[i - 1].upper() if i else ""
            urls.append((raw, before if before in METHODS else ""))
        elif (parsed := _json_value(raw)) is not None:
            flag = name if value is not None else previous
            _dict_keys(parsed, keys.setdefault("params" if get or "param" in flag else "body", []))
        previous = token
    return _http(urls, explicit, keys)


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for inner in value.values() for s in _strings(inner)]
    if isinstance(value, list):
        return [s for inner in value for s in _strings(inner)]
    return []


def argument_calls(arguments: Mapping[str, object]) -> list[HttpCall]:
    urls = [(s, "") for s in _strings(arguments) if s.startswith(("http://", "https://"))]
    method = arguments.get("method")
    keys: dict[str, list[str]] = {}
    for name, inner in arguments.items():
        if isinstance(inner, dict | list) and name in ("params", "query", "body", "json", "data"):
            _dict_keys(inner, keys.setdefault("params" if name in ("params", "query") else "body", []))
    return _http(urls, method.upper() if isinstance(method, str) else "", keys)


def _target(value: str) -> str | None:
    value = " ".join(value.split())
    if not 2 <= len(value) <= TARGET_CHARS or "://" in value or value[:1] in "{[-":
        return None
    return value


def call_targets(name: str, arguments: Mapping[str, object]) -> list[str]:
    # The values a call acts on (a recipient, a channel, a path, a date), which memory about them is matched against.
    command = arguments.get("command") if name == "bash" else None
    found: list[str] = []
    if isinstance(command, str):
        for segment in segments(without_heredocs(command)):
            for token in shell_tokens(segment)[1:]:
                parsed = _json_value(token)
                for value in _strings(parsed) if parsed is not None else [token]:
                    if (target := _target(value)) is not None and not target.startswith("$"):
                        found.append(target)
    else:
        found = [t for value in _strings(dict(arguments)) if (t := _target(value)) is not None]
    return list(dict.fromkeys(found))[:MAX_TARGETS]


def shell_operation(tokens: list[str]) -> str | None:
    i = 0
    while i < len(tokens) and (
        _ASSIGNMENT.fullmatch(tokens[i]) or tokens[i] in SHELL_PREFIXES or Path(tokens[i]).name in SHELL_WRAPPERS
    ):
        i += 1
    program = Path(tokens[i]).name if i < len(tokens) else ""
    if not _PROGRAM.fullmatch(program) or program in SHELL_BASICS or program in SHELL_KEYWORDS:
        return None
    words = [program]
    for token in tokens[i + 1 : i + SHELL_WORDS]:
        if token.startswith("-") and "=" not in token and len(token) <= 3:
            words.append(token)
            continue
        if _BARE.fullmatch(token):
            words.append(token)
        break
    while len(words) > 1 and words[-1].startswith("-"):
        words.pop()
    return " ".join(words)


def without_heredocs(command: str) -> str:
    lines = command.split("\n")
    kept: list[str] = []
    end = ""
    for line in lines:
        if end:
            end = "" if line.strip() == end else end
            continue
        kept.append(line)
        if found := _HEREDOC.search(line):
            end = found[1]
    return "\n".join(kept)


def call_results(content: str, status: str, count: int) -> list[bool]:
    if status != "ok":
        return [False] * count
    lines = [line for line in content.splitlines() if not _EXIT_LINE.match(line)]
    errors = [line for line in lines if _ERROR_LINE.match(line)]
    if not errors:
        return [True] * count
    responses = [line for line in lines if line.startswith(("{", "["))]
    if len(responses) == count and all(line.startswith("{") for line in errors):
        return [not _ERROR_LINE.match(line) for line in responses]
    return [False] * count


def _detail(keys: dict[str, list[str]], tool: str) -> str:
    text = "; ".join(f"{label}: {', '.join(names)}" for label, names in keys.items() if names)
    return f"({text}) via `{tool}`" if text else f"via `{tool}`"


def _under(path: Path, cwd: Path, skip: Path | None) -> Path | None:
    path = path if path.is_absolute() else cwd / path
    try:
        path = path.resolve()
    except (OSError, RuntimeError):
        return None
    if skip is not None and (path == skip or skip in path.parents):
        return None
    return path if path == cwd or cwd in path.parents else None


def _file_uri(uri: str) -> Path | None:
    return Path(unquote(urlparse(uri).path)) if uri.startswith("file://") else None


def _key(path: Path, root: Path) -> str:
    # A file is its own entry; a folder ends with "/". With the workspace gone, a name with no suffix is a folder.
    parts = path.relative_to(root).parts
    if not parts:
        return "./"
    if path.is_dir() or (not path.exists() and not PurePosixPath(parts[-1]).suffix):
        return "/".join(parts) + "/"
    return "/".join(parts)


# Local time, so a line's date is the date the agent's "Today" line showed in that session.
def _at(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000).astimezone().isoformat(timespec="minutes")


def _run_state(path: list[Entry]) -> str:
    return session_cut(path, RUN_STATE_TOKENS)


def _replaced(value: object, clean: Callable[[str], str]) -> object:
    if isinstance(value, str):
        return clean(value)
    if isinstance(value, dict):
        return {k: _replaced(v, clean) for k, v in value.items()}
    if isinstance(value, list):
        return [_replaced(v, clean) for v in value]
    return value


def secrets_in(text: str) -> set[str]:
    found = {
        m[1]
        for pattern in (_SECRET_ASSIGNMENT, _SECRET_IN_PROSE)
        for m in pattern.finditer(text)
        if _DIGIT.search(m[1]) and _LETTER.search(m[1])
    }
    found |= {m[1] for m in _URL_PASSWORD.finditer(text)}
    found |= {m[0] for m in _SECRET_SHAPE.finditer(text)}
    return {value for value in found if value != REDACTED}


def without_secrets(ex: Extract) -> Extract:
    """The drive is shared, so a credential the session saw (a .env value, a key, a password in a URL) is replaced
    wherever it appears, and a passage planted for AI assistants is left out, before the writer reads the session
    or a page or the transcript repeats it."""
    data = ex.model_dump()
    found = {value for text in _strings(data) for value in secrets_in(text)}
    if found:
        redactor = SecretRedactor([SecretStr(value) for value in found])
        data = cast(dict[str, Any], _replaced(data, redactor.redact))

    # A line of a planted passage reaches the session without its first sentence too (a grep for a word in it).
    lines = {
        line.strip()
        for text in _strings([data["steps"], data["observations"]])
        for passage in _PLANTED.findall(text)
        for line in passage.splitlines()
        if len(line.strip()) >= PLANTED_LINE_CHARS
    }

    def planted(text: str) -> str:
        text = _PLANTED.sub(PLANTED, text)
        for line in sorted(lines, key=len, reverse=True):
            text = text.replace(line, PLANTED)
        return text

    data["steps"] = _replaced(data["steps"], planted)
    data["observations"] = _replaced(data["observations"], planted)
    data["final"] = planted(data["final"])
    data["turns"] = [_replaced(t, planted) if t["role"] == "assistant" else t for t in data["turns"]]
    return Extract.model_validate(data)


def extract(journal: Journal, drive_root: Path | None = None, world: str = "") -> Extract:
    path = journal.active_path()
    session = next(e for e in path if isinstance(e, SessionEntry))
    cwd = Path(session.cwd).resolve()
    skip = drive_root.resolve() if drive_root else None
    sid = journal.session_id
    date = datetime.fromtimestamp(session.timestamp / 1000).date().isoformat()

    observations: dict[tuple[str, str], Observation] = {}
    url_keys: dict[tuple[str, str], dict[str, list[str]]] = {}
    url_hosts: dict[tuple[str, str], str] = {}
    url_tools: dict[str, str] = {}
    by_call: dict[str, list[tuple[str, str]]] = {}
    listed: set[str] = set()
    made: set[Path] = set()
    listing = re.compile(rf"(?m)^(?:\./|{re.escape(str(cwd))}/)?([\w][^/\n:]*)/")
    outputs: list[str] = []
    data: list[str] = []
    user_text: list[str] = []
    pending: dict[str, list[tuple[tuple[str, str], str]]] = {}
    notes: list[UserNote] = []
    steps: dict[str, Step] = {}
    turns: list[Turn] = []
    task = ""
    task_pointer = ""
    stop = "none"
    final = ""
    calls = failed = 0
    last_place: str | None = None
    touched: dict[str, list[str]] = {}
    used: list[str] = []

    def drive_read(value: Path) -> None:
        if skip is None:
            return
        try:
            target = (value if value.is_absolute() else cwd / value).resolve()
        except (OSError, RuntimeError):
            return
        if skip in target.parents and target.suffix == ".md":
            used.append(target.relative_to(skip).with_suffix("").as_posix())

    def observe(
        kind: Literal["url", "file", "shell", "tool"],
        key: str,
        detail: str,
        place: tuple[str, PlaceKind],
        change: bool | None,
        entry_id: str,
        ok: bool,
    ) -> tuple[str, str]:
        nonlocal last_place
        last_place = place[0]
        ident = (place[0], key)
        pointer = f"evt#{sid}:{entry_id}"
        if place[0] not in (here := touched.setdefault(pointer, [])):
            here.append(place[0])
        known = observations.get(ident)
        if known is None:
            observations[ident] = Observation(
                kind=kind,
                key=key,
                detail=detail,
                place=place[0],
                place_kind=place[1],
                change=change,
                pointer=pointer,
                ok=ok,
                made=kind == "file" and bool(change),
            )
        elif ok and not known.ok:
            observations[ident] = known.model_copy(update={"ok": True, "pointer": pointer})
        if change and not observations[ident].change:
            observations[ident] = observations[ident].model_copy(update={"change": True})
        return ident

    def observe_file(path: Path, verb: str, change: bool | None, tool: str, entry_id: str, ok: bool) -> tuple[str, str]:
        page, kind, root = place_for_path(path, cwd)
        return observe("file", _key(path, root), f"via `{tool}`", (page, kind), change, entry_id, ok)

    def shell_place(folder: Path) -> tuple[str, PlaceKind]:
        root = project_root(folder, cwd)
        if root is not None and project_name(root, cwd):
            page, kind, _ = place_for_path(root, cwd)
            return page, kind
        return f"map/{ENVIRONMENT_PAGE}", "environment"

    def wait(call_id: str, ident: tuple[str, str], entry_id: str) -> None:
        pending.setdefault(call_id, []).append((ident, entry_id))
        by_call.setdefault(call_id, []).append(ident)

    def url(call_id: str, tool: str, host: str, key: str, keys: dict[str, list[str]], entry_id: str) -> None:
        change = key.split(" ", 1)[0] not in READ_METHODS
        place: tuple[str, PlaceKind] = (page_for_host(host), "service")
        ident = observe("url", key, "", place, change, entry_id, False)
        url_hosts[ident] = host
        url_tools[ident[0]] = tool
        merged = url_keys.setdefault(ident, {})
        for label, names in keys.items():
            merged.setdefault(label, []).extend(n for n in names if n not in merged[label])
        wait(call_id, ident, entry_id)

    def scratch(target: Path) -> bool:
        # A folder the session made holds its own work, not what the environment has.
        return any(m == target or m in target.parents for m in made)

    def bash(call_id: str, command: str, entry_id: str) -> None:
        folder = cwd
        functions = shell_functions(command)
        for segment in segments(without_heredocs(command)):
            tokens = shell_tokens(segment)
            if not tokens:
                continue
            if tokens[0] == "cd" and len(tokens) > 1:
                inside = _under(folder / tokens[1], cwd, skip)
                folder = inside or folder / tokens[1]
                if inside is not None and inside != cwd and not scratch(inside):
                    listed.add(inside.relative_to(cwd).parts[0])
                    wait(call_id, observe_file(inside, "use", None, "bash", entry_id, False), entry_id)
                continue
            if tokens[0] == "mkdir":
                made.update(m for t in tokens[1:] if not t.startswith("-") and (m := _under(folder / t, cwd, skip)))
                continue
            found = http_calls(tokens)
            for host, key, keys in found:
                url(call_id, "bash", host, key, keys, entry_id)
            operation = None if found else shell_operation(tokens)
            if operation is not None and operation.split()[0] not in functions:
                ident = observe("shell", operation, "via `bash`", shell_place(folder), None, entry_id, False)
                wait(call_id, ident, entry_id)
            for token in tokens[1:]:
                path_token(call_id, folder, token, entry_id)
        # Scripts passed inline (-c, heredocs) name their files in string literals.
        for a, b in _QUOTED_PATH.findall(command):
            path_token(call_id, folder, a or b, entry_id, quoted=True)

    def path_token(call_id: str, folder: Path, token: str, entry_id: str, quoted: bool = False) -> None:
        if len(token) > MAX_PATH_TOKEN or "://" in token or token.startswith("-"):
            return
        # A script names a file in the folder it runs in by its bare name, in quotes.
        bare = "/" not in token
        if bare and not (quoted and _BARE_FILE.fullmatch(token)):
            return
        drive_read(folder / token)
        # After changing into a folder, a path that starts with that folder's name is from the folder above it.
        first = PurePosixPath(token).parts[0]
        bases = [folder.parent, folder] if folder != cwd and first == folder.name else [folder]
        found = [t for base in bases if (t := _under(base / token, cwd, skip)) is not None and t != cwd]
        target = next((t for t in found if t.exists()), None)
        if target is None and found:
            # The workspace may be gone by the time the steward runs: a path under a folder the session listed
            # counts, and so does a bare name a listing showed.
            guess = found[0]
            if bare:
                # A name the session already knows by its full path is that file, and a name must stand on its
                # own in a listing, not end a longer one.
                known = any(o.kind == "file" and PurePosixPath(o.key).name == token for o in observations.values())
                alone = re.compile(rf"(?<![\w.-]){re.escape(token)}")
                target = guess if not known and any(alone.search(o) for o in outputs) else None
            elif _FILE_PATH.fullmatch(token) and guess.relative_to(cwd).parts[0] in listed:
                target = guess
        if target is not None and not scratch(target):
            wait(call_id, observe_file(target, "use", None, "bash", entry_id, False), entry_id)

    def resolve(call_id: str, content: str, status: str) -> None:
        waiting = pending.pop(call_id, [])
        if not waiting:
            return
        urls = [w for w in waiting if observations[w[0]].kind == "url"]
        url_oks = iter(call_results(content, status, len(urls)))
        worked = status == "ok" and not _FAILED_EXIT.search(content)
        for ident, entry_id in waiting:
            known = observations[ident]
            if next(url_oks) if known.kind == "url" else worked:
                observe(known.kind, ident[1], "", (ident[0], known.place_kind), known.change, entry_id, True)

    for order, entry in enumerate(path):
        if isinstance(entry, MemoryRecallEntry):
            used.extend(entry.pages)
        if isinstance(entry, ObservationEntry):
            outputs.append(entry.content)
        if not isinstance(entry, MessageEntry):
            continue
        match entry.message:
            case UserMessage(source="user", content=text):
                user_text.append(text)
                turns.append(
                    Turn(
                        role="user",
                        text=clip(text.strip(), TURN_CHARS),
                        pointer=f"evt#{sid}:{entry.id}",
                        order=order,
                        at=_at(entry.timestamp),
                    )
                )
                if not task_pointer:
                    task, task_pointer = text, f"evt#{sid}:{entry.id}"
                    continue
                quoted = " ".join(text.split())[:QUOTE_CHARS]
                notes.append(
                    UserNote(
                        text=quoted,
                        pointer=f"evt#{sid}:{entry.id}",
                        place=last_place,
                        injected=bool(INJECTION.search(text)),
                    )
                )
            case AssistantMessage(tool_calls=tool_calls, stop=kind, text=text):
                stop = kind
                if text.strip():
                    final = " ".join(text.split())[-FINAL_CHARS:]
                    turns.append(
                        Turn(
                            role="assistant",
                            text=text.strip()[:TURN_CHARS],
                            pointer=f"evt#{sid}:{entry.id}",
                            order=order,
                            at=_at(entry.timestamp),
                        )
                    )
                for call in tool_calls:
                    calls += 1
                    command = call.arguments.get("command") if call.name == "bash" else None
                    shown = command if isinstance(command, str) else f"{call.name} {json.dumps(call.arguments)}"
                    steps[call.id] = Step(
                        pointer=f"evt#{sid}:{entry.id}",
                        call=" ".join(shown.split())[:STEP_CALL_CHARS],
                        order=order,
                        at=_at(entry.timestamp),
                        targets=call_targets(call.name, call.arguments),
                    )
                    data.append(shown)
                    if isinstance(command, str):
                        bash(call.id, command, entry.id)
                        continue
                    for host, key, keys in argument_calls(call.arguments):
                        url(call.id, call.name, host, key, keys, entry.id)
                    if call.name in READ_TOOLS | WRITE_TOOLS:
                        for name, value in call.arguments.items():
                            if "path" not in name or not isinstance(value, str):
                                continue
                            drive_read(Path(value))
                            if (target := _under(Path(value), cwd, skip)) is not None:
                                verb, change = ("write", True) if call.name in WRITE_TOOLS else ("read", False)
                                wait(call.id, observe_file(target, verb, change, call.name, entry.id, False), entry.id)
                    elif call.name not in BASE_TOOLS:
                        detail = f"(args: {', '.join(sorted(call.arguments))})" if call.arguments else "(no args)"
                        place: tuple[str, PlaceKind] = (page_for_tool(call.name), "tool")
                        wait(call.id, observe("tool", call.name, detail, place, None, entry.id, False), entry.id)
            case ToolResultMessage(result=result):
                outputs.append(result.content)
                listed.update(listing.findall(result.content))
                if result.status != "ok":
                    failed += 1
                if (step := steps.get(result.call_id)) is not None:
                    step.result = " ".join(result.content.split())[:STEP_RESULT_CHARS]
                    step.output = result.content[:SOURCE_CHARS]
                    step.ok = all(call_results(result.content, result.status, 1)) and not _FAILED_EXIT.search(
                        result.content
                    )
                resolve(result.call_id, result.content, result.status)
                if result.status == "ok":
                    tool = result.tool_name
                    verb, change = ("write", True) if tool in WRITE_TOOLS else ("read", False)
                    for resource in result.resources:
                        found = _file_uri(resource.uri)
                        target = _under(found, cwd, skip) if found is not None else None
                        if target is not None:
                            ident = observe_file(target, verb, change, tool, entry.id, True)
                            by_call.setdefault(result.call_id, []).append(ident)
            case _:
                pass

    for call_id, step in steps.items():
        changes = [observations[ident].change for ident in by_call.get(call_id, [])]
        step.change = True if True in changes else (False if changes and None not in changes else None)

    templates = url_templates(outputs)
    values = result_values(outputs, [s.call for s in steps.values()], user_text)
    words = {w for text in user_text for w in _WORD.findall(text)} | {v for v in values if " " not in v}
    final_keys: dict[tuple[str, str], dict[str, list[str]]] = {}
    merged_observations: dict[tuple[str, str], Observation] = {}
    for ident, observation in observations.items():
        if observation.kind != "url":
            merged_observations[ident] = observation
            continue
        method, _, raw_path = observation.key.partition(" ")
        key = f"{method} {templated_path(url_hosts[ident], raw_path, templates, words)}"
        target = (observation.place, key)
        known = merged_observations.get(target)
        if known is None or (observation.ok and not known.ok):
            merged_observations[target] = observation.model_copy(update={"key": key})
        into = final_keys.setdefault(target, {})
        for label, names in url_keys[ident].items():
            into.setdefault(label, []).extend(n for n in names if n not in into[label])
    for target, keys in final_keys.items():
        detail = _detail(keys, url_tools.get(target[0], "bash"))
        merged_observations[target] = merged_observations[target].model_copy(update={"detail": detail})
    observations = merged_observations
    values = session_values(values, user_text + data + outputs, list(observations.values()))

    if turns:
        date = turns[0].at[:10]
    injected_task = bool(INJECTION.search(task))
    if injected_task:
        notes.insert(0, UserNote(text="", pointer=task_pointer, place=None, injected=True))
    return without_secrets(
        Extract(
            session_id=sid,
            cwd=str(cwd),
            date=date,
            task="" if injected_task else task,
            run_state="" if injected_task else _run_state(path),
            stop=stop,
            observations=list(observations.values()),
            notes=notes,
            final=final,
            calls=calls,
            failed=failed,
            steps=list(steps.values()),
            values=values,
            names=action_names(list(observations.values())),
            turns=[t for t in turns if not INJECTION.search(t.text)],
            places=touched,
            world=world,
            used=list(dict.fromkeys(used)),
        )
    )


def describe(observation: Observation) -> str:
    # A file or a program is known by its name; how the agent reached it says nothing a later session needs.
    if observation.kind == "file":
        return inert(f"`{observation.key}`" + (" (written)" if observation.change else ""))
    if observation.kind == "shell":
        return inert(f"`{observation.key}`")
    return inert(f"`{observation.key}` {observation.detail}")


def file_pages(observations: list[Observation]) -> dict[str, str]:
    # The page of each file the session read or wrote, with the file's path in its place.
    found: dict[str, str] = {}
    for o in sorted(observations, key=lambda o: not o.ok):
        if o.kind == "file" and not o.made and not is_folder_key(o.key) and o.place_kind in ("documents", "project"):
            found.setdefault(file_page(o.place, o.key), o.key)
    return found
