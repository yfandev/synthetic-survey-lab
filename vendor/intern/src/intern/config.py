"""User configuration and session assembly, shared by the CLI and the JSON-RPC server.

Everything user-level lives under one config home, `$INTERN_HOME` (default `~/.intern`):
`config.toml`, `profiles/`, `AGENTS.md`, `SYSTEM2.md`, `SYSTEM1.md`, `sessions/` and `spill/`. Pointing the variable
at an empty folder gives a fully isolated run, which is what evals do. A memory drive is mounted from
`--memory`, `[capabilities.memory] drive` or `$INTERN_MEMORY`, in that order, else from `memory/` in
the config home, created on first use. `--no-memory` or `enabled = false` there turns it off.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from intern import capability_toggles, names
from intern.capability_settings import CapabilitySettings
from intern.decisions import (
    MAX_CONCURRENCY,
    DecisionBackend,
    DecisionClient,
    DecisionsApiBackend,
    System1Unavailable,
)
from intern.events import EventSink, MemoryMount, SecretUnavailable
from intern.executor import LocalExecutor, physical_memory
from intern.extensions import REGISTRY
from intern.extensions.code_search import CodeSearch
from intern.extensions.memory import Memory
from intern.extensions.memory.reminders import MemoryActions
from intern.extensions.workspace.extension import Workspace
from intern.hooks import Extension, Hooks, build_extension, resolve_extension
from intern.instructions import FileInstructions
from intern.journal import Journal
from intern.loop import EngineDeps, ModelBinding
from intern.memory.drive import Drive, GitError, create
from intern.memory.spend import STEWARD_DIR
from intern.memory.steward.writer import writer_from_ref
from intern.models.catalog import (
    DECISIONS_PREFIX,
    NACE_CONSOLE_URL,
    CapabilityInfo,
    Catalog,
    ModelInfo,
    System1Sources,
    list_models,
    profile_models,
)
from intern.models.profile import effort_choices, load_profile, profile_candidates
from intern.models.registry import PROVIDERS, Api, ConfigError, ProviderSpec, resolve_endpoint
from intern.models.window import advertised_limits, bound_output, working_window
from intern.providers.anthropic import AnthropicProvider
from intern.providers.openai_compat import OpenAICompatProvider
from intern.providers.openai_responses import OpenAIResponsesProvider
from intern.secret_store import StoreEntry, check_entries, read_store
from intern.session import Session
from intern.session_setup import capability_extensions, capability_problem, prepare_session, runner_choices
from intern.tools import load_packs, with_extension_tools
from intern.tools.files import ReadTool

HOME_ENV = f"{names.ENV_PREFIX}HOME"
SYSTEM2_ENV = f"{names.ENV_PREFIX}SYSTEM2"
MEMORY_ENV = f"{names.ENV_PREFIX}MEMORY"
DRIVE_DIR = "memory"
# System 1 on the console (`drex:<model>`): https://console.nace.ai serves /v1/systemone.
DECISION_URL_ENV = "DREX_BASE_URL"
DECISION_KEY_ENV = "DREX_API_KEY"
DEFAULT_HOME = Path(names.HOME)
CONFIG_FILE = "config.toml"
ENV_FILE = ".env"
"""Provider keys in the harness home, `KEY=value` a line, for those the shell does not export."""
DEFAULT_SYSTEM1_MODEL = "drex-latest"
PROVIDER_CLASSES = {"anthropic": AnthropicProvider, "responses": OpenAIResponsesProvider}
"""The adapter for each native API; every other endpoint is OpenAI-compatible."""
# The models the harness knows, with the names the terminal UI shows for them.
# Without a configured System 2 the UI starts on the first whose provider key is set; Nemotron Lightning answers
# plain chat erratically, so it comes last.
KNOWN_SYSTEM2: dict[str, str] = {
    "anthropic/claude-haiku-5-5": "Claude Haiku 5.5",
    "openrouter/qwen/qwen3.8-27b": "Qwen3.8 27B",
    "fireworks/accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b": "Nemotron 3.5 Lightning 30B-A3B",
}
SYSTEM1_HELP = (
    f"a decision model on the console ({DEFAULT_SYSTEM1_MODEL}), drex:<model>[,<fallback>...], "
    "or decisions:<model>[,<fallback>...]"
)
VISION_LINE = (
    "You can see images: read an image file (png, jpg, gif, webp) to look at it. To look at a scanned PDF page, "
    "a chart or a slide, render it to PNG with a command-line tool and read the PNG."
)
HOST_VISION_LINE = (
    "You can see images through the available image-viewing tools. Use their descriptions to choose how to view "
    "an image; read only identifies image files."
)


class System1Config(BaseModel):
    backend: Literal["console", "decisions-api"] = Field(
        default="console",
        description="Where System 1 is served: the console (the default) or a decisions API (OpenRouter: "
        "Jev, `nace-ai/drex-*`, ...). System 1 is required; there is no running without it.",
    )
    model: str | None = Field(default=None, description="The model; on the console, the first of `models` by default.")
    fallback_models: list[str] = Field(default_factory=list)
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_s: float = 10.0
    max_state_tokens: int = Field(default=32_768, gt=0)
    max_concurrency: int = Field(default=MAX_CONCURRENCY, gt=0)
    models: list[str] = Field(
        default_factory=lambda: [DEFAULT_SYSTEM1_MODEL],
        description="Decision models on the console the pickers offer when the console cannot be listed; the first "
        "is the default System 1.",
    )
    console_url: str = Field(default=NACE_CONSOLE_URL, description="Where the console serves `models`.")
    console_key_env: str = Field(default="NACE_API_KEY", description="The console's API key variable.")


class CommandsConfig(BaseModel):
    memory_share: float = Field(
        default=1 / 2,
        gt=0,
        le=1,
        description="Share of physical memory a shell command's process group may hold before it is stopped.",
    )


class GuardConfig(BaseModel):
    system_writes: Literal["refuse", "allow"] = Field(
        default="refuse",
        description="Writes to system files (/etc, /usr, ...): refused by a fixed rule, or left to System 1's "
        "judgement. `allow` is for a disposable machine, such as a container the agent runs in as root; deleting "
        "a system directory stays refused.",
    )


class UserConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    # The old keys `model`, `[decider]` and `[memory]` still load, so existing config files keep working.
    system2: str | None = Field(default=None, validation_alias=AliasChoices("system2", "model"))
    system1: System1Config = Field(default_factory=System1Config, validation_alias=AliasChoices("system1", "decider"))
    providers: dict[str, ProviderSpec] = Field(default_factory=dict)
    capabilities: CapabilitySettings = Field(default_factory=CapabilitySettings)
    commands: CommandsConfig = Field(default_factory=CommandsConfig)
    secrets: dict[str, StoreEntry] = Field(default_factory=dict)
    """Where each stored secret comes from: `"keychain"` or a command; see `secret_store`."""
    guard: GuardConfig = Field(default_factory=GuardConfig)

    @field_validator("secrets", mode="before")
    @classmethod
    def secret_entries(cls, value: Any) -> Any:
        return check_entries(value)

    @model_validator(mode="before")
    @classmethod
    def old_memory_table(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "memory" not in data:
            return data
        data = dict(data)
        old = data.pop("memory")
        capabilities = dict(data.get("capabilities") or {})
        if "memory" in capabilities:
            raise ValueError("[memory] and [capabilities.memory] both set memory; keep [capabilities.memory]")
        data["capabilities"] = {**capabilities, "memory": old}
        return data


class Options(BaseModel):
    """What a client asked for; anything left as None comes from the config home."""

    cwd: Path
    system2: str | None = None
    system1: str | None = Field(default=None, description="decisions:<model>[,<fallback>...] or drex:<model>[,...]")
    window_cap: int | None = None
    max_turns: int | None = None
    profile: Path | None = None
    context_files: bool = True
    extensions: bool = Field(
        default=True,
        description="Load the profile's extensions; extra_extensions load either way, and capabilities are unaffected.",
    )
    extra_extensions: list[str] = Field(default_factory=list)
    session_file: Path | None = None
    persist: bool = True
    capabilities: CapabilitySettings | None = None
    memory: Path | None = None
    memory_off: bool = Field(default=False, description="No memory drive, even when the config or $env names one.")


def _command_memory(share: float) -> int | None:
    """A shell command's memory budget in bytes: ``share`` of physical memory, or none where that is unknown."""
    memory = physical_memory()
    return int(memory * share) if memory else None


def home_dir(env: Mapping[str, str]) -> Path:
    return Path(env.get(HOME_ENV) or DEFAULT_HOME).expanduser()


def read_env_file(path: Path) -> dict[str, str]:
    """`KEY=value` lines; blank lines and `#` comments skipped, an `export ` prefix and matching quotes dropped.
    A missing or unreadable file is empty."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.removeprefix("export ").lstrip().partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key.isidentifier():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def process_env() -> dict[str, str]:
    """The process environment over the harness home's `.env`: a key put there once reaches every run from any
    directory, and a key exported in the shell wins over it. The home itself comes from the environment."""
    env = dict(os.environ)
    return {**read_env_file(home_dir(env) / ENV_FILE), **env}


def harness_keys(config: UserConfig) -> frozenset[str]:
    """The environment variables the harness reads keys from itself: each provider's `api_key_env`, and every
    `*_env`, `header_env` and `env_refs` the config names (search, System 1, MCP, runners). A stored secret with
    one of these names stays out of the agent's commands, so a model key is never the agent's to spend."""
    names = {spec.api_key_env for spec in (*PROVIDERS.values(), *config.providers.values()) if spec.api_key_env}
    names.add(DECISION_KEY_ENV)

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("header_env", "env_refs") and isinstance(item, dict):
                    names.update(source for source in item.values() if isinstance(source, str))
                elif isinstance(key, str) and key.endswith("_env") and key != "base_url_env" and isinstance(item, str):
                    names.add(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(config.model_dump(exclude={"secrets"}))
    return frozenset(names)


def load_user_config(home: Path) -> UserConfig:
    path = home / CONFIG_FILE
    if not path.is_file():
        return UserConfig()
    try:
        return UserConfig.model_validate(tomllib.loads(path.read_text()))
    except (tomllib.TOMLDecodeError, ValueError) as e:
        raise ConfigError(f"{path} is invalid: {e}") from e


class System2Unavailable(ConfigError):
    """No System 2 could be bound: none is configured, or the one asked for cannot run (e.g. its key is missing)."""

    def __init__(self, message: str, *, ref: str, system1: str) -> None:
        super().__init__(message)
        self.ref = ref
        """The System 2 the session would have run on; empty when none is configured."""
        self.system1 = system1
        """The System 1 the session would have started on."""


@dataclass(frozen=True)
class CapabilityExtension:
    """An extension that sits behind a capability: its capability's switch turns it on, never a profile's
    `extensions` list or `--extension`."""

    cls: type[Extension]
    """The class refused when named as an extension; its subclasses are refused too."""
    toggle: str
    """The capability's switch (`capability_toggles` id)."""
    message: str
    """What to do instead, shown when it is named as an extension."""


CAPABILITY_EXTENSIONS: dict[str, CapabilityExtension] = {
    CodeSearch.name: CapabilityExtension(
        CodeSearch,
        "cap.code-search",
        "code search is turned on by its capability, not loaded as an extension: "
        f"set [capabilities.code_search] enabled = true in {CONFIG_FILE}, or switch it with /caps",
    ),
    Workspace.name: CapabilityExtension(
        Workspace,
        "cap.workspace",
        "the workspace is turned on by its capability, not loaded as an extension: "
        f"set [capabilities.workspace] enabled = true in {CONFIG_FILE}, or switch it with /caps",
    ),
    Memory.name: CapabilityExtension(
        MemoryActions,
        "cap.memory",
        "memory is a capability, on by default, not loaded as an extension: "
        f"switch it with [capabilities.memory] enabled in {CONFIG_FILE}, --memory/--no-memory, or /caps",
    ),
}


def _extension(name: str) -> Extension:
    """A profile's or `--extension`'s extension. Extensions behind a capability (code search, the workspace,
    memory) are refused by name or by class before they are built: their capability turns them on."""
    if name in CAPABILITY_EXTENSIONS:
        raise ConfigError(CAPABILITY_EXTENSIONS[name].message)
    cls = resolve_extension(name, REGISTRY)
    for entry in CAPABILITY_EXTENSIONS.values():
        if issubclass(cls, entry.cls):
            raise ConfigError(entry.message)
    return build_extension(name, cls)


class Bootstrap:
    """Turns options plus the config home into model bindings and a ready session."""

    def __init__(self, options: Options, env: Mapping[str, str] | None = None) -> None:
        self.options = options
        self.env = process_env() if env is None else dict(env)
        self.home = home_dir(self.env)
        self.config = load_user_config(self.home)
        self.store, self.store_failures = read_store(
            self.config.secrets, self.env, cwd=self.home if self.home.is_dir() else None
        )
        """The `[secrets]` values, read once per process; they also sit in `env`, over an exported variable of the
        same name. An entry that could not be read is in `store_failures`, and each session reports it."""
        self.env.update({name: value.get_secret_value() for name, value in self.store.items()})

    def default_system2(self) -> str:
        """The System 2 a session opens on unless a client picks one; empty when none is configured."""
        return self.options.system2 or self.config.system2 or self.env.get(SYSTEM2_ENV) or ""

    async def bind(self, ref: str) -> ModelBinding:
        endpoint = resolve_endpoint(ref, self.env, self.config.providers)
        profile = load_profile(endpoint.ref, search_dirs=[self.home / "profiles"], override=self.options.profile)
        if self.options.max_turns:
            profile.max_turns = self.options.max_turns
        limits = await advertised_limits(endpoint, profile)
        bound_output(profile, limits.max_output)
        profile.vision = profile.vision or limits.vision is True  # the provider says it takes images
        window = working_window(limits.window, limits.source, profile, self.options.window_cap)
        provider = PROVIDER_CLASSES.get(endpoint.api, OpenAICompatProvider)(endpoint)
        return ModelBinding(
            provider=provider,
            ref=endpoint.ref,
            wire_model=endpoint.model,
            profile=profile,
            window=window,
        )

    def capability_settings(self) -> CapabilitySettings:
        """What a session opens with: the options' whole object when given, else the saved `[capabilities]`;
        `--memory` and `--no-memory` then set its memory."""
        caps = self.options.capabilities if self.options.capabilities is not None else self.config.capabilities
        if self.options.memory_off:
            memory = caps.memory.model_copy(update={"enabled": False})
        elif self.options.memory is not None:
            memory = caps.memory.model_copy(update={"enabled": True, "drive": self.options.memory})
        else:
            return caps
        return caps.model_copy(update={"memory": memory})

    def memory_path(self) -> Path | None:
        """The memory drive a session mounts: `[capabilities.memory] drive`, `$INTERN_MEMORY`, else
        `memory/` in the config home; none when memory is off."""
        memory = self.capability_settings().memory
        if not memory.enabled:
            return None
        path = memory.drive or self.env.get(MEMORY_ENV)
        return Path(path).expanduser() if path else self.home / DRIVE_DIR

    def memory(self) -> Drive | None:
        path = self.memory_path()
        if path is None:
            return None
        # Only the default drive is created; a named one that is missing is a mistake to report.
        if path == self.home / DRIVE_DIR and not path.exists():
            try:
                create(path)
            except GitError as e:
                raise ConfigError(f"memory drive {path} cannot be created: {e}") from e
        return Drive.mount(path)

    def _steward_on_close(self, journal: Path, drive: Path, system2: str) -> Callable[[Session], None]:
        argv = [sys.executable, "-m", "intern.cli", "steward", str(journal), "--drive", str(drive), "--merge"]
        if self.options.system1:
            argv += ["--system1", self.options.system1]
        writer = self.capability_settings().memory.writer or system2
        try:
            writer_from_ref(writer, self.env)
        except ConfigError:
            pass
        else:
            argv += ["--writer", writer]
        log = self.home / STEWARD_DIR / f"{journal.stem}.log"

        def start(session: Session) -> None:
            if session.last_outcome is None or not journal.is_file():
                return
            # A steward that cannot start costs this session's memory, never its close.
            with contextlib.suppress(OSError):
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("ab") as out:
                    # Its own process group, so the steward outlives the client and a Ctrl-C to it.
                    subprocess.Popen(
                        argv,
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=out,
                        env={**os.environ, **self.env},
                        start_new_session=True,
                    )

        return start

    def _system1_config(self, spec: str | None) -> System1Config:
        """`[system1]` as `spec` amends it: `decisions:<model>[,<fallback>...]` on the decisions API,
        `drex:<model>[,<fallback>...]` or a bare decision model on the console."""
        cfg = self.config.system1
        if not spec:
            return cfg
        if spec == "heuristic":
            raise ConfigError(f"System 1 is required: --system1 heuristic is gone; pass {SYSTEM1_HELP}")
        kind, _, rest = spec.partition(":")
        models = [m for m in rest.split(",") if m]
        if kind == "decisions" and models:
            return cfg.model_copy(update={"backend": "decisions-api", "model": None, "fallback_models": models})
        if kind == "drex" and models:
            return cfg.model_copy(update={"backend": "console", "model": models[0], "fallback_models": models[1:]})
        if ":" not in spec and spec.strip():
            return cfg.model_copy(update={"backend": "console", "model": spec, "fallback_models": []})
        raise ConfigError(f"system1 {spec!r} must be {SYSTEM1_HELP}")

    def _console(self, cfg: System1Config) -> tuple[str, str | None]:
        """The console's URL and key: `$DREX_BASE_URL` and `$DREX_API_KEY` when set, else `[system1].console_url`
        and the key in `[system1].console_key_env`."""
        base = self.env.get(DECISION_URL_ENV)
        url = f"{base.rstrip('/')}/v1" if base else cfg.console_url
        return url, self.env.get(cfg.console_key_env) or self.env.get(DECISION_KEY_ENV)

    def _system1(self, spec: str | None = None) -> DecisionBackend:
        """The backend `spec` names, or without one the backend `--system1` or `[system1]` configured. System 1 is
        required: one that cannot be set up (no key, no model) is a ConfigError."""
        cfg = self._system1_config(self.options.system1 if spec is None else spec)
        if cfg.backend == "console":
            first = cfg.model or (cfg.models[0] if cfg.models else None)
            models = [m for m in [first, *cfg.fallback_models] if m]
            if not models:
                raise ConfigError("system1 backend console needs system1.model or system1.models")
            url, key = self._console(cfg)
            if not key:
                raise ConfigError(
                    f"System 1 ({models[0]} from the console) needs a key in ${cfg.console_key_env} (or "
                    f"${DECISION_KEY_ENV}); or set [system1] in {self.home / CONFIG_FILE}, or pass --system1 "
                    "decisions:<model>"
                )
            return DecisionsApiBackend(url, SecretStr(key), models, timeout_s=cfg.timeout_s)
        models = [m for m in [cfg.model, *cfg.fallback_models] if m]
        if not models:
            raise ConfigError("system1 backend decisions-api needs system1.model")
        key = self.env.get(cfg.api_key_env)
        if not key:
            raise ConfigError(f"the decisions API needs a key in ${cfg.api_key_env}")
        return DecisionsApiBackend(cfg.base_url, SecretStr(key), models, timeout_s=cfg.timeout_s)

    def system1_choice(self) -> str:
        """The System 1 a session starts on, as the pickers name it: a decision model on the console,
        `decisions:<model>` on the decisions API; empty when none is configured, which `_system1` refuses."""
        cfg = self._system1_config(self.options.system1)
        if cfg.backend == "console":
            return cfg.model or (cfg.models[0] if cfg.models else "")
        models = [m for m in [cfg.model, *cfg.fallback_models] if m]
        return f"{DECISIONS_PREFIX}{models[0]}" if models else ""

    def system1_backend(self, choice: str) -> DecisionBackend:
        """A picked System 1: the one the session started on (its own endpoint, fallbacks and key),
        `decisions:<model>` on the decisions API, or a decision model on the console. One that cannot be set up
        is a ConfigError, and the session keeps the System 1 it had."""
        return self._system1() if choice == self.system1_choice() else self._system1(choice)

    def system1_sources(self) -> System1Sources:
        cfg = self.config.system1
        url, key = self._console(cfg)
        return System1Sources(
            console_url=url,
            console_key=key,
            console_models=tuple(cfg.models),
            decisions_url=cfg.base_url,
            decisions_key=self.env.get(cfg.api_key_env),
        )

    async def known_system2(self) -> str | None:
        """A System 2 to start on when none is configured: the first of KNOWN_SYSTEM2 whose provider key is set,
        else the first model the reachable providers list; None when no provider can be reached."""
        for ref in KNOWN_SYSTEM2:
            try:
                resolve_endpoint(ref, self.env, self.config.providers)
            except ConfigError:
                continue
            return ref
        listed = (await self.catalog()).system2
        return listed[0].id if listed else None

    def _api(self, ref: str) -> Api:
        """The API a `provider/model` is reached through, without needing its key."""
        spec = {**PROVIDERS, **self.config.providers}.get(ref.partition("/")[0])
        return spec.api if spec is not None else "openai"

    async def catalog(self) -> Catalog:
        """The models clients can pick, with the default marked, the reasoning efforts each takes (as a session on
        it offers them), and each one's window: the profile's or the listing's, no larger than the profile's
        effective window."""
        default = self.default_system2()
        catalog = await list_models(
            self.env,
            providers=self.config.providers,
            extra_system2=[*profile_models([self.home / "profiles"]), ModelInfo(id=default, name=default)],
            system1=self.system1_sources(),
        )
        candidates = profile_candidates([self.home / "profiles"])

        def marked(model: ModelInfo) -> ModelInfo:
            profile = load_profile(model.id, override=self.options.profile, candidates=candidates)
            window = profile.window or model.window
            if window and profile.effective_window:
                window = min(window, profile.effective_window)
            provider = PROVIDER_CLASSES.get(self._api(model.id), OpenAICompatProvider)
            efforts = effort_choices(getattr(provider, "effort_levels", ()), profile)
            return model.model_copy(
                update={
                    "default": model.id == default,
                    "reasoning": bool(efforts),
                    "efforts": list(efforts),
                    "window": window,
                }
            )

        caps = self.capability_settings()
        packs = load_profile(default, override=self.options.profile, candidates=candidates).tool_packs
        capabilities = [
            CapabilityInfo(
                id=toggle.id,
                name=toggle.name,
                description=toggle.note,
                enabled=toggle.on,
                unavailable=capability_problem(toggle.id, caps, self.env, installs=True),
            )
            for toggle in capability_toggles.toggles(caps, caps, packs)
        ]
        return catalog.model_copy(
            update={
                "system2": [marked(model) for model in catalog.system2],
                "capabilities": capabilities,
                "runners": runner_choices(caps.runner),
            }
        )

    async def session(self, emit: EventSink) -> Session:
        options = self.options
        capabilities = self.capability_settings()
        ref = self.default_system2()
        if not ref:
            raise System2Unavailable(
                f"no model given: pass --system2 provider/model, set ${SYSTEM2_ENV}, "
                f"or set system2 in {self.home / CONFIG_FILE}",
                ref="",
                system1=self.system1_choice(),
            )
        try:
            system2 = await self.bind(ref)
        except ConfigError as e:
            raise System2Unavailable(str(e), ref=ref, system1=self.system1_choice()) from None
        try:
            packs = capabilities.effective_tool_packs(system2.profile.tool_packs)
        except ValueError as error:
            raise ConfigError(str(error)) from None
        system2.profile = system2.profile.model_copy(update={"tool_packs": packs})
        memory = self.memory()

        names = [*(system2.profile.extensions if options.extensions else []), *options.extra_extensions]
        fixed_extensions = [_extension(name) for name in names]
        extensions = [*fixed_extensions, *capability_extensions(capabilities)]
        if memory is not None:
            # Last, so its drive write guard sees the arguments other extensions replaced.
            extensions.append(Memory(memory))
        hooks = Hooks(extensions, emit)
        system1 = self._system1()
        cfg = self.config.system1
        decisions = DecisionClient(
            system1,
            emit,
            timeout_s=cfg.timeout_s,
            max_state_tokens=cfg.max_state_tokens,
            max_concurrency=cfg.max_concurrency,
        )
        try:
            await decisions.check()
        except System1Unavailable as e:
            raise ConfigError(f"{e}; the harness does not run without System 1") from None

        journal_path = None
        if options.persist:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            journal_path = options.session_file or self.home / "sessions" / f"{stamp}-{uuid4().hex}.jsonl"
        journal = Journal(cwd=str(options.cwd), path=journal_path)
        instructions = FileInstructions(self.home, context_files=options.context_files).load(options.cwd)
        pack_tools = load_packs(system2.profile.tool_packs)
        fixed_tools = with_extension_tools(pack_tools, fixed_extensions)
        configured_read = next((tool for tool in fixed_tools if tool.name == "read"), None)
        tools = with_extension_tools(fixed_tools, extensions[len(fixed_extensions) :])
        read = next((tool for tool in tools if tool.name == "read"), None)
        if isinstance(configured_read, ReadTool) and not configured_read.attach_images and read is not configured_read:
            raise ConfigError(
                "A capability replaces the extension's read after image attachments were disabled. "
                "Set [capabilities.workspace] enabled = false to keep the configured reader."
            )
        if system2.profile.vision:
            vision_line = HOST_VISION_LINE if isinstance(read, ReadTool) and not read.attach_images else VISION_LINE
            instructions = instructions.model_copy(update={"system": f"{instructions.system}\n{vision_line}"})
        deps = EngineDeps(
            system2=system2,
            tools=tools,
            hooks=hooks,
            journal=journal,
            instructions=instructions,
            decisions=decisions,
            executor=LocalExecutor(memory_limit=_command_memory(self.config.commands.memory_share)),
            cwd=options.cwd,
            spill_dir=self.home / "spill" / journal.session_id,
            emit=emit,
            system_writes=self.config.guard.system_writes == "allow",
        )
        if memory is not None:
            emit(MemoryMount(drive=str(memory.root), commit=memory.commit))
        for failure in self.store_failures:
            emit(SecretUnavailable(name=failure.name, reason=failure.reason, message=str(failure)))
        prepared = await prepare_session(
            deps,
            capabilities,
            pack_tools=pack_tools,
            env=self.env,
            home=self.home,
            resolve_model=self.bind,
            secret_store=self.store,
            harness_only=harness_keys(self.config),
        )
        on_close = None
        if memory is not None and journal_path is not None and capabilities.memory.steward:
            on_close = self._steward_on_close(journal_path, memory.root, ref)
        try:
            return Session(
                prepared.deps,
                resolve_model=prepared.resolve_model,
                resources=prepared.resources,
                model_budget=prepared.budget,
                resolve_system1=self.system1_backend,
                system1=self.system1_choice(),
                on_close=on_close,
                idle_s=capabilities.memory.steward_idle_s,
            )
        except BaseException:
            if prepared.resources is not None:
                await prepared.resources.aclose()
            raise
