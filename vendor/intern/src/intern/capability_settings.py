"""Saved settings for existing optional capabilities, without starting services.

Tool selection stays with existing packs unless explicitly overridden. Credential
values are resolved by session assembly, never stored in these models.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from intern.browser import BrowserOptions
from intern.runners import NoOptions, load_runner
from intern.tools import PACKS
from intern.tools.base import Effect
from intern.web.reader import ReadSettings
from intern.web.search import SearchBackend

EnvName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=128)]
SecretName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]+$", max_length=128)]
ScopeName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:/-]+$", max_length=128)]
HeaderName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+$", max_length=128)]
ModelRef = Annotated[str, StringConstraints(pattern=r"^[^/\s]+/\S+$", max_length=512)]
DeclaredEffects = frozenset[Effect]
DEFAULT_ALLOWED_ENV = frozenset({"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "SYSTEMROOT"})
"""Environment names a credential-bearing child process inherits by default."""


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


def _http_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or any(ord(char) <= 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError
        _ = parsed.port
    except ValueError:
        raise ValueError("Use an HTTP(S) URL without embedded credentials, whitespace or fragment") from None
    return value


class SearchConfig(_Settings):
    backend: SearchBackend | None = None
    api_key_env: EnvName | None = None
    searxng_url: str | None = None
    timeout_s: float = Field(default=20, gt=0, le=120, allow_inf_nan=False)
    max_attempts: int = Field(default=2, ge=1, le=3)

    @field_validator("searxng_url")
    @classmethod
    def valid_url(cls, value: str | None) -> str | None:
        return _http_url(value) if value is not None else None

    @model_validator(mode="after")
    def backend_fields(self) -> SearchConfig:
        if self.searxng_url is not None and self.backend not in {None, "searxng"}:
            raise ValueError("searxng_url requires the searxng search backend")
        if self.backend == "searxng" and self.api_key_env is not None:
            raise ValueError("The existing searxng search adapter does not use an API key")
        return self


class ReaderConfig(ReadSettings):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    storage_dir: Path | None = None


class ProcessConfig(_Settings):
    max_running: int = Field(default=4, ge=1, le=64)
    max_records: int = Field(default=64, ge=1, le=10_000)
    max_log_bytes: int = Field(default=65_536, ge=1, le=16_777_216)
    move_after_s: float = Field(
        default=30,
        ge=0,
        le=86_400,
        description="A `bash` command still running after this many seconds goes on as a background job; 0 never "
        "moves one.",
    )

    @model_validator(mode="after")
    def capacity(self) -> ProcessConfig:
        if self.max_records < self.max_running:
            raise ValueError("Process record capacity must cover all running processes")
        return self


class BrowserConfig(BrowserOptions):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    enabled: bool = False
    artifact_dir: Path | None = None


class CodeSearchConfig(_Settings):
    """`decision_code` over the working directory (the code-search extension); the only way to turn it on."""

    enabled: bool = False


class WorkspaceConfig(_Settings):
    """Documents the agent indexes (`index_files`, `decision_files`, `sql`, a document-aware `read`: the workspace
    extension); the only way to turn it on."""

    enabled: bool = False


class MemoryConfig(_Settings):
    """Recall from a memory drive, a git checkout of main, and steward each session into it when it closes."""

    enabled: bool = True
    drive: Path | None = None
    """Unset is `$INTERN_MEMORY`, else `memory/` in the config home, created on first use."""
    steward: bool = True
    steward_idle_s: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 300
    """Seconds after a run ends with no new one before the session so far is stewarded; 0 waits for the close."""
    writer: ModelRef | None = None
    """The provider/model the steward phrases notes with; unset is the session's System 2."""


RunnerName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]+$", max_length=256)]


class UnresolvedOptions(BaseModel):
    """Options for a runner this install cannot load. They are kept as given; the session fails at setup with why."""

    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)


class RunnerConfig(_Settings):
    """`[capabilities.runner]`: where commands run. A runner by name (`local`, `docker`, `opensandbox`, one a package
    registers, or `module:Class`) and that runner's own options, validated by it when the config loads.

    A runner this install cannot load (its package missing) does not stop the config loading, so commands such as
    `models --json` keep working; a session that would use it fails at setup and says why."""

    name: RunnerName = "local"
    workspace: Path | None = None
    """Isolated runners only: the folder that holds the session working directory, relative to it."""
    options: SerializeAsAny[BaseModel] = Field(default_factory=NoOptions)

    @model_validator(mode="before")
    @classmethod
    def runner_options(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        name = data.get("name", "local")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", name):
            return data  # the field reports it
        options = data.get("options", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        raw = options.model_dump() if isinstance(options, BaseModel) else options
        try:
            kind = load_runner(name)
        except ValueError:
            return {
                **data,
                "options": options if isinstance(options, BaseModel) else UnresolvedOptions.model_validate(raw),
            }
        if data.get("workspace") is not None and not kind.isolated:  # pyright: ignore[reportUnknownMemberType]
            raise ValueError(f"runner.workspace applies to an isolated runner, not {name!r}")
        if not isinstance(options, kind.Options):
            try:
                options = kind.Options.model_validate(raw)
            except ValidationError as error:
                problems = "; ".join(
                    f"{'.'.join(map(str, problem['loc'])) or 'options'}: {problem['msg']}"
                    for problem in error.errors(include_input=False, include_url=False)
                )
                raise ValueError(f"Invalid options for runner {name!r}: {problems}") from None
        return {**data, "options": options}


class SecretBindingConfig(_Settings):
    """Where one secret's value comes from: an environment variable (`source_env`) or a host command that prints it
    (`command`, an absolute path, with literal `args`), and the scopes allowed to use it."""

    source_env: EnvName | None = None
    command: str | None = Field(default=None, min_length=1, max_length=4096)
    args: list[str] = Field(default_factory=list, max_length=256)
    env_refs: dict[EnvName, EnvName] = Field(default_factory=dict)
    """`command` only: `{variable the command sees: host variable it is read from}`, e.g. a 1Password service token."""
    timeout_s: float = Field(default=30, gt=0, le=300)
    scopes: Annotated[frozenset[ScopeName], Field(min_length=1)]

    @model_validator(mode="after")
    def one_source(self) -> SecretBindingConfig:
        if (self.source_env is None) == (self.command is None):
            raise ValueError("A secret binding takes exactly one of source_env or command")
        if self.command is None and (self.args or self.env_refs):
            raise ValueError("args and env_refs apply to a command binding")
        if self.command is not None and (
            any(ord(c) < 32 or ord(c) == 127 for c in self.command) or not Path(self.command).is_absolute()
        ):
            raise ValueError("A secret command must be an absolute path without control characters")
        return self


class SecretsConfig(_Settings):
    scope: ScopeName = "executor"
    bindings: dict[SecretName, SecretBindingConfig] = Field(default_factory=dict)
    inject: dict[EnvName, SecretName] = Field(default_factory=dict)
    allowed_env: frozenset[EnvName] = Field(default_factory=lambda: DEFAULT_ALLOWED_ENV)

    @model_validator(mode="after")
    def authorized_bindings(self) -> SecretsConfig:
        if self.allowed_env.intersection(self.inject):
            raise ValueError("Injected credential names cannot also be caller-overridable environment names")
        if any(
            name not in self.bindings or self.scope not in self.bindings[name].scopes for name in self.inject.values()
        ):
            raise ValueError("Injected secret references must be declared and authorized for the selected scope")
        return self


_BLOCKED_HEADERS = frozenset(
    {"host", "connection", "content-length", "transfer-encoding", "proxy-authorization", "proxy-connection", "upgrade"}
)
# Headers the MCP Streamable HTTP transport sets itself; a credential must not replace them.
_MCP_PROTOCOL_HEADERS = frozenset({"mcp-session-id", "mcp-protocol-version", "content-type", "accept"})


class CredentialCommandConfig(_Settings):
    """A host command that prints one header value for an HTTP MCP entry, run again after a 401."""

    command: str = Field(min_length=1, max_length=4096)
    args: list[str] = Field(default_factory=list, max_length=256)
    env_refs: dict[EnvName, EnvName] = Field(default_factory=dict)
    """`{variable the command sees: host variable it is read from}`."""
    header: HeaderName = "Authorization"

    @field_validator("command")
    @classmethod
    def absolute_command(cls, value: str) -> str:
        if any(ord(c) < 32 or ord(c) == 127 for c in value) or not Path(value).is_absolute():
            raise ValueError("A credential command must be an absolute path without control characters")
        return value


class McpOAuthConfig(_Settings):
    """MCP OAuth sign-in for an HTTP entry: `intern mcp login <namespace>` signs in, sessions reuse the tokens.

    Without `client_id` the client registers itself with the server (dynamic client registration).
    """

    client_id: str | None = Field(default=None, min_length=1, max_length=512)
    client_secret_env: EnvName | None = None
    scope: str | None = Field(default=None, min_length=1, max_length=1024)
    callback_port: int = Field(default=33418, ge=1024, le=65535)
    """The local port the browser returns to after sign-in: `http://127.0.0.1:<port>/callback`."""

    @model_validator(mode="after")
    def secret_needs_client(self) -> McpOAuthConfig:
        if self.client_secret_env is not None and self.client_id is None:
            raise ValueError("An OAuth client secret needs a pre-registered client_id")
        return self


class McpConfig(_Settings):
    namespace: str = Field(pattern=r"^[a-z][a-z0-9_]{0,23}$")
    transport: Literal["stdio", "http"]
    remote_tools: dict[Annotated[str, StringConstraints(min_length=1, max_length=128)], DeclaredEffects] = Field(
        min_length=1, max_length=512
    )
    timeout_s: float = Field(default=30, gt=0, le=600, allow_inf_nan=False)
    command: str | None = Field(default=None, min_length=1, max_length=4096)
    args: list[str] = Field(default_factory=list, max_length=256)
    cwd: Path | None = None
    env_refs: dict[EnvName, EnvName] = Field(default_factory=dict)
    url: str | None = None
    header_env: dict[HeaderName, EnvName] = Field(default_factory=dict)
    credential_command: CredentialCommandConfig | None = None
    oauth: McpOAuthConfig | None = None

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str | None) -> str | None:
        return _http_url(value) if value is not None else None

    @model_validator(mode="after")
    def transport_fields(self) -> McpConfig:
        if self.transport == "stdio":
            if self.command is None or not self.command.strip() or any(ord(c) < 32 for c in self.command):
                raise ValueError("A stdio MCP connection requires a valid executable command")
            if self.url is not None or self.header_env or self.credential_command is not None or self.oauth is not None:
                raise ValueError("HTTP URL/header/credential/OAuth settings cannot be used with stdio MCP")
        elif self.url is None or self.command is not None or self.args or self.cwd is not None or self.env_refs:
            raise ValueError("HTTP MCP requires a URL and cannot use stdio command/environment settings")
        headers = [name.lower() for name in self.header_env]
        if len(headers) != len(set(headers)) or set(headers).intersection(_BLOCKED_HEADERS):
            raise ValueError("MCP authentication headers must be unique and cannot control HTTP routing/framing")
        if self.credential_command is not None:
            header = self.credential_command.header.lower()
            if header in headers:
                raise ValueError("The credential command header is already set by header_env")
            if header in _BLOCKED_HEADERS or header in _MCP_PROTOCOL_HEADERS:
                raise ValueError("The credential command header cannot be a routing/framing or MCP protocol header")
        if self.oauth is not None:
            if self.credential_command is not None:
                raise ValueError("Use either OAuth or a credential command for one MCP entry, not both")
            if "authorization" in headers:
                raise ValueError("OAuth sets the Authorization header; remove it from header_env")
        exported = [f"mcp__{self.namespace}__{re.sub(r'[^a-zA-Z0-9_]', '_', name)}" for name in self.remote_tools]
        if len(exported) != len(set(exported)) or any(len(name) > 64 for name in exported):
            raise ValueError("Selected MCP tool names collide after namespacing or exceed the 64-character limit")
        return self


class SessionBudgetConfig(_Settings):
    """One session-cumulative model-call ledger, retained across follow-ups/model changes.

    Relative deadlines apply to model calls from session setup, not all tools or
    human waiting. Observed cost is a threshold, not a hard billing guarantee.
    """

    max_calls: int = Field(default=100, ge=1)
    max_observed_cost_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    deadline_after_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class CapabilitySettings(_Settings):
    tool_packs: list[str] | None = None
    search: SearchConfig = Field(default_factory=SearchConfig)
    reader: ReaderConfig = Field(default_factory=ReaderConfig)
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    code_search: CodeSearchConfig = Field(default_factory=CodeSearchConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)
    mcp: list[McpConfig] = Field(default_factory=list, max_length=32)
    session_budget: SessionBudgetConfig | None = None
    system2_fallbacks: list[ModelRef] = Field(default_factory=list, max_length=7)

    @model_validator(mode="before")
    @classmethod
    def renamed_tables(cls, data: object) -> object:
        """`[capabilities.executor]` became `[capabilities.runner]` in 0.19.0; say so instead of "extra input"."""
        if isinstance(data, dict) and "executor" in data:
            raise ValueError(
                '[capabilities.executor] is now [capabilities.runner]: rename the table, and its backend = "..." to '
                'name = "..." ([capabilities.executor.options] is [capabilities.runner.options])'
            )
        return data

    @model_validator(mode="after")
    def unique_configuration(self) -> CapabilitySettings:
        namespaces = [entry.namespace for entry in self.mcp]
        if len(namespaces) != len(set(namespaces)):
            raise ValueError("MCP namespaces must be unique within a session")
        exported = [
            f"mcp__{entry.namespace}__{re.sub(r'[^a-zA-Z0-9_]', '_', name)}"
            for entry in self.mcp
            for name in entry.remote_tools
        ]
        if len(exported) != len(set(exported)):
            raise ValueError("Selected MCP tool names collide across connections after namespacing")
        if len(self.system2_fallbacks) != len(set(self.system2_fallbacks)):
            raise ValueError("System 2 fallback model references must be distinct")
        if self.tool_packs is not None:
            self.effective_tool_packs(())
        return self

    def effective_tool_packs(self, profile_packs: Sequence[str]) -> list[str]:
        """Validate selection after the primary model profile is known; start nothing."""
        names = list(profile_packs if self.tool_packs is None else self.tool_packs)
        if len(names) != len(set(names)) or any(name not in PACKS for name in names):
            raise ValueError("Tool packs must be distinct names from the existing tool-pack registry")
        return names
