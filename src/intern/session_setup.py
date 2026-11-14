"""Assemble existing capabilities once, with explicit session resource ownership.

This is startup plumbing, not a tool registry or a reasoning extension. Selected
tools keep the ordinary Tool boundary; providers keep the existing transport and
DecisionClient paths. No service is opened by importing this module.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx
from pydantic import SecretStr, ValidationError

from intern import capability_toggles, names
from intern.browser import BROWSER_NEEDS, BrowserOptions, BrowserSession, browser_tool
from intern.capability_settings import (
    DEFAULT_ALLOWED_ENV,
    CapabilitySettings,
    McpConfig,
    ProcessConfig,
    ReaderConfig,
    RunnerConfig,
    SearchConfig,
    SecretBindingConfig,
    SecretsConfig,
)
from intern.decision_failover import BudgetDecisionBackend
from intern.executor import ClosableExecutor, Executor, JobExecutor, LocalExecutor, mount_of, shell_env
from intern.extensions.code_search import CodeSearch
from intern.extensions.workspace.extension import Workspace
from intern.hooks import Extension
from intern.loop import EngineDeps, ModelBinding
from intern.mcp_client import MCP_NEEDS, StdioConfig, http_connection, stdio_connection
from intern.mcp_credentials import (
    CommandCredential,
    CredentialCommand,
    CredentialCommandError,
    run_credential_command,
)
from intern.mcp_oauth import OAuthSignInRequired, OAuthStore, oauth_auth, oauth_store
from intern.models.catalog import RunnerInfo
from intern.models.registry import ConfigError
from intern.needs import Needs
from intern.processes import ProcessManager
from intern.providers.failover import (
    BudgetPolicy,
    BudgetProvider,
    FallbackProvider,
    FallbackTarget,
    RunBudget,
    TargetConfig,
)
from intern.runners import LocalRunner, load_runner, runner_names
from intern.secrets import (
    CommandBinding,
    CommandSecrets,
    EnvironmentBinding,
    EnvironmentSecrets,
    RoutedSecrets,
    SecretExecutor,
    SecretRedactor,
    SecretRef,
    SecretResolver,
)
from intern.session_resources import ResourceSetupError, SessionResources
from intern.tools.base import Tool
from intern.tools.files import EditTool, ReadTool, WriteTool
from intern.tools.process import ProcessTool
from intern.tools.search import FindTool, GrepTool
from intern.tools.shell import BashTool
from intern.tools.web_read import WebReadTool
from intern.tools.web_search import WebSearchTool
from intern.web.reader import ReadSettings, WebReader
from intern.web.search import SEARCH_PROVIDER_ENV, HTTPSearchProvider, SearchError, SearchSettings
from intern.workspace_tools import workspace_tools

BindingResolver = Callable[[str], Awaitable[ModelBinding]]


@dataclass
class PreparedSession:
    deps: EngineDeps
    resources: SessionResources | None
    resolve_model: BindingResolver | None
    budget: RunBudget | None = None


@dataclass
class _McpPlan:
    config: McpConfig
    stdio: StdioConfig | None
    headers: dict[str, SecretStr]
    credential: CredentialCommand | None = None
    credential_secrets: list[SecretStr] = field(default_factory=list[SecretStr])
    """The `env_refs` values, redacted from a failing credential command's stderr."""
    oauth: OAuthStore | None = None
    oauth_secret: SecretStr | None = None


CREDENTIAL_TIMEOUT_S = 30.0
"""Most a credential command may take; also capped below the entry's own call timeout."""


def _secret(env: Mapping[str, str], name: str, purpose: str) -> SecretStr:
    value = env.get(name, "")
    if not value.strip():
        raise ConfigError(f"{purpose} needs the environment variable {name}")
    return SecretStr(value)


def _search(config: SearchConfig, env: Mapping[str, str]) -> HTTPSearchProvider:
    settings = _search_settings(config, env).model_copy(
        update={"timeout_s": config.timeout_s, "max_attempts": config.max_attempts}
    )
    return HTTPSearchProvider(settings)


def _search_settings(config: SearchConfig, env: Mapping[str, str]) -> SearchSettings:
    source = dict(env)
    if config.backend is not None:
        source[SEARCH_PROVIDER_ENV] = config.backend
    backend = source.get(SEARCH_PROVIDER_ENV, "").strip().lower()
    if config.searxng_url is not None:
        source["SEARXNG_URL"] = config.searxng_url
    if config.api_key_env is not None:
        key_names = {"brave": "BRAVE_SEARCH_API_KEY", "tavily": "TAVILY_API_KEY", "exa": "EXA_API_KEY"}
        if backend not in key_names:
            raise ConfigError("capabilities.search.api_key_env needs a brave, tavily or exa backend")
        source[key_names[backend]] = _secret(env, config.api_key_env, "Search").get_secret_value()
    try:
        return SearchSettings.from_env(source)
    except SearchError as error:
        reason = str(error).rstrip(".")
        raise ConfigError(f"Web search is not set up: {reason[:1].lower()}{reason[1:]}") from None


def _require(needs: Needs | None) -> None:
    """Refuse a capability whose extra this environment lacks, in the capability's own words."""
    if needs is not None and (problem := needs.problem()) is not None:
        raise ConfigError(problem)


def capability_needs(toggle_id: str) -> Needs | None:
    """What the capability behind a toggle needs installed, as that capability declares it (an extension's `needs`,
    the browser's, MCP's), else None: a tool pack or the memory drive needs nothing beyond the base install."""
    if toggle_id == f"{capability_toggles.PREFIX}browser":
        return BROWSER_NEEDS
    if toggle_id == f"{capability_toggles.PREFIX}code-search":
        return CodeSearch.needs
    if toggle_id == f"{capability_toggles.PREFIX}workspace":
        return Workspace.needs
    if toggle_id.startswith(capability_toggles.MCP_PREFIX):
        return MCP_NEEDS
    return None


def capability_problem(
    toggle_id: str, settings: CapabilitySettings, env: Mapping[str, str], *, installs: bool = False
) -> str | None:
    """Why a capability toggle cannot switch on in this install and environment (a missing extra or search
    provider), else None. Only what needs no session: `prepare_session` raises these same reasons, and also
    rejects combinations, such as an extension's process tool on an executor that runs no background jobs. With
    `installs`, for a front end that installs a missing extra on switch-on, an extra this environment can install is
    none."""
    try:
        if toggle_id == f"{capability_toggles.PREFIX}web-search":
            _search_settings(settings.search, env)
        elif not (installs and (needs := capability_needs(toggle_id)) is not None and needs.installable()):
            _require(capability_needs(toggle_id))
    except ConfigError as error:
        return str(error)
    return None


def runner_config(name: str, configured: RunnerConfig) -> RunnerConfig:
    """The `[capabilities.runner]` a session runs on when it picks `name`: the configured one as configured, any other
    on its default options (never another runner's). Raises ValueError when `name` cannot run on defaults."""
    if name == configured.name:
        return configured.model_copy(deep=True)
    try:
        return RunnerConfig.model_validate({"name": name})
    except ValidationError:
        raise ValueError(
            f"{name} needs its options: set [capabilities.runner] name = {name!r} and its "
            "[capabilities.runner.options] in config.toml"
        ) from None


def runner_problem(name: str, configured: RunnerConfig) -> str | None:
    """Why a session cannot pick runner `name` in this install and environment, else None: it cannot load, it needs
    options only config.toml can give, or it reports its own reason (a missing extra or key)."""
    try:
        runner = load_runner(name)
        config = runner_config(name, configured)
    except ValueError as error:
        return str(error)
    if not isinstance(config.options, runner.Options):
        return None if name == configured.name else f"{name} cannot load its options"
    return runner.problem(config.options)


def runner_choices(configured: RunnerConfig) -> list[RunnerInfo]:
    """Every runner a session can pick, each with its name, note and why it cannot start here (None when it can)."""
    choices = []
    for name in runner_names(configured.name):
        try:
            runner = load_runner(name)
            title, note = runner.title or name, runner.note
        except ValueError:
            title, note = name, "a runner this install cannot load"
        choices.append(
            RunnerInfo(
                id=name,
                name=title,
                description=note,
                default=name == configured.name,
                unavailable=runner_problem(name, configured),
            )
        )
    return choices


def _credential(config: McpConfig, env: Mapping[str, str], home: Path) -> tuple[CredentialCommand, list[SecretStr]]:
    """The credential command's whole environment: the default allowlist, PATH without our venv, and env_refs."""
    assert config.credential_command is not None
    entry = config.credential_command
    base = shell_env(env)
    command_env = {name: SecretStr(base[name]) for name in sorted(DEFAULT_ALLOWED_ENV) if base.get(name)}
    refs = {
        target: _secret(env, source, f"MCP {config.namespace} credential command")
        for target, source in entry.env_refs.items()
    }
    home.mkdir(parents=True, exist_ok=True)  # the command's working directory, which the model does not write
    spec = CredentialCommand(
        command=entry.command,
        args=tuple(entry.args),
        env={**command_env, **refs},
        cwd=home,
        timeout_s=min(CREDENTIAL_TIMEOUT_S, 0.8 * config.timeout_s),
    )
    return spec, list(refs.values())


def _mcp_plan(config: McpConfig, env: Mapping[str, str], cwd: Path, home: Path) -> _McpPlan:
    _require(MCP_NEEDS)
    if config.transport == "stdio":
        assert config.command is not None
        directory = config.cwd
        if directory is not None:
            directory = directory.expanduser()
            directory = directory if directory.is_absolute() else cwd / directory
            if not directory.is_dir():
                raise ConfigError(f"MCP {config.namespace} needs an existing working directory")
        return _McpPlan(
            config,
            StdioConfig(
                command=config.command,
                args=config.args,
                cwd=directory or cwd,
                env={
                    target: _secret(env, source, f"MCP {config.namespace}")
                    for target, source in config.env_refs.items()
                },
            ),
            {},
        )
    headers = {header: _secret(env, source, f"MCP {config.namespace}") for header, source in config.header_env.items()}
    if config.oauth is not None:
        secret = oauth_client_secret(config, env)
        return _McpPlan(config, None, headers, oauth=oauth_store(config, home, secret), oauth_secret=secret)
    if config.credential_command is None:
        return _McpPlan(config, None, headers)
    spec, secrets = _credential(config, env, home)
    return _McpPlan(config, None, headers, spec, secrets)


def _command_binding(
    name: str, binding: SecretBindingConfig, env: Mapping[str, str], home: Path
) -> tuple[CommandBinding, list[SecretStr]]:
    """A command binding with its whole environment (the default allowlist, PATH without our venv, and env_refs) and
    the env_refs values, to redact from what the command prints on failure."""
    assert binding.command is not None
    base = shell_env(env)
    command_env = {key: SecretStr(base[key]) for key in sorted(DEFAULT_ALLOWED_ENV) if base.get(key)}
    refs = {target: _secret(env, source, f"Secret {name} command") for target, source in binding.env_refs.items()}
    home.mkdir(parents=True, exist_ok=True)  # the command's working directory, which the model does not write
    command = CredentialCommand(
        command=binding.command,
        args=tuple(binding.args),
        env={**command_env, **refs},
        cwd=home,
        timeout_s=binding.timeout_s,
    )
    return CommandBinding(reference=SecretRef(name=name), command=command, scopes=binding.scopes), list(refs.values())


async def _secret_resolver(secrets: SecretsConfig, env: Mapping[str, str], home: Path) -> SecretResolver:
    """The configured bindings as one resolver. Every injected value is resolved here, so a missing variable or a
    failing command stops setup before any command runs."""
    environment: list[EnvironmentBinding] = []
    commands: list[CommandBinding] = []
    redact: list[SecretStr] = []
    for name, binding in secrets.bindings.items():
        if binding.source_env is not None:
            reference = SecretRef(name=name)
            environment.append(
                EnvironmentBinding(reference=reference, source_env=binding.source_env, scopes=binding.scopes)
            )
        else:
            command, values = _command_binding(name, binding, env, home)
            commands.append(command)
            redact.extend(values)
    from_env, from_commands = EnvironmentSecrets(environment, env), CommandSecrets(commands)
    resolver = RoutedSecrets(
        {b.reference.name: from_env for b in environment} | {b.reference.name: from_commands for b in commands}
    )
    for reference in secrets.inject.values():
        source = secrets.bindings[reference].source_env
        if source is not None:
            _secret(env, source, "Configured command secret")
            continue
        try:
            await resolver.resolve(SecretRef(name=reference), scope=secrets.scope)
        except CredentialCommandError as error:
            tail = SecretRedactor(redact).redact(error.stderr_tail).strip()
            detail = f": {tail}" if tail else ""
            raise ConfigError(f"Secret {reference} command failed ({error.reason}){detail}") from None
    return resolver


STORE_SCOPE = "store"


def _store_executor(executor: Executor, store: Mapping[str, SecretStr]) -> SecretExecutor:
    """The `[secrets]` values injected into every command and job, over the user's own environment on this machine
    (minus our virtualenv, as LocalExecutor does), and redacted from what they print. Another runner keeps the
    short allowlist, since this machine's variables mean nothing there."""
    if type(executor) is LocalExecutor:
        base = {
            name: value
            for name, value in shell_env(os.environ).items()
            if name not in store and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        }
        allowed = frozenset(base)
    else:
        base, allowed = dict(os.environ), frozenset(DEFAULT_ALLOWED_ENV - store.keys())
    bindings = [
        EnvironmentBinding(reference=SecretRef(name=name), source_env=name, scopes=frozenset({STORE_SCOPE}))
        for name in store
    ]
    values = {name: value.get_secret_value() for name, value in store.items()}
    return SecretExecutor(
        executor,
        EnvironmentSecrets(bindings, values),
        scope=STORE_SCOPE,
        inject={name: SecretRef(name=name) for name in store},
        base_env=base,
        allowed_env=allowed,
    )


def oauth_client_secret(config: McpConfig, env: Mapping[str, str]) -> SecretStr | None:
    """The pre-registered OAuth client secret an entry names, if any; a named but unset variable fails setup."""
    source = config.oauth.client_secret_env if config.oauth is not None else None
    return _secret(env, source, f"MCP {config.namespace} OAuth") if source else None


def _require_sign_in(plans: list[_McpPlan]) -> None:
    for plan in plans:
        if plan.oauth is not None and not plan.oauth.signed_in():
            raise ConfigError(str(OAuthSignInRequired(plan.config.namespace)))


async def _first_credentials(plans: list[_McpPlan]) -> dict[str, SecretStr]:
    """Run each credential command once, before any MCP service starts, so a broken one fails setup readably."""
    values: dict[str, SecretStr] = {}
    for plan in plans:
        if plan.credential is None:
            continue
        try:
            values[plan.config.namespace] = await run_credential_command(plan.credential)
        except CredentialCommandError as error:
            tail = SecretRedactor(plan.credential_secrets).redact(error.stderr_tail).strip()
            detail = f": {tail}" if tail else ""
            raise ConfigError(
                f"MCP {plan.config.namespace} credential command failed ({error.reason}){detail}"
            ) from None
    return values


def capability_extensions(settings: CapabilitySettings) -> list[Extension]:
    """Extensions a capability setting turns on. Code search and the workspace load only this way, never by
    extension name; each needs its extra, checked here so a run fails before it starts, not at the first tool."""
    extensions: list[Extension] = []
    for extension, enabled in ((CodeSearch, settings.code_search.enabled), (Workspace, settings.workspace.enabled)):
        if enabled:
            _require(extension.needs)
            extensions.append(extension())
    return extensions


def _storage(base: Path | None, default: Path, scope: str, cwd: Path) -> Path:
    root = (base or default).expanduser()
    if not root.is_absolute():
        root = cwd / root if base is not None else root.absolute()
    if root.is_symlink():
        raise ConfigError("Capability storage root must not be a symlink")
    # Keep artifacts across close, in separate private session directories. Never
    # use a journal-provided session ID directly as a filesystem component.
    return root / scope


def _unique_tools(tools: list[Tool]) -> None:
    names = [tool.name for tool in tools]
    if len(names) != len(set(names)):
        raise ConfigError("Configured capability tools collide with an existing session tool")


class _Models:
    def __init__(self, settings: CapabilitySettings, resolver: BindingResolver | None) -> None:
        self.settings, self.resolver = settings, resolver
        config = settings.session_budget
        self.budget = (
            RunBudget(
                BudgetPolicy(
                    max_calls=config.max_calls,
                    max_observed_cost_usd=config.max_observed_cost_usd,
                    deadline_unix=time.time() + config.deadline_after_s if config.deadline_after_s else None,
                )
            )
            if config is not None
            else None
        )

    async def resolve(self, ref: str) -> ModelBinding:
        if self.resolver is None:
            raise ConfigError("This session has no model resolver")
        return await self.wrap(await self.resolver(ref))

    async def wrap(self, primary: ModelBinding) -> ModelBinding:
        budget = self.budget
        if not self.settings.system2_fallbacks:
            return replace(primary, provider=BudgetProvider(primary.provider, budget)) if budget else primary
        if self.resolver is None:
            raise ConfigError("Configured System 2 fallbacks need the existing model resolver")
        targets = [primary]
        for ref in self.settings.system2_fallbacks:
            # A target becoming primary after a model switch is no longer its
            # own backup. The remaining explicitly configured targets stay put.
            if ref == primary.ref:
                continue
            backup = await self.resolver(ref)
            if any(backup.ref == model.ref for model in targets):
                raise ConfigError("Configured fallback targets resolve to the same model")
            left, right = primary.profile, backup.profile
            if (
                backup.window.working < primary.window.working
                or backup.window.advertised < primary.window.advertised
                or left.sampling != right.sampling
                or left.reasoning_effort != right.reasoning_effort
                or left.reasoning_replay != right.reasoning_replay
                or left.compat != right.compat
                or left.vision != right.vision
            ):
                raise ConfigError(
                    "System 2 backup profiles need compatible request settings/vision "
                    "and at least the primary context window"
                )
            targets.append(backup)
        providers = [
            FallbackTarget(
                TargetConfig(
                    ref=model.ref,
                    wire_model=model.wire_model,
                    max_output_tokens=model.profile.output_cap(model.window.advertised),
                ),
                BudgetProvider(model.provider, budget) if budget else model.provider,
            )
            for model in targets
        ]
        return replace(primary, provider=FallbackProvider(providers))


async def prepare_session(
    deps: EngineDeps,
    settings: CapabilitySettings,
    *,
    pack_tools: list[Tool],
    env: Mapping[str, str],
    home: Path,
    resolve_model: BindingResolver | None,
    secret_store: Mapping[str, SecretStr] | None = None,
    harness_only: frozenset[str] = frozenset(),
) -> PreparedSession:
    """Validate completely before starting selected services; leave inputs intact on failure.

    The builder has already applied effective_tool_packs before extension-tool
    registration. This function never replaces an extension's same-name tool.
    """
    settings = settings.model_copy(deep=True)
    settings.effective_tool_packs(deps.system2.profile.tool_packs)
    tools = list(deps.tools)
    _unique_tools(tools)
    models = _Models(settings, resolve_model)
    system2 = await models.wrap(deps.system2)
    scope = hashlib.sha256(deps.journal.session_id.encode()).hexdigest()[:16] + "-" + uuid.uuid4().hex
    storage_root = home / "session-resources"
    selected = {tool.name: tool for tool in tools}
    # An extension can return configured instances of the stock tool classes.
    # Preserve provenance across the existing same-name extension merge.
    pack_ids = {id(tool) for tool in pack_tools}
    replacements: dict[str, Tool] = {}

    search = selected.get("web_search")
    if id(search) in pack_ids and type(search) is WebSearchTool:
        replacements["web_search"] = WebSearchTool(_search(settings.search, env))
    elif search is not None and settings.search != SearchConfig():
        raise ConfigError("Search settings conflict with an extension replacement of web_search")
    executor = deps.executor
    owned: ClosableExecutor | None = None  # only an executor this setup built; never the caller's deps.executor
    try:
        runner = load_runner(settings.runner.name)
    except ValueError as error:
        raise ConfigError(str(error)) from None
    if not isinstance(runner, LocalRunner):
        workspace = (settings.runner.workspace or deps.cwd).expanduser()
        workspace = workspace if workspace.is_absolute() else deps.cwd / workspace
        options = settings.runner.options
        if not isinstance(options, runner.Options):
            raise ConfigError(f"Runner {runner.name!r} could not check its options; see [capabilities.runner.options]")
        if (problem := runner.problem(options)) is not None:
            raise ConfigError(problem)
        try:
            executor = runner.create(options, workspace=workspace, cwd=deps.cwd)
        except ValueError as error:
            raise ConfigError(str(error)) from None
        owned = executor if isinstance(executor, ClosableExecutor) else None
        if runner.jobs != isinstance(executor, JobExecutor):
            raise ConfigError(
                f"Runner {runner.name!r} says jobs = {runner.jobs} but its executor "
                f"{'has no' if runner.jobs else 'has a'} start_job; the two must agree"
            )
    if runner.isolated:
        mount = mount_of(executor)
        if mount is None:
            raise ConfigError(f"Runner {runner.name!r} is isolated but its executor gives no mount")
        try:
            mount.to_inner(deps.cwd)
        except ValueError:
            raise ConfigError(
                f"The {runner.name} runner's workspace must contain the session working directory"
            ) from None
        stock = {
            tool.name: type(tool) for tool in (ReadTool(), WriteTool(), EditTool(), BashTool(), GrepTool(), FindTool())
        }
        for tool in tools:
            if tool.name in stock and (id(tool) not in pack_ids or type(tool) is not stock[tool.name]):
                raise ConfigError(
                    f"The {runner.name} runner cannot replace an extension's file/shell tools; "
                    "configure a compatible tool adapter"
                )
        replacements.update({tool.name: tool for tool in workspace_tools() if tool.name in selected})

    reader = selected.get("web_read")
    if id(reader) in pack_ids and type(reader) is WebReadTool:
        mount = mount_of(executor)
        default_store = storage_root / "web" if mount is None else mount.host / names.RESOURCES_DIR / "web"
        store = _storage(settings.reader.storage_dir, default_store, scope, deps.cwd)
        display_dir = None
        if mount is not None:
            try:
                display_dir = Path(mount.to_inner(store))
            except ValueError:
                raise ConfigError("Sandboxed reader storage must be inside its mounted workspace") from None
        replacements["web_read"] = WebReadTool(
            WebReader(ReadSettings.model_validate(settings.reader.model_dump(exclude={"storage_dir"}))),
            storage_dir=store,
            display_dir=display_dir,
        )
    elif reader is not None and settings.reader != ReaderConfig():
        raise ConfigError("Reader settings conflict with an extension replacement of web_read")

    secrets = settings.secrets
    # Keys the harness reads itself (a model's, search's, MCP's) are not the agent's to use.
    for_agent = {name: value for name, value in (secret_store or {}).items() if name not in harness_only}
    if for_agent and secrets.inject:
        raise ConfigError("[secrets] and [capabilities.secrets] both give commands secrets; keep one")
    if for_agent:
        executor = _store_executor(executor, for_agent)
    elif secrets.inject:
        resolver = await _secret_resolver(secrets, env, home)
        executor = SecretExecutor(
            executor,
            resolver,
            scope=secrets.scope,
            inject={name: SecretRef(name=ref) for name, ref in secrets.inject.items()},
            base_env=env,
            allowed_env=secrets.allowed_env,
        )

    # One job manager per session, wherever its commands can run in the background: the `process` tool's jobs run in
    # it, and it keeps a record of every command running in the session (`bash` calls, what they leave running).
    process = selected.get("process")
    inner = executor.executor if isinstance(executor, SecretExecutor) else executor
    runs_jobs = type(inner) is LocalExecutor or isinstance(inner, JobExecutor)
    stock_process = id(process) in pack_ids and type(process) is ProcessTool
    if type(process) is ProcessTool and not stock_process and not runs_jobs:
        raise ConfigError("The process tool needs an executor that runs background jobs; this one does not")
    if process is not None and not stock_process and settings.process != ProcessConfig():
        raise ConfigError("Process settings conflict with an extension replacement of process")
    manager = ProcessManager(**settings.process.model_dump()) if runs_jobs else None
    if stock_process and manager is not None:
        replacements["process"] = ProcessTool(manager)
    # A base tool, so an executor that runs no background jobs goes without it rather than refusing the session.
    tools = [
        replacements.get(tool.name, tool)
        for tool in tools
        if not (stock_process and manager is None and tool is process)
    ]
    if manager is None or manager.move_after_s == 0:
        # Its commands never go on in the background: the harness's `bash` says how to leave one running instead.
        ours = pack_ids | {id(tool) for tool in replacements.values()}
        for tool in tools:
            if type(tool) is BashTool and id(tool) in ours:
                tool.stays = True

    browser: BrowserSession | None = None
    if settings.browser.enabled:
        _require(BROWSER_NEEDS)
        browser = BrowserSession(
            _storage(settings.browser.artifact_dir, storage_root / "browser", scope, deps.cwd),
            BrowserOptions.model_validate(settings.browser.model_dump(exclude={"enabled", "artifact_dir"})),
        )
        tools.append(browser_tool(browser))
    plans = [_mcp_plan(entry, env, deps.cwd, home) for entry in settings.mcp]
    _unique_tools(tools)
    exported = [
        f"mcp__{plan.config.namespace}__{re.sub(r'[^a-zA-Z0-9_]', '_', name)}"
        for plan in plans
        for name in plan.config.remote_tools
    ]
    if set(exported).intersection(tool.name for tool in tools):
        raise ConfigError("Selected MCP tool names collide with existing session tools")
    _require_sign_in(plans)
    first_credentials = await _first_credentials(plans)

    resources: SessionResources | None = None
    if manager is not None or browser is not None or plans or owned is not None:
        resources = SessionResources()
        stage = "capability resources"

        async def open_resources(stack: AsyncExitStack) -> list[Tool]:
            nonlocal stage
            if owned is not None:
                # Registered first so it closes last, after anything that may still run a command in it.
                stack.push_async_callback(owned.aclose)
            if manager is not None:
                stack.push_async_callback(manager.aclose)
            if browser is not None:
                # Constructor and callback only: entering BrowserSession would
                # eagerly start Chromium instead of waiting for its first use.
                stack.push_async_callback(browser.aclose)
            remote: list[Tool] = []
            for plan in plans:
                entry = plan.config
                stage = f"MCP {entry.namespace}"
                if plan.stdio is not None:
                    connection = await stack.enter_async_context(
                        stdio_connection(plan.stdio, entry.namespace, timeout_s=entry.timeout_s)
                    )
                else:
                    assert entry.url is not None
                    auth: httpx.Auth | None = None
                    if plan.credential is not None and entry.credential_command is not None:
                        auth = CommandCredential(
                            plan.credential, entry.credential_command.header, first_credentials[entry.namespace]
                        )
                    elif plan.oauth is not None:
                        auth = oauth_auth(entry, plan.oauth, plan.oauth_secret)
                    client = await stack.enter_async_context(
                        httpx.AsyncClient(
                            headers={name: secret.get_secret_value() for name, secret in plan.headers.items()},
                            auth=auth,
                            timeout=entry.timeout_s,
                            follow_redirects=False,
                            trust_env=False,
                        )
                    )
                    connection = await stack.enter_async_context(
                        http_connection(entry.url, entry.namespace, timeout_s=entry.timeout_s, http_client=client)
                    )
                remote.extend(await connection.tools(entry.remote_tools))
            _unique_tools([*tools, *remote])
            return [*tools, *remote]

        try:
            tools = await resources.open(open_resources)
        except ResourceSetupError as error:
            rollback = "; resource cleanup is incomplete" if not error.report.cleanup_complete else ""
            raise ConfigError(
                f"{stage} setup failed ({error.error_type}){rollback}; check its configured service and tool selection"
            ) from None
    if models.budget is not None:
        deps.decisions.backend = BudgetDecisionBackend(deps.decisions.backend, models.budget)
    redact = SecretRedactor(list(secret_store.values())).redact if secret_store else deps.redact
    prepared = replace(deps, tools=tools, executor=executor, system2=system2, redact=redact, processes=manager)
    resolver = models.resolve if settings.system2_fallbacks or models.budget is not None else resolve_model
    return PreparedSession(prepared, resources, resolver, models.budget)
