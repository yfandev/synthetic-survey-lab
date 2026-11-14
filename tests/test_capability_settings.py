from __future__ import annotations

import math
import tomllib

import pytest
from pydantic import ValidationError

from intern.capability_settings import CapabilitySettings
from intern.runners.docker import DockerConfig


def test_empty_settings_preserve_profile_selection_and_do_not_enable_services():
    settings = CapabilitySettings()
    assert settings.tool_packs is None
    assert settings.effective_tool_packs(["base", "web-search"]) == ["base", "web-search"]
    assert not settings.browser.enabled and not settings.mcp
    assert settings.runner.name == "local" and not settings.secrets.inject
    assert settings.session_budget is None and not settings.system2_fallbacks
    assert CapabilitySettings(tool_packs=[]).effective_tool_packs(["base"]) == []


def test_saved_toml_covers_all_eight_existing_components_without_credential_values():
    settings = CapabilitySettings.model_validate(
        tomllib.loads("""
tool_packs = ["base", "web-search", "web-read"]
system2_fallbacks = ["vllm/backup"]
[search]
backend = "brave"
api_key_env = "BRAVE_SEARCH_API_KEY"
[reader]
storage_dir = "retained-reader"
max_redirects = 3
[process]
max_running = 2
max_records = 8
[browser]
enabled = true
artifact_dir = "retained-screenshots"
[runner]
name = "docker"
[runner.options]
image = "python:3.12.11-slim-bookworm"
network = "none"
[secrets]
scope = "command"
inject = { API_TOKEN = "deploy" }
[secrets.bindings.deploy]
source_env = "DEPLOY_TOKEN"
scopes = ["command"]
[[mcp]]
namespace = "docs"
transport = "http"
url = "https://docs.example.test/mcp"
header_env = { Authorization = "DOCS_AUTHORIZATION" }
remote_tools = { search = ["read", "network"] }
[session_budget]
max_calls = 12
max_observed_cost_usd = 0.25
deadline_after_s = 60
""")
    )
    assert settings.effective_tool_packs(["base", "process"]) == ["base", "web-search", "web-read"]
    assert settings.search.api_key_env == "BRAVE_SEARCH_API_KEY"
    assert isinstance(settings.runner.options, DockerConfig) and settings.runner.options.network == "none"
    assert settings.browser.artifact_dir is not None and settings.browser.artifact_dir.name == "retained-screenshots"
    assert settings.secrets.bindings["deploy"].source_env == "DEPLOY_TOKEN"
    assert settings.mcp[0].remote_tools == {"search": frozenset({"read", "network"})}
    assert settings.session_budget is not None and settings.session_budget.max_calls == 12


@pytest.mark.parametrize("executor", [{"name": "docker"}, {"name": "local"}])
def test_process_runs_with_injected_secrets_on_any_job_capable_executor(executor):
    """The process tool runs where commands run, Docker included, and with injected secrets too."""
    plain = CapabilitySettings.model_validate({"runner": executor})
    assert plain.effective_tool_packs(["base", "process"]) == ["base", "process"]
    secret = {
        "bindings": {"deploy": {"source_env": "DEPLOY_TOKEN", "scopes": ["executor"]}},
        "inject": {"API_TOKEN": "deploy"},
    }
    settings = CapabilitySettings.model_validate({"runner": executor, "secrets": secret})
    assert settings.effective_tool_packs(["base", "process"]) == ["base", "process"]
    CapabilitySettings.model_validate({"tool_packs": ["process"], "runner": executor, "secrets": secret})


@pytest.mark.parametrize("field", ["search", "reader", "process", "browser", "runner", "secrets"])
def test_unknown_nested_options_are_rejected_without_echoing_values(field):
    with pytest.raises(ValidationError) as caught:
        CapabilitySettings.model_validate({field: {"api_key": "private-synthetic-value"}})
    assert "private-synthetic-value" not in str(caught.value)


@pytest.mark.parametrize(
    "bad",
    [
        {"tool_packs": ["missing"]},
        {"tool_packs": ["base", "base"]},
        {"runner": {"workspace": "/tmp/work"}},
        {"runner": {"options": {"network": "bridge"}}},
        {"runner": {"name": "docker", "options": {"mystery": True}}},
        {"process": {"max_running": 4, "max_records": 2}},
        {"search": {"api_key_env": "not-an-environment-name"}},
        {"search": {"backend": "brave", "searxng_url": "http://localhost/search"}},
        {"search": {"backend": "searxng", "api_key_env": "SECRET"}},
        {"secrets": {"inject": {"API_KEY": "missing"}}},
        {"secrets": {"bindings": {"x": {"source_env": "TOKEN", "scopes": ["other"]}}, "inject": {"KEY": "x"}}},
        {"secrets": {"bindings": {"x": {"source_env": "TOKEN", "scopes": ["executor"]}}, "inject": {"PATH": "x"}}},
        {"secrets": {"bindings": {"x": {"scopes": ["executor"]}}}},
        {"secrets": {"bindings": {"x": {"source_env": "TOKEN", "command": "/bin/op", "scopes": ["executor"]}}}},
        {"secrets": {"bindings": {"x": {"command": "op", "scopes": ["executor"]}}}},
        {"secrets": {"bindings": {"x": {"source_env": "TOKEN", "args": ["read"], "scopes": ["executor"]}}}},
        {"session_budget": {"max_completed_output_tokens": 10}},
        {"system2_fallbacks": ["missing-provider"]},
        {"system2_fallbacks": ["vllm/one", "vllm/one"]},
        {"system2_fallbacks": [f"vllm/{i}" for i in range(8)]},
    ],
)
def test_invalid_or_ignored_configuration_is_not_silently_accepted(bad):
    with pytest.raises(ValidationError):
        CapabilitySettings.model_validate(bad)


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf, 0, -1])
@pytest.mark.parametrize(
    "section,field",
    [
        ("search", "timeout_s"),
        ("browser", "timeout_s"),
        ("session_budget", "max_observed_cost_usd"),
        ("session_budget", "deadline_after_s"),
    ],
)
def test_nonfinite_and_nonpositive_limits_are_rejected(section, field, number):
    with pytest.raises(ValidationError):
        CapabilitySettings.model_validate({section: {field: number}})


def mcp(**updates):
    return {
        "namespace": "docs",
        "transport": "stdio",
        "command": "server",
        "remote_tools": {"lookup": ["read"]},
        **updates,
    }


def test_stdio_mcp_uses_explicit_environment_references_and_effect_allowlist():
    settings = CapabilitySettings.model_validate({"mcp": [mcp(env_refs={"TOKEN": "DOCS_TOKEN"})]})
    assert settings.mcp[0].env_refs == {"TOKEN": "DOCS_TOKEN"}
    assert settings.mcp[0].remote_tools == {"lookup": frozenset({"read"})}


def test_pure_mcp_tool_can_explicitly_declare_no_effects():
    settings = CapabilitySettings.model_validate({"mcp": [mcp(remote_tools={"echo": []})]})
    assert settings.mcp[0].remote_tools == {"echo": frozenset()}
    entry = mcp()
    del entry["remote_tools"]
    with pytest.raises(ValidationError, match="remote_tools"):
        CapabilitySettings.model_validate({"mcp": [entry]})


@pytest.mark.parametrize(
    "bad",
    [
        mcp(remote_tools={}),
        mcp(remote_tools={"lookup": ["admin"]}),
        mcp(remote_tools={"look-up": ["read"], "look_up": ["read"]}),
        mcp(remote_tools={"x" * 64: ["read"]}),
        mcp(namespace="not-a-namespace"),
        mcp(env_refs={"TOKEN": "bad env"}),
        mcp(command=None),
        mcp(url="https://example.test/mcp"),
        mcp(header_env={"Authorization": "TOKEN"}),
        {"namespace": "docs", "transport": "http", "remote_tools": {"read": ["read"]}},
        mcp(transport="http", url="https://example.test/mcp"),
        mcp(transport="http", command=None, url="https://user:private-synthetic-value@example.test/mcp"),
        mcp(transport="http", command=None, url="https://example.test/mcp#fragment"),
        mcp(transport="http", command=None, url="https://example.test:invalid/mcp"),
        mcp(transport="http", command=None, url="https://example.test/mcp", header_env={"Host": "OVERRIDE"}),
        mcp(
            transport="http",
            command=None,
            url="https://example.test/mcp",
            header_env={"Authorization": "TOKEN", "authorization": "OTHER_TOKEN"},
        ),
    ],
)
def test_invalid_mcp_authority_or_transport_configuration_is_rejected(bad):
    with pytest.raises(ValidationError) as caught:
        CapabilitySettings.model_validate({"mcp": [bad]})
    assert "private-synthetic-value" not in str(caught.value)


def http_mcp(**changes: object) -> dict[str, object]:
    return mcp(transport="http", command=None, url="https://example.test/mcp", **changes)


def test_http_mcp_accepts_a_credential_command():
    entry = http_mcp(
        header_env={"x-pd-project-id": "PD_PROJECT"},
        credential_command={"command": "/usr/local/bin/token", "env_refs": {"CLIENT_SECRET": "PD_SECRET"}},
    )
    command = CapabilitySettings.model_validate({"mcp": [entry]}).mcp[0].credential_command
    assert command is not None
    assert command.header == "Authorization"
    assert command.env_refs == {"CLIENT_SECRET": "PD_SECRET"}


@pytest.mark.parametrize(
    "bad",
    [
        mcp(credential_command={"command": "/usr/local/bin/token"}),
        http_mcp(credential_command={"command": "token"}),
        http_mcp(credential_command={"command": "/usr/local/bin/to\nken"}),
        http_mcp(credential_command={"command": "/bin/token", "env_refs": {"SECRET": "bad env"}}),
        http_mcp(header_env={"authorization": "TOKEN"}, credential_command={"command": "/bin/token"}),
        http_mcp(credential_command={"command": "/bin/token", "header": "Host"}),
        http_mcp(credential_command={"command": "/bin/token", "header": "Mcp-Session-Id"}),
        http_mcp(credential_command={"command": "/bin/token", "header": "content-type"}),
        http_mcp(credential_command={"command": "/bin/token", "unknown": True}),
    ],
)
def test_invalid_credential_command_is_rejected(bad):
    with pytest.raises(ValidationError):
        CapabilitySettings.model_validate({"mcp": [bad]})


def test_http_mcp_accepts_oauth_with_defaults_or_a_preregistered_client():
    default = CapabilitySettings.model_validate({"mcp": [http_mcp(oauth={})]}).mcp[0].oauth
    assert default is not None and default.client_id is None and default.callback_port == 33418
    entry = http_mcp(oauth={"client_id": "app", "client_secret_env": "APP_SECRET", "scope": "read write"})
    configured = CapabilitySettings.model_validate({"mcp": [entry]}).mcp[0].oauth
    assert configured is not None and configured.client_secret_env == "APP_SECRET"


@pytest.mark.parametrize(
    "bad",
    [
        mcp(oauth={}),
        http_mcp(oauth={}, credential_command={"command": "/bin/token"}),
        http_mcp(oauth={}, header_env={"Authorization": "TOKEN"}),
        http_mcp(oauth={"client_secret_env": "APP_SECRET"}),
        http_mcp(oauth={"callback_port": 80}),
        http_mcp(oauth={"unknown": True}),
    ],
)
def test_invalid_oauth_configuration_is_rejected(bad):
    with pytest.raises(ValidationError):
        CapabilitySettings.model_validate({"mcp": [bad]})


def test_mcp_namespaces_are_unique_and_default_collections_are_isolated():
    with pytest.raises(ValidationError, match="namespaces must be unique"):
        CapabilitySettings.model_validate({"mcp": [mcp(), mcp()]})
    first, second = CapabilitySettings(), CapabilitySettings()
    first.secrets.inject["KEY"] = "ref"
    first.system2_fallbacks.append("vllm/one")
    assert not second.secrets.inject and not second.system2_fallbacks


def test_distinct_namespaces_cannot_produce_the_same_exported_tool_name():
    with pytest.raises(ValidationError, match="collide across connections"):
        CapabilitySettings.model_validate(
            {"mcp": [mcp(namespace="a", remote_tools={"_b__x": []}), mcp(namespace="a_", remote_tools={"b__x": []})]}
        )


def test_the_old_executor_table_says_what_it_is_now_called():
    with pytest.raises(ValidationError, match=r"\[capabilities.executor\] is now \[capabilities.runner\]"):
        CapabilitySettings.model_validate({"executor": {"backend": "docker"}})
