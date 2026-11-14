"""Model references, providers and roles.

A model is named `provider/model`, e.g. `openrouter/qwen/qwen3.8-27b` or `ollama/qwen3:30b`. The
provider part picks a base URL and the environment variable holding its key; everything after the
first slash is sent to the provider unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, SecretStr

# System 2 is the agent model that reasons and calls tools; System 1 answers fast typed judgments
# (decisions.py). Both decide things; they differ in speed and in what they are asked.
Role = Literal["system2", "system1", "escalation"]


class ConfigError(Exception):
    """A configuration the user has to fix; the message says how."""


Api = Literal["openai", "anthropic", "responses"]


class ProviderSpec(BaseModel):
    base_url: str
    api_key_env: str | None = None
    base_url_env: str | None = None
    api: Api = "openai"
    # The assistant message field replayed reasoning goes back in; a provider rejects a field it does not know.
    reasoning_field: str = "reasoning_content"
    # A header carrying the session id, for a provider whose prompt cache is per replica: it routes a session's
    # requests to the replica that holds its prefix.
    session_header: str | None = None


# Every provider speaks OpenAI chat completions except Anthropic, which uses its native Messages API (its
# OpenAI compatibility layer has no prompt caching), and OpenAI itself, which uses the Responses API (its chat
# completions refuse function tools for GPT-5.6 and later).
PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        api="responses",
    ),
    "openrouter": ProviderSpec(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        base_url_env="OPENROUTER_BASE_URL",
    ),
    "fireworks": ProviderSpec(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        base_url_env="FIREWORKS_BASE_URL",
        # Fireworks caches prompts per replica and routes on this header (docs, "Prompt caching").
        session_header="x-session-affinity",
    ),
    "together": ProviderSpec(
        base_url="https://api.together.xyz/v1", api_key_env="TOGETHER_API_KEY", base_url_env="TOGETHER_BASE_URL"
    ),
    "deepinfra": ProviderSpec(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_API_KEY",
        base_url_env="DEEPINFRA_BASE_URL",
    ),
    "groq": ProviderSpec(base_url="https://api.groq.com/openai/v1", api_key_env="GROQ_API_KEY"),
    "cerebras": ProviderSpec(
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        base_url_env="CEREBRAS_BASE_URL",
        reasoning_field="reasoning",
    ),
    "anthropic": ProviderSpec(
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        base_url_env="ANTHROPIC_BASE_URL",
        api="anthropic",
    ),
    "ollama": ProviderSpec(base_url="http://localhost:11434/v1", base_url_env="OLLAMA_BASE_URL"),
    "llamacpp": ProviderSpec(base_url="http://localhost:8080/v1", base_url_env="LLAMACPP_BASE_URL"),
    "lmstudio": ProviderSpec(base_url="http://localhost:1234/v1", base_url_env="LMSTUDIO_BASE_URL"),
    "vllm": ProviderSpec(base_url="http://localhost:8000/v1", base_url_env="VLLM_BASE_URL"),
}


class Endpoint(BaseModel):
    provider: str
    model: str
    base_url: str
    api_key: SecretStr | None = None
    api: Api = "openai"
    reasoning_field: str = "reasoning_content"
    session_header: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.model}"


def resolve_endpoint(
    ref: str, env: Mapping[str, str], extra_providers: Mapping[str, ProviderSpec] | None = None
) -> Endpoint:
    """Turn `provider/model` into an endpoint, reading the key and any base-URL override from `env`."""
    providers = {**PROVIDERS, **(extra_providers or {})}
    provider, sep, model = ref.partition("/")
    if not sep or not model:
        raise ConfigError(f"model {ref!r} must be written as provider/model, e.g. openrouter/qwen/qwen3.8-27b")
    spec = providers.get(provider)
    if spec is None:
        known = ", ".join(sorted(providers))
        raise ConfigError(f"unknown provider {provider!r} in {ref!r}; known providers: {known}")

    base_url = (env.get(spec.base_url_env) if spec.base_url_env else None) or spec.base_url
    key = env.get(spec.api_key_env) if spec.api_key_env else None
    if spec.api_key_env and not key:
        raise ConfigError(f"{provider} needs an API key in ${spec.api_key_env}")
    return Endpoint(
        provider=provider,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=SecretStr(key) if key else None,
        api=spec.api,
        reasoning_field=spec.reasoning_field,
        session_header=spec.session_header,
    )
