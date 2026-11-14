"""Context-length lookup and the working window.

The advertised window comes from the first source that knows it: the profile, the provider's
`/models` listing, llama.cpp's `/props`, then a conservative default. The working window is the
smallest of the advertised window, the measured effective window and the user's cap. The same
listing says, for some models, how many tokens a reply may have (Anthropic's `max_tokens`,
OpenRouter's `top_provider.max_completion_tokens`); that bounds the profile's output cap, so a 1M
window never asks for more output than the model allows. It also says, for some, whether the model
takes images (Anthropic's `capabilities.image_input`, OpenRouter's `architecture.input_modalities`,
Fireworks' `supports_image_input`); a model that does sees them, whatever its profile says.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx

from intern.events import WindowInfo
from intern.models.profile import Profile
from intern.models.registry import Endpoint

DEFAULT_WINDOW = 32_768
LOOKUP_TIMEOUT_S = 5.0
ANTHROPIC_API_VERSION = "2023-06-01"
_CONTEXT_KEYS = ("context_length", "max_model_len", "context_window", "max_context_length", "max_input_tokens")
_OUTPUT_KEYS = ("max_tokens", "max_output_tokens", "max_completion_tokens")

WindowSource = Literal["profile", "provider", "llama.cpp", "default"]


@dataclass(frozen=True)
class ModelLimits:
    """What a model is advertised to take and give: its window, where that came from, and its output cap."""

    window: int
    source: WindowSource
    max_output: int | None = None
    """The most output tokens one reply may have, when the provider's listing says."""
    vision: bool | None = None
    """Whether the model takes images, when the provider's listing says."""


def _output_cap(item: dict[str, Any]) -> int | None:
    for key in _OUTPUT_KEYS:
        value = item.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    top = item.get("top_provider")
    if isinstance(top, dict):
        return _output_cap(top)  # pyright: ignore[reportUnknownArgumentType]
    return None


def context_length(item: dict[str, Any]) -> int | None:
    """The window a `/models` listing entry advertises, when it says."""
    for key in _CONTEXT_KEYS:
        value = item.get(key)
        if isinstance(value, int) and value > 0:
            return value
    top = item.get("top_provider")
    if isinstance(top, dict):
        return context_length(top)  # pyright: ignore[reportUnknownArgumentType]
    return None


def takes_images(item: dict[str, Any]) -> bool | None:
    """Whether a `/models` listing entry says the model takes images, when it says."""
    capabilities, architecture = item.get("capabilities"), item.get("architecture")
    # Anthropic's capabilities, OpenRouter's architecture, Fireworks' flag.
    if isinstance(capabilities, dict) and isinstance(image_input := capabilities.get("image_input"), dict):
        return image_input.get("supported") is True  # pyright: ignore[reportUnknownMemberType]
    if isinstance(architecture, dict) and isinstance(modalities := architecture.get("input_modalities"), list):
        return "image" in modalities
    supported = item.get("supports_image_input")
    return supported if isinstance(supported, bool) else None


def auth_headers(endpoint: Endpoint) -> dict[str, str]:
    """The headers a provider's model listing needs: Anthropic's own, or a bearer token."""
    if endpoint.api_key and endpoint.api == "anthropic":
        return {"x-api-key": endpoint.api_key.get_secret_value(), "anthropic-version": ANTHROPIC_API_VERSION}
    if endpoint.api_key:
        return {"Authorization": f"Bearer {endpoint.api_key.get_secret_value()}"}
    return {}


Found = tuple[int | None, int | None, bool | None]
"""A lookup's answer: the window, the output cap and whether the model takes images, each None when the source
does not say."""


async def _from_models_listing(client: httpx.AsyncClient, endpoint: Endpoint) -> Found:
    if endpoint.api == "anthropic":
        # Anthropic's Models API describes one model at a time: `max_input_tokens` is its window, `max_tokens`
        # its output cap.
        response = await client.get(f"{endpoint.base_url}/models/{endpoint.model}")
        response.raise_for_status()
        item = response.json()
        return context_length(item), _output_cap(item), takes_images(item)
    response = await client.get(f"{endpoint.base_url}/models")
    response.raise_for_status()
    for item in response.json().get("data") or []:
        if isinstance(item, dict) and item.get("id") == endpoint.model:
            return context_length(item), _output_cap(item), takes_images(item)  # pyright: ignore[reportUnknownArgumentType]
    return None, None, None


async def _from_llamacpp_props(client: httpx.AsyncClient, endpoint: Endpoint) -> Found:
    root = endpoint.base_url.removesuffix("/v1")
    response = await client.get(f"{root}/props")
    response.raise_for_status()
    body = response.json()
    settings = body.get("default_generation_settings") or {}
    n_ctx = settings.get("n_ctx") or body.get("n_ctx")
    return (n_ctx if isinstance(n_ctx, int) and n_ctx > 0 else None), None, None


async def advertised_limits(
    endpoint: Endpoint, profile: Profile, transport: httpx.AsyncBaseTransport | None = None
) -> ModelLimits:
    """The window from the first source that knows it, with the output cap that source gave, if any. A profile
    that sets the window is trusted for both and asks no provider."""
    if profile.window:
        return ModelLimits(profile.window, "profile")
    async with httpx.AsyncClient(
        timeout=LOOKUP_TIMEOUT_S, headers=auth_headers(endpoint), transport=transport
    ) as client:
        lookups: list[tuple[WindowSource, Any]] = [
            ("provider", _from_models_listing),
            ("llama.cpp", _from_llamacpp_props),
        ]
        sees: bool | None = None  # a listing may say what the model takes without its window (Fireworks')
        for source, lookup in lookups:
            try:
                window, max_output, vision = await lookup(client, endpoint)
            except (httpx.HTTPError, ValueError):
                continue
            sees = vision if sees is None else sees
            if window:
                return ModelLimits(window, source, max_output, sees)
    return ModelLimits(DEFAULT_WINDOW, "default", vision=sees)


async def advertised_window(
    endpoint: Endpoint, profile: Profile, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[int, WindowSource]:
    limits = await advertised_limits(endpoint, profile, transport)
    return limits.window, limits.source


def bound_output(profile: Profile, max_output: int | None) -> None:
    """Lower the profile's output cap to what the provider allows; a profile that already asks for less keeps
    its own. Without this a 1M-window model gets `output_reserve` of 1M as max_tokens and is refused."""
    if max_output and (profile.max_output_tokens is None or profile.max_output_tokens > max_output):
        profile.max_output_tokens = max_output


def working_window(advertised: int, source: WindowSource, profile: Profile, user_cap: int | None) -> WindowInfo:
    working = min(limit for limit in (advertised, profile.effective_window, user_cap) if limit)
    return WindowInfo(
        advertised=advertised,
        advertised_source=source,
        effective=profile.effective_window,
        user_cap=user_cap,
        working=working,
    )


def recapped(window: WindowInfo, profile: Profile, user_cap: int | None) -> WindowInfo:
    """`window` under another user cap. The advertised window is kept, so no provider is asked again."""
    return working_window(window.advertised, window.advertised_source, profile, user_cap)
