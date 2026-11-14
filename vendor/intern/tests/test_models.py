"""Model references, profiles and the working window."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from intern import config
from intern.config import Bootstrap, Options
from intern.messages import Usage
from intern.models.prices import priced, prices
from intern.models.profile import Profile, load_profile, profile_candidates
from intern.models.registry import ConfigError, resolve_endpoint
from intern.models.window import (
    DEFAULT_WINDOW,
    ModelLimits,
    advertised_limits,
    advertised_window,
    bound_output,
    working_window,
)


def test_endpoint_reads_key_and_base_url_override_from_env():
    endpoint = resolve_endpoint(
        "fireworks/accounts/fireworks/models/x", {"FIREWORKS_API_KEY": "k", "FIREWORKS_BASE_URL": "http://gw/v1/"}
    )
    assert endpoint.model == "accounts/fireworks/models/x"
    assert endpoint.base_url == "http://gw/v1"
    assert endpoint.api_key is not None and endpoint.api_key.get_secret_value() == "k"


def test_local_providers_need_no_key():
    assert resolve_endpoint("ollama/qwen3:30b", {}).api_key is None


@pytest.mark.parametrize(
    ("ref", "message"),
    [("qwen3", "provider/model"), ("nowhere/model", "unknown provider"), ("openrouter/x", "OPENROUTER_API_KEY")],
)
def test_bad_references_say_how_to_fix_them(ref: str, message: str):
    with pytest.raises(ConfigError, match=message):
        resolve_endpoint(ref, {})


def test_the_most_specific_profile_wins_and_merges_over_the_default():
    profile = load_profile("fireworks/accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b")
    assert profile.window == 262_144
    assert profile.thresholds.compact_at == Profile().thresholds.compact_at
    assert load_profile("openrouter/unknown/model").window is None
    deployment = load_profile("fireworks/accounts/example/deployments/qwen3p8-27b")
    assert (deployment.window, deployment.max_output_tokens) == (131_072, 32_768)
    assert (deployment.reasoning_replay, deployment.reasoning_effort) == (True, "medium")
    assert deployment.vision and load_profile("openrouter/qwen/qwen3.8-27b").vision  # read shows it images


def test_user_profiles_and_overrides_apply_in_order(tmp_path: Path):
    folder = tmp_path / "profiles"
    folder.mkdir()
    (folder / "mine.toml").write_text('match = ["openrouter/acme/*"]\nwindow = 65536\n[thresholds]\ncompact_at = 0.6\n')
    override = tmp_path / "run.toml"
    override.write_text("max_turns = 7\n")
    profile = load_profile("openrouter/acme/small", search_dirs=[folder], override=override)
    assert (profile.window, profile.thresholds.compact_at, profile.max_turns) == (65_536, 0.6, 7)


def test_working_window_is_the_smallest_known_limit():
    profile = Profile(effective_window=48_000)
    assert working_window(131_072, "provider", profile, None).working == 48_000
    assert working_window(131_072, "provider", profile, 32_768).working == 32_768
    assert working_window(131_072, "provider", Profile(), None).working == 131_072
    assert working_window(4_096, "llama.cpp", Profile(), None).working == 4_096


def _transport(routes: dict[str, httpx.Response]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return routes.get(request.url.path, httpx.Response(404))

    return httpx.MockTransport(handler)


async def test_window_lookup_ladder():
    endpoint = resolve_endpoint("vllm/m", {})
    listing = httpx.Response(200, json={"data": [{"id": "m", "max_model_len": 40_960}]})
    assert await advertised_window(endpoint, Profile(), _transport({"/v1/models": listing})) == (40_960, "provider")

    props = httpx.Response(200, json={"default_generation_settings": {"n_ctx": 16_384}})
    assert await advertised_window(endpoint, Profile(), _transport({"/props": props})) == (16_384, "llama.cpp")

    assert await advertised_window(endpoint, Profile(), _transport({})) == (DEFAULT_WINDOW, "default")
    assert await advertised_window(endpoint, Profile(window=99_000), _transport({})) == (99_000, "profile")


async def test_the_provider_s_output_cap_is_read_with_the_window_and_bounds_the_profile():
    # Anthropic's Models API describes one model: max_input_tokens is the window, max_tokens the output cap.
    anthropic = resolve_endpoint("anthropic/claude-opus-5-9", {"ANTHROPIC_API_KEY": "k"}, {})
    described = httpx.Response(
        200, json={"id": "claude-opus-5-9", "max_input_tokens": 1_000_000, "max_tokens": 128_000}
    )
    limits = await advertised_limits(anthropic, Profile(), _transport({"/v1/models/claude-opus-5-9": described}))
    assert limits == ModelLimits(1_000_000, "provider", 128_000)
    # OpenRouter's listing puts the output cap under top_provider.
    openrouter = resolve_endpoint("openrouter/qwen/qwen3.8-27b", {"OPENROUTER_API_KEY": "k"}, {})
    listing = httpx.Response(
        200,
        json={
            "data": [
                {"id": "qwen/qwen3.8-27b", "context_length": 131_072, "top_provider": {"max_completion_tokens": 32_768}}
            ]
        },
    )
    limits = await advertised_limits(openrouter, Profile(), _transport({"/api/v1/models": listing}))
    assert limits == ModelLimits(131_072, "provider", 32_768)
    # A listing without a cap, or a profile window, leaves the cap unknown.
    bare = httpx.Response(200, json={"data": [{"id": "qwen/qwen3.8-27b", "context_length": 131_072}]})
    assert (await advertised_limits(openrouter, Profile(), _transport({"/api/v1/models": bare}))).max_output is None
    assert (await advertised_limits(openrouter, Profile(window=99_000), _transport({}))).max_output is None
    # The profile's cap comes down to the provider's, never up; 15% of 1M would have been refused.
    profile = Profile()
    bound_output(profile, 128_000)
    assert profile.max_output_tokens == 128_000 and profile.output_cap(1_000_000) == 128_000
    lower = Profile(max_output_tokens=32_768)
    bound_output(lower, 128_000)
    assert lower.max_output_tokens == 32_768
    bound_output(lower, None)
    assert lower.max_output_tokens == 32_768


def test_a_reply_is_priced_by_its_fresh_cached_and_written_tokens_and_a_long_prompt_at_the_long_rates():
    # Opus 5.5: $4 input, $5 cache write, $0.20 cache read, $20 output per million tokens.
    usage = Usage(input_tokens=190_377, cache_read_tokens=189_494, cache_write_tokens=881, output_tokens=349)
    cost = (2 * 4 + 189_494 * 0.20 + 881 * 5 + 349 * 20) / 1e6
    assert priced(usage, "anthropic/claude-opus-5-5").cost_usd == pytest.approx(cost)
    # Haiku 5.5 charges a prompt over 100K tokens its higher rates, on every token of it.
    short, long = Usage(input_tokens=100_000, output_tokens=1000), Usage(input_tokens=100_001, output_tokens=1000)
    assert priced(short, "anthropic/claude-haiku-5-5").cost_usd == pytest.approx((100_000 * 0.10 + 1000 * 0.50) / 1e6)
    assert priced(long, "anthropic/claude-haiku-5-5").cost_usd == pytest.approx((100_001 * 0.50 + 1000 * 2.50) / 1e6)
    # A model with no cache prices prices its cached tokens as input; a dated snapshot takes its model's price.
    cached = Usage(input_tokens=1000, cache_read_tokens=1000, output_tokens=0)
    assert priced(cached, "openai/gpt-5.5-pro-2026-04-23").cost_usd == pytest.approx(1000 * 30 / 1e6)
    assert priced(cached, "anthropic/claude-unknown").cost_usd == 0  # no price: unknown, not invented


def test_every_claude_and_gpt_model_with_a_packaged_profile_has_a_price():
    names = [name for name, _ in profile_candidates()]
    assert {"claude-opus-5-5.toml", "claude-sonnet-5-5.toml", "claude-haiku-5-5.toml"} <= set(names)
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
        assert f"anthropic/{model}" in prices()
    assert all(ref.split("/")[0] in ("anthropic", "openai", "fireworks") for ref in prices())  # the rest report theirs


def test_the_claude_5_5_profiles_cap_the_reply_at_what_the_api_allows():
    for ref in ("anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5", "anthropic/claude-haiku-5-5"):
        profile = load_profile(ref)
        assert (profile.window, profile.max_output_tokens, profile.vision) == (1_000_000, 128_000, True), ref
        assert profile.output_cap(1_000_000) == 128_000


async def test_a_listing_that_says_the_model_takes_images_is_read_with_the_window():
    anthropic = resolve_endpoint("anthropic/claude-opus-5-9", {"ANTHROPIC_API_KEY": "k"}, {})
    described = httpx.Response(
        200,
        json={
            "id": "claude-opus-5-9",
            "max_input_tokens": 1_000_000,
            "capabilities": {"image_input": {"supported": True}},
        },
    )
    limits = await advertised_limits(anthropic, Profile(), _transport({"/v1/models/claude-opus-5-9": described}))
    assert limits.vision is True
    openrouter = resolve_endpoint("openrouter/openai/gpt-6-astra", {"OPENROUTER_API_KEY": "k"}, {})
    for modalities, sees in ((["file", "image", "text"], True), (["text"], False)):
        entry = {
            "id": "openai/gpt-6-astra",
            "context_length": 400_000,
            "architecture": {"input_modalities": modalities},
        }
        listing = httpx.Response(200, json={"data": [entry]})
        assert (await advertised_limits(openrouter, Profile(), _transport({"/api/v1/models": listing}))).vision is sees
    fireworks = resolve_endpoint("fireworks/accounts/fireworks/models/qwen3p8-max", {"FIREWORKS_API_KEY": "k"}, {})
    entry = {"id": "accounts/fireworks/models/qwen3p8-max", "context_length": 262_144, "supports_image_input": True}
    listing = httpx.Response(200, json={"data": [entry]})
    limits = await advertised_limits(fireworks, Profile(), _transport({"/inference/v1/models": listing}))
    assert limits.vision is True
    # Fireworks' listing gives no window for some models: what it says about images still counts.
    entry = {"id": "accounts/fireworks/models/qwen3p8-max", "supports_image_input": True}
    unsized = httpx.Response(200, json={"data": [entry]})
    limits = await advertised_limits(fireworks, Profile(), _transport({"/inference/v1/models": unsized}))
    assert (limits.source, limits.vision) == ("default", True)
    # A listing that does not say leaves it to the profile.
    bare = httpx.Response(200, json={"data": [{"id": "openai/gpt-6-astra", "context_length": 400_000}]})
    assert (await advertised_limits(openrouter, Profile(), _transport({"/api/v1/models": bare}))).vision is None


async def test_a_model_the_provider_says_takes_images_is_bound_to_see_them(tmp_path: Path, monkeypatch):
    bootstrap = Bootstrap(Options(cwd=tmp_path), {"INTERN_HOME": str(tmp_path), "OPENROUTER_API_KEY": "k"})
    for said, sees in ((True, True), (False, False), (None, False)):

        async def limits(endpoint, profile, transport=None, said=said):
            return ModelLimits(400_000, "provider", None, said)

        monkeypatch.setattr(config, "advertised_limits", limits)
        assert (await bootstrap.bind("openrouter/openai/gpt-6-astra")).profile.vision is sees
    # A profile that says the model sees keeps it whatever the listing says: OpenAI's own listing says nothing.
    assert (await bootstrap.bind("openrouter/qwen/qwen3.8-27b")).profile.vision is True
    assert load_profile("openai/gpt-6-astra").vision and load_profile("openai/gpt-5-mini").vision
