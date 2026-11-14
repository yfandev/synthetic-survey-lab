"""The reasoning efforts a binding can be set to: its provider's run when the API takes one for every model, the
common three on the profile's word, none otherwise. What the ACP and the TUI offer as thought levels."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intern.models.profile import COMMON_EFFORTS, EFFORT_LEVELS, Profile, effort_choices, load_profile
from intern.providers.anthropic import AnthropicProvider
from intern.providers.fake import FakeProvider
from intern.providers.openai_compat import OpenAICompatProvider
from intern.providers.openai_responses import OpenAIResponsesProvider

from .support import binding


class Ladder(FakeProvider):
    """A provider whose API takes the whole run for every model, as the Anthropic provider does."""

    effort_levels = EFFORT_LEVELS


def test_a_binding_takes_an_effort_on_its_provider_s_word_or_its_profile_s():
    plain = binding(FakeProvider([]))
    assert plain.effort_levels == () and not plain.takes_effort
    named = binding(FakeProvider([]), profile=Profile(reasoning_effort="medium"))
    assert named.effort_levels == COMMON_EFFORTS == ("low", "medium", "high") and named.takes_effort
    assert binding(Ladder([])).effort_levels == EFFORT_LEVELS  # the provider's run, profile or not
    assert binding(Ladder([]), profile=Profile(reasoning_effort="low")).effort_levels == EFFORT_LEVELS
    # Levels a profile declares hold without an effort, so `default` (none sent) comes first and stays offered.
    declared = binding(FakeProvider([]), profile=Profile(effort_levels=["max", "low"]))
    assert declared.effort_levels == ("low", "max") and declared.effort_choices == ("default", "low", "max")
    assert named.effort_choices == COMMON_EFFORTS  # a named effort alone is always sent
    with pytest.raises(ValidationError, match="not off"):
        Profile(effort_levels=["low", "off"])


def test_deepseek_v4_flash_on_fireworks_offers_every_level_after_default():
    for model in (
        "models/deepseek-v4p1-flash",
        "routers/deepseek-v4p1-flash-1017-ultrafast",
        "routers/spot-deepseek-v4-flash",
    ):
        profile = load_profile(f"fireworks/accounts/fireworks/{model}")
        assert effort_choices((), profile) == ("default", *EFFORT_LEVELS) and profile.reasoning_effort is None
    assert effort_choices((), load_profile("openrouter/deepseek/deepseek-v4-flash")) == ()


def test_the_native_apis_take_a_run_for_every_model_and_compatible_endpoints_go_by_the_profile():
    assert AnthropicProvider.effort_levels == ("low", "medium", "high", "xhigh", "max")
    assert OpenAIResponsesProvider.effort_levels == ("low", "medium", "high", "xhigh")
    assert not hasattr(OpenAICompatProvider, "effort_levels")  # some reject the field; a profile says when it is taken
    assert EFFORT_LEVELS[:3] == COMMON_EFFORTS and len(set(EFFORT_LEVELS)) == 5
