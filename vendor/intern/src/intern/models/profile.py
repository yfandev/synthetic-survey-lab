"""Model profiles: per-model data that switches harness behaviour without code changes.

A profile is a TOML file whose `match` globs are compared against `provider/model`. The packaged
`default.toml` is the base; the most specific matching profile is merged over it, and a profile file
passed by the user is merged last.
"""

from __future__ import annotations

import fnmatch
import tomllib
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, JsonValue, field_validator

PACKAGED_PROFILES = "intern.profiles"
DEFAULT_PROFILE = "default.toml"


class Sampling(BaseModel):
    temperature: float | None = None
    top_p: float | None = None


class Thresholds(BaseModel):
    """Budgets as fractions of the working window, so they hold for a 32k window and a 256k one alike."""

    compact_at: float = Field(default=0.75, gt=0, lt=1, description="Compact history above this share.")
    keep_recent: float = Field(default=0.35, gt=0, lt=1, description="History kept verbatim when compacting.")
    tool_result: float = Field(default=0.15, gt=0, lt=1, description="One tool result's share of the window.")
    output_reserve: float = Field(default=0.15, gt=0, lt=1, description="Room left for the reply.")
    memory: float = Field(default=0.08, gt=0, lt=1, description="The recalled memory block's share of the window.")
    requests: float = Field(
        default=0.08, gt=0, lt=1, description="The user's latest requests, kept as written in the State."
    )


class Compat(BaseModel):
    """Provider quirks, as flags rather than branches in the adapter."""

    stream: bool = True
    stream_usage: bool = True
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    send_tool_choice: bool = True
    parallel_tool_calls: bool | None = None
    think_tags: bool = Field(default=True, description="Route <think>...</think> in content to reasoning.")
    text_tool_calls: bool = Field(default=True, description="Recover <tool_call> JSON a model wrote as text.")
    extra_body: dict[str, JsonValue] = Field(default_factory=dict)


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
"""The reasoning efforts, least to most. A provider takes a run of them (its `effort_levels`); the profile's
`reasoning_effort` holds one, or none for the model's own."""

COMMON_EFFORTS = EFFORT_LEVELS[:3]
"""The run every reasoning API takes: what a profile-named effort offers on a provider that declares none."""


def effort_levels(provider_levels: Sequence[str], profile: Profile) -> tuple[str, ...]:
    """The reasoning efforts a model can be set to, least to most: the provider's run when its API takes one for
    every model it serves (Anthropic's `output_config.effort`, OpenAI's `reasoning.effort`); else the profile's
    `effort_levels`; else the three every reasoning API takes when the profile names an effort, since
    OpenAI-compatible endpoints go only on the profile's word (some reject the field); else none."""
    if provider_levels or profile.effort_levels:
        return tuple(provider_levels or profile.effort_levels)
    return COMMON_EFFORTS if profile.reasoning_effort is not None else ()


DEFAULT_EFFORT = "default"
"""The thought level that sends no effort, so the model uses its own."""


def effort_choices(provider_levels: Sequence[str], profile: Profile) -> tuple[str, ...]:
    """What a thought level can be set to: `default` first when the levels hold without an effort (the API's own
    run, or the profile's `effort_levels`), then the efforts. On a named effort alone an effort is always sent."""
    levels = effort_levels(provider_levels, profile)
    return (DEFAULT_EFFORT, *levels) if provider_levels or profile.effort_levels else levels


class Profile(BaseModel):
    match: list[str] = Field(default_factory=lambda: ["*"])
    name: str | None = Field(default=None, description="The model catalog's name for the concrete ids in `match`.")
    window: int | None = Field(default=None, description="Advertised window; overrides provider lookup.")
    effective_window: int | None = Field(default=None, description="Where quality still holds, as measured.")
    max_output_tokens: int | None = None
    max_tool_result_tokens: int | None = Field(
        default=20_000,
        gt=0,
        description="The most of one tool result the model is shown, whatever the window: `thresholds.tool_result` "
        "of the window up to this. A model reads no more of one output, and pays no less to be sent it again every "
        "turn, because its window is larger. None leaves the share alone.",
    )
    sampling: Sampling = Field(default_factory=Sampling)
    reasoning_effort: str | None = None
    effort_levels: list[str] = Field(
        default_factory=list,
        description="The reasoning efforts the model takes, on an API that does not say (OpenAI-compatible "
        "endpoints). Sessions offer them after `default`, which sends none, so nothing changes until one is picked.",
    )
    reasoning_replay: bool = True
    max_turns: int = 150
    max_continuations: int = 3
    vision: bool = Field(
        default=False,
        description="The model accepts images; read then shows image files. A provider listing that says the model "
        "takes images sets it too.",
    )
    tool_packs: list[str] = Field(default_factory=lambda: ["base"])
    extensions: list[str] = Field(default_factory=list)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    compat: Compat = Field(default_factory=Compat)

    @field_validator("effort_levels")
    @classmethod
    def _known_efforts(cls, levels: list[str]) -> list[str]:
        if unknown := sorted(set(levels) - set(EFFORT_LEVELS)):
            raise ValueError(f"effort_levels takes {', '.join(EFFORT_LEVELS)}; not {', '.join(unknown)}")
        return [level for level in EFFORT_LEVELS if level in levels]

    def output_cap(self, working_window: int) -> int:
        reserve = int(working_window * self.thresholds.output_reserve)
        return min(self.max_output_tokens, reserve) if self.max_output_tokens else reserve


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)  # pyright: ignore[reportUnknownArgumentType]
        else:
            merged[key] = value
    return merged


def _specificity(patterns: list[str], ref: str) -> int:
    """Length of the longest pattern matching `ref`, ignoring wildcards; -1 when none matches."""
    hits = [len(p.replace("*", "")) for p in patterns if fnmatch.fnmatchcase(ref.lower(), p.lower())]
    return max(hits, default=-1)


def _packaged() -> list[tuple[str, dict[str, Any]]]:
    folder = resources.files(PACKAGED_PROFILES)
    return [
        (item.name, tomllib.loads(item.read_text()))
        for item in sorted(folder.iterdir(), key=lambda i: i.name)
        if item.name.endswith(".toml")
    ]


def profile_candidates(search_dirs: list[Path] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """The packaged profiles, then those in `search_dirs`, parsed: what `load_profile` picks from."""
    candidates = _packaged()
    for folder in search_dirs or []:
        if folder.is_dir():
            candidates += [(p.name, tomllib.loads(p.read_text())) for p in sorted(folder.glob("*.toml"))]
    return candidates


def load_profile(
    ref: str,
    *,
    search_dirs: list[Path] | None = None,
    override: Path | None = None,
    candidates: list[tuple[str, dict[str, Any]]] | None = None,
) -> Profile:
    """Resolve the profile for `provider/model`.

    Args:
        ref: The model reference, e.g. `fireworks/accounts/fireworks/models/qwen3p8-27b`.
        search_dirs: Extra folders of profile TOML files; they win ties against packaged ones.
        override: A profile file applied last, whatever its `match` says.
        candidates: `profile_candidates(search_dirs)`, read once by callers resolving many models.
    """
    if candidates is None:
        candidates = profile_candidates(search_dirs)

    base = next(data for name, data in candidates if name == DEFAULT_PROFILE)
    best: dict[str, Any] | None = None
    best_score = 0
    for name, data in candidates:
        if name == DEFAULT_PROFILE:
            continue
        score = _specificity(data.get("match", []), ref)
        if score >= best_score and score >= 0:
            best, best_score = data, score

    merged = _merge(base, best or {})
    if override is not None:
        merged = _merge(merged, tomllib.loads(override.read_text()))
    return Profile.model_validate(merged)
