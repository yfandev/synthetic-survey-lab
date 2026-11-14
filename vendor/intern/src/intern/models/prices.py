"""What a call cost, where the provider's reply gives only token counts.

Anthropic's and OpenAI's own APIs and Fireworks report tokens, never dollars, so their adapters price each
reply's usage from the published prices in `prices.toml`, as pi does. OpenRouter and the Decisions API report each
call's cost themselves, and a model with no price here keeps a cost of 0.
"""

from __future__ import annotations

import functools
import re
import tomllib
from importlib import resources

from pydantic import BaseModel, Field

from intern.messages import Usage

SNAPSHOT = re.compile(r"-\d{4}-?\d{2}-?\d{2}$")
"""A dated snapshot's suffix: `-20251101`, `-2026-04-23`."""


class Rates(BaseModel):
    """US dollars per million tokens. A cache read or write without a price of its own costs as much as input."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)


class LongRates(Rates):
    above: int = Field(gt=0, description="A prompt over this many tokens, cache reads and writes included.")


class Price(Rates):
    long: LongRates | None = None


@functools.cache
def prices() -> dict[str, Price]:
    """The packaged price table, by `provider/model`."""
    text = resources.files("intern.models").joinpath("prices.toml").read_text()
    return {ref: Price.model_validate(price) for ref, price in tomllib.loads(text).items()}


def priced(usage: Usage, ref: str) -> Usage:
    """`usage` with its cost from `ref`'s published price; as it was when the model has none."""
    price = prices().get(ref) or prices().get(SNAPSHOT.sub("", ref))
    if price is None:
        return usage
    rates: Rates = price.long if price.long is not None and usage.input_tokens > price.long.above else price
    read, written = usage.cache_read_tokens, usage.cache_write_tokens
    fresh = max(usage.input_tokens - read - written, 0)  # input_tokens is the whole prompt, the cache included
    cost = (
        fresh * rates.input
        + read * (rates.input if rates.cache_read is None else rates.cache_read)
        + written * (rates.input if rates.cache_write is None else rates.cache_write)
        + usage.output_tokens * rates.output
    )
    return usage.model_copy(update={"cost_usd": cost / 1_000_000})
