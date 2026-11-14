"""Explicit provider fallback and shared, observable invocation budgets."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intern.messages import AssistantMessage, Usage
from intern.providers.base import (
    Completed,
    Provider,
    ProviderError,
    ProviderRequest,
    StreamItem,
)


class TargetConfig(BaseModel):
    ref: str = Field(min_length=1)
    wire_model: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)


@dataclass
class FallbackTarget:
    config: TargetConfig
    provider: Provider


class ProviderAttempt(BaseModel):
    model: str
    status: str
    http_status: int | None = None
    usage_unknown: bool = False


class FallbackProvider:
    """Buffer a complete reply before emission, so failed partial streams never mix."""

    def __init__(self, targets: list[FallbackTarget]) -> None:
        if not 1 <= len(targets) <= 8:
            raise ValueError("explicit fallback chain requires 1–8 configured targets")
        self.targets = targets
        self.attempts: list[ProviderAttempt] = []

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamItem]:
        for index, target in enumerate(self.targets):
            config = target.config
            messages = [
                message.model_copy(update={"reasoning_blocks": [], "reasoning": None})
                if isinstance(message, AssistantMessage) and message.model != config.ref
                else message
                for message in request.messages
            ]
            prepared = request.model_copy(
                update={
                    "model": config.wire_model,
                    "messages": messages,
                    "max_output_tokens": min(request.max_output_tokens, config.max_output_tokens),
                }
            )
            buffered: list[StreamItem] = []
            size = 0
            completed = False
            try:
                async for item in target.provider.stream(prepared):
                    if completed:
                        raise ProviderError("provider emitted data after its completed response")
                    size += len(item.model_dump_json())
                    if size > 32 * 1024 * 1024:
                        raise ProviderError("fallback reply exceeded the bounded response buffer")
                    if isinstance(item, Completed):
                        completed = True
                        item = item.model_copy(
                            update={"message": item.message.model_copy(update={"model": config.ref})}
                        )
                    buffered.append(item)
                if not completed:
                    raise ProviderError("provider stream ended without a completed response", retryable=True)
            except ProviderError as error:
                self.attempts.append(
                    ProviderAttempt(
                        model=config.ref, status="failed", http_status=error.status, usage_unknown=not completed
                    )
                )
                if not error.retryable or index == len(self.targets) - 1:
                    # Do not leak provider exception bodies, which can contain request credentials.
                    raise ProviderError(
                        "configured provider chain failed", status=error.status, retryable=False
                    ) from None
            else:
                self.attempts.append(ProviderAttempt(model=config.ref, status="completed"))
                for item in buffered:
                    yield item
                return


class BudgetPolicy(BaseModel):
    max_calls: int = Field(default=100, ge=1)
    max_completed_output_tokens: int | None = Field(default=None, ge=1)
    max_observed_cost_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    deadline_unix: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class BudgetState(BaseModel):
    calls: int = Field(default=0, ge=0)
    completed_calls: int = Field(default=0, ge=0)
    unknown_usage_calls: int = Field(default=0, ge=0)
    unknown_output_calls: int = Field(default=0, ge=0)
    usage: Usage = Field(default_factory=Usage)

    @model_validator(mode="after")
    def valid_settled_state(self) -> BudgetState:
        if self.completed_calls + self.unknown_usage_calls != self.calls or self.unknown_output_calls > self.calls:
            raise ValueError("budget snapshots must describe settled calls")
        if self.usage.cost_usd < 0 or not math.isfinite(self.usage.cost_usd):
            raise ValueError("observed budget cost must be finite and nonnegative")
        if any(v < 0 for k, v in self.usage.model_dump().items() if k != "cost_usd"):
            raise ValueError("observed token counts must be nonnegative")
        return self


class BudgetReservation(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    max_output_tokens: int | None = Field(default=None, gt=0)


class RunBudget:
    def __init__(
        self, policy: BudgetPolicy, state: BudgetState | None = None, *, clock: Callable[[], float] = time.time
    ) -> None:
        self.policy = BudgetPolicy.model_validate(policy.model_dump())
        self.state = BudgetState.model_validate((state or BudgetState()).model_dump())
        self.clock = clock
        self._lock = asyncio.Lock()
        self._reserved = 0
        self._reservations: dict[str, BudgetReservation] = {}

    async def reserve(self, requested_output: int | None) -> BudgetReservation:
        async with self._lock:
            policy, state = self.policy, self.state
            if requested_output is not None and requested_output < 1:
                raise ProviderError("invalid requested output budget")
            if requested_output is None and policy.max_completed_output_tokens is not None:
                raise ProviderError("decision backends do not report output tokens; use a separate output-token budget")
            if policy.deadline_unix is not None and self.clock() >= policy.deadline_unix:
                raise ProviderError("run deadline reached")
            if state.calls >= policy.max_calls:
                raise ProviderError("run provider-call budget reached")
            if policy.max_observed_cost_usd is not None and state.usage.cost_usd >= policy.max_observed_cost_usd:
                raise ProviderError("observed provider cost budget reached")
            remaining = (
                policy.max_completed_output_tokens - state.usage.output_tokens - self._reserved
                if policy.max_completed_output_tokens is not None
                else requested_output or 0
            )
            if requested_output is not None and remaining < 1:
                raise ProviderError("run output-token budget reached")
            cap = min(requested_output, remaining) if requested_output is not None else None
            reservation = BudgetReservation(id=uuid.uuid4().hex, max_output_tokens=cap)
            self._reserved += cap or 0
            self._reservations[reservation.id] = reservation
            state.calls += 1
            return reservation

    async def finish(self, reservation: BudgetReservation, usage: Usage | None) -> None:
        async with self._lock:
            if self._reservations.get(reservation.id) != reservation:
                raise ValueError("budget reservation is unknown, changed or already settled")
            del self._reservations[reservation.id]
            self._reserved -= reservation.max_output_tokens or 0
            if reservation.max_output_tokens is None:
                self.state.unknown_output_calls += 1
            if usage is not None and (
                not math.isfinite(usage.cost_usd)
                or usage.cost_usd < 0
                or any(v < 0 for k, v in usage.model_dump().items() if k != "cost_usd")
            ):
                usage = None
            if usage is None:
                self.state.unknown_usage_calls += 1
            else:
                self.state.completed_calls += 1
                self.state.usage += usage

    async def snapshot(self) -> BudgetState:
        async with self._lock:
            if self._reservations:
                raise ValueError("snapshot budget only at a settled call boundary")
            return self.state.model_copy(deep=True)


class BudgetProvider:
    def __init__(self, provider: Provider, budget: RunBudget) -> None:
        self.provider, self.budget = provider, budget

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamItem]:
        reserved = await self.budget.reserve(request.max_output_tokens)
        usage: Usage | None = None
        timer: asyncio.Timeout | None = None
        try:
            assert reserved.max_output_tokens is not None
            prepared = request.model_copy(update={"max_output_tokens": reserved.max_output_tokens})
            deadline = self.budget.policy.deadline_unix
            timeout = None if deadline is None else max(0, deadline - self.budget.clock())
            timer = asyncio.timeout(timeout)
            async with timer:
                async for item in self.provider.stream(prepared):
                    if isinstance(item, Completed):
                        usage = item.message.usage
                    yield item
        except TimeoutError:
            if timer is not None and timer.expired():
                raise ProviderError("run deadline reached during provider request") from None
            # Preserve transport failover through the typed Provider protocol.
            raise ProviderError("provider request timed out", retryable=True) from None
        finally:
            await self.budget.finish(reserved, usage)
