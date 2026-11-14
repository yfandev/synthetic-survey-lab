"""System 1 adapters used inside DecisionClient, sharing budgets with System 2.

These adapters never make model HTTP requests themselves. Existing DecisionBackend
implementations retain transport/authentication; DecisionClient retains its cache,
retries and decision events.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Sequence
from typing import Literal

import httpx
from pydantic import BaseModel

from intern.decisions import Answers, DecisionBackend, Question
from intern.messages import Usage
from intern.providers.base import ProviderError
from intern.providers.failover import RunBudget


class DecisionAttempt(BaseModel):
    backend: str
    status: Literal["completed", "failed"]
    error_type: str | None = None


class FallbackDecisionBackend:
    """Explicit transport failover; answer suitability still belongs to DecisionClient.

    Native DecisionsApiBackend already supports a model list. This adapter is useful
    for different endpoints/backends or per-target budgets. Its default recoverable
    exceptions include the native API's RuntimeError normalization of transport and
    response-validation failures. Direct invalid typed results and budget refusals
    stop the chain; cancellation always propagates.
    """

    def __init__(
        self,
        backends: Sequence[DecisionBackend],
        *,
        recoverable: tuple[type[Exception], ...] = (httpx.HTTPError, OSError, RuntimeError),
    ) -> None:
        if not 1 <= len(backends) <= 8 or not recoverable:
            raise ValueError("use 1–8 explicit decision backends and recoverable exception types")
        self.backends = tuple(backends)
        self.recoverable = recoverable
        self.name = "decision-fallback:" + ",".join(backend.name for backend in backends)
        self._attempts: deque[DecisionAttempt] = deque(maxlen=128)

    @property
    def attempts(self) -> list[DecisionAttempt]:
        return [attempt.model_copy(deep=True) for attempt in self._attempts]

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        for index, backend in enumerate(self.backends):
            try:
                isolated = {name: question.model_copy(deep=True) for name, question in questions.items()}
                raw = await backend.decide(state, isolated)
            except Exception as error:
                self._attempts.append(
                    DecisionAttempt(backend=backend.name, status="failed", error_type=type(error).__name__)
                )
                if not isinstance(error, self.recoverable) or index == len(self.backends) - 1:
                    # Native backend exceptions can contain credential-bearing URLs.
                    raise ValueError("configured decision backend chain failed") from None
            else:
                try:
                    result = Answers.model_validate(raw)
                    if not math.isfinite(result.cost_usd) or result.cost_usd < 0:
                        raise ValueError("decision backend returned an invalid observed cost")
                except Exception as error:
                    self._attempts.append(
                        DecisionAttempt(backend=backend.name, status="failed", error_type=type(error).__name__)
                    )
                    # Returned data must be valid even if the caller configured
                    # broad transport recovery. Never mask bad costs by retrying.
                    raise ValueError("configured decision backend returned invalid data") from None
                self._attempts.append(DecisionAttempt(backend=backend.name, status="completed"))
                return result.model_copy(deep=True)
        raise AssertionError("nonempty decision chain did not return")


class BudgetDecisionBackend:
    """Account for logical decision dispatches, without inventing missing token usage.

    Share one RunBudget with BudgetProvider. Because Answers reports only cost,
    output-token budgets are refused before dispatch. Existing backend-internal HTTP
    retries/batches are not separate logical calls; wrap single-target backends when
    every fallback target should reserve a call. Costs remain observed thresholds.
    """

    def __init__(self, backend: DecisionBackend, budget: RunBudget) -> None:
        self.backend, self.budget = backend, budget
        self.name = "budget:" + backend.name

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        reservation = await self.budget.reserve(None)
        usage: Usage | None = None
        timer: asyncio.Timeout | None = None
        try:
            deadline = self.budget.policy.deadline_unix
            timeout = None if deadline is None else max(0, deadline - self.budget.clock())
            timer = asyncio.timeout(timeout)
            async with timer:
                result = Answers.model_validate(await self.backend.decide(state, questions))
                if not math.isfinite(result.cost_usd) or result.cost_usd < 0:
                    raise ValueError("decision backend returned an invalid observed cost")
                usage = Usage(cost_usd=result.cost_usd)
                return result.model_copy(deep=True)
        except TimeoutError:
            if timer is not None and timer.expired():
                raise ProviderError("run deadline reached during decision request") from None
            raise TimeoutError("decision backend transport timed out") from None
        finally:
            await self.budget.finish(reservation, usage)
