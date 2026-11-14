import asyncio
import traceback
from collections.abc import AsyncIterator

import pytest

from intern.messages import AssistantMessage, Usage
from intern.providers.base import Completed, ProviderError, ProviderRequest, StreamItem, TextChunk
from intern.providers.failover import (
    BudgetPolicy,
    BudgetProvider,
    FallbackProvider,
    FallbackTarget,
    RunBudget,
    TargetConfig,
)
from intern.providers.fake import FakeProvider, Turn


def target(provider, name):
    return FallbackTarget(TargetConfig(ref=f"fake/{name}", wire_model=name, max_output_tokens=100), provider)


class PartialFailure:
    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamItem]:
        yield TextChunk(text="partial text that must disappear")
        raise ProviderError("connection reset", retryable=True)


async def test_partial_failure_does_not_mix_outputs_and_actual_model_is_recorded(make_harness):
    backup = FakeProvider([Turn(text="complete answer")])
    provider = FallbackProvider([target(PartialFailure(), "primary"), target(backup, "backup")])
    harness = make_harness([])
    harness.deps.system2.provider = provider
    outcome = await harness.run()
    assert outcome.final_text == "complete answer"
    assert "partial text" not in str(harness.recorder.events)
    assert backup.requests[0].model == "backup"
    assert provider.attempts[0].usage_unknown


async def test_nonretryable_failure_does_not_fall_back(make_harness):
    backup = FakeProvider([Turn(text="should not run")])
    provider = FallbackProvider([target(FakeProvider([Turn(error="bad input")]), "a"), target(backup, "b")])
    harness = make_harness([])
    harness.deps.system2.provider = provider
    assert (await harness.run()).reason == "provider_error"
    assert not backup.requests


async def test_terminal_provider_error_traceback_withholds_original_credentials():
    class PrivateFailure:
        async def stream(self, request):
            if False:
                yield
            raise ProviderError("private-provider-url?credential=synthetic-secret", retryable=False, status=401)

    provider = FallbackProvider([target(PrivateFailure(), "private")])
    with pytest.raises(ProviderError) as captured:
        _ = [item async for item in provider.stream(ProviderRequest(model="a", messages=[], max_output_tokens=10))]
    error = captured.value
    formatted = "".join(traceback.format_exception(error))
    assert "synthetic-secret" not in formatted and "private-provider-url" not in formatted
    assert error.status == 401 and error.__cause__ is None
    assert "configured provider chain failed" in formatted


async def test_cross_model_signed_reasoning_is_removed():
    backup = FakeProvider([Turn(text="ok")])
    provider = FallbackProvider([target(backup, "b")])
    request = ProviderRequest(
        model="a",
        max_output_tokens=1000,
        messages=[AssistantMessage(model="fake/a", text="old", reasoning_blocks=[{"signature": "private"}])],
    )
    items = [item async for item in provider.stream(request)]
    assert isinstance(items[-1], Completed) and items[-1].message.model == "fake/b"
    assert isinstance(backup.requests[0].messages[0], AssistantMessage)
    assert not backup.requests[0].messages[0].reasoning_blocks
    assert backup.requests[0].max_output_tokens == 100


async def test_budget_caps_calls_and_output_and_snapshots():
    backend = FakeProvider([Turn(text="first", output_tokens=4), Turn(text="second", output_tokens=2)])
    budget = RunBudget(BudgetPolicy(max_calls=2, max_completed_output_tokens=6))
    provider = BudgetProvider(backend, budget)
    request = ProviderRequest(model="a", max_output_tokens=100, messages=[])
    for _ in range(2):
        assert [item async for item in provider.stream(request)]
    assert [r.max_output_tokens for r in backend.requests] == [6, 2]
    state = await budget.snapshot()
    assert state.calls == 2 and state.usage.output_tokens == 6
    restored = RunBudget(budget.policy, state)
    with pytest.raises(ProviderError, match="budget"):
        await restored.reserve(1)


async def test_unknown_usage_and_deadlines_are_visible():
    budget = RunBudget(BudgetPolicy(max_calls=2, deadline_unix=20), clock=lambda: 10)
    provider = BudgetProvider(PartialFailure(), budget)
    with pytest.raises(ProviderError):
        _ = [item async for item in provider.stream(ProviderRequest(model="a", max_output_tokens=10, messages=[]))]
    assert (await budget.snapshot()).unknown_usage_calls == 1
    expired = RunBudget(BudgetPolicy(deadline_unix=10), clock=lambda: 11)
    with pytest.raises(ProviderError, match="deadline"):
        await expired.reserve(1)


async def test_observed_cost_threshold_stops_later_requests():
    budget = RunBudget(BudgetPolicy(max_observed_cost_usd=0.1))
    reserved = await budget.reserve(5)
    await budget.finish(reserved, Usage(output_tokens=1, cost_usd=0.12))
    with pytest.raises(ProviderError, match="cost"):
        await budget.reserve(1)


async def test_reservations_cannot_be_settled_twice_or_used_by_another_budget():
    budget = RunBudget(BudgetPolicy())
    other = RunBudget(BudgetPolicy())
    reservation = await budget.reserve(5)
    with pytest.raises(ValueError, match="unknown"):
        await other.finish(reservation, Usage())
    await budget.finish(reservation, Usage(output_tokens=2))
    with pytest.raises(ValueError, match="already settled"):
        await budget.finish(reservation, Usage(output_tokens=2))
    assert (await budget.snapshot()).usage.output_tokens == 2


async def test_invalid_provider_usage_is_explicitly_unknown_not_a_negative_or_nan_budget():
    budget = RunBudget(BudgetPolicy())
    reservation = await budget.reserve(5)
    await budget.finish(reservation, Usage(cost_usd=float("nan")))
    state = await budget.snapshot()
    assert state.calls == state.unknown_usage_calls == 1
    assert state.usage == Usage()


@pytest.mark.parametrize("deadline", [None, 11])
async def test_budget_transport_timeout_remains_retryable_with_no_or_future_deadline(deadline):
    class TransportTimeout:
        async def stream(self, request):
            yield TextChunk(text="failed partial output")
            raise TimeoutError("private transport diagnostic")

    budget = RunBudget(BudgetPolicy(max_calls=2, deadline_unix=deadline), clock=lambda: 10)
    backup = FakeProvider([Turn(text="backup")])
    provider = FallbackProvider(
        [
            target(BudgetProvider(TransportTimeout(), budget), "primary"),
            target(BudgetProvider(backup, budget), "backup"),
        ]
    )
    items = [item async for item in provider.stream(ProviderRequest(model="a", messages=[], max_output_tokens=10))]
    assert isinstance(items[-1], Completed) and items[-1].message.text == "backup"
    assert "failed partial" not in str(items)
    state = await budget.snapshot()
    assert state.calls == 2 and state.completed_calls == state.unknown_usage_calls == 1
    assert len(backup.requests) == 1


async def test_owned_budget_deadline_does_not_retry_system2():
    class Slow:
        async def stream(self, request):
            await asyncio.sleep(10)
            yield TextChunk(text="too late")

    budget = RunBudget(BudgetPolicy(deadline_unix=10.01), clock=lambda: 10)
    backup = FakeProvider([Turn(text="must not run")])
    provider = FallbackProvider([target(BudgetProvider(Slow(), budget), "primary"), target(backup, "backup")])
    with pytest.raises(ProviderError):
        _ = [item async for item in provider.stream(ProviderRequest(model="a", messages=[], max_output_tokens=10))]
    assert not backup.requests and (await budget.snapshot()).unknown_usage_calls == 1


async def test_standalone_provider_transport_timeout_is_retryable_and_private():
    class PrivateTimeout:
        async def stream(self, request):
            if False:
                yield
            raise TimeoutError("credential-bearing-private-timeout")

    provider = BudgetProvider(PrivateTimeout(), RunBudget(BudgetPolicy()))
    with pytest.raises(ProviderError) as captured:
        _ = [item async for item in provider.stream(ProviderRequest(model="a", messages=[], max_output_tokens=10))]
    assert captured.value.retryable
    assert "credential-bearing-private" not in "".join(traceback.format_exception(captured.value))
