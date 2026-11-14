"""A provider that replays scripted replies, for deterministic tests of everything above the wire."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from intern.messages import AssistantMessage, InvalidToolCall, JsonObject, StopKind, ToolCall, Usage
from intern.providers.base import Completed, ProviderError, ProviderRequest, StreamItem, TextChunk


@dataclass
class Turn:
    """One scripted reply. `tool_calls` maps tool name to arguments, in order."""

    text: str = ""
    tool_calls: list[tuple[str, JsonObject]] = field(default_factory=list)
    invalid_tool_calls: list[tuple[str, str]] = field(default_factory=list)
    reasoning: str | None = None
    stop: StopKind | None = None
    input_tokens: int | None = None
    output_tokens: int = 10
    error: str | None = None
    retryable: bool = False


Script = list[Turn] | Callable[[ProviderRequest], Turn]


class FakeProvider:
    def __init__(self, script: Script, model: str = "fake/model") -> None:
        self._script = script
        self._index = 0
        self.model = model
        self.requests: list[ProviderRequest] = []

    def _next(self, request: ProviderRequest) -> Turn:
        if callable(self._script):
            return self._script(request)
        if self._index >= len(self._script):
            return Turn(text="(script exhausted)")
        turn = self._script[self._index]
        self._index += 1
        return turn

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamItem]:
        self.requests.append(request)
        turn = self._next(request)
        if turn.error:
            raise ProviderError(turn.error, retryable=turn.retryable)
        for word in turn.text.split(" "):
            if word:
                yield TextChunk(text=word + " ")
        calls = [
            ToolCall(id=f"call_{self._index}_{i}", name=n, arguments=a) for i, (n, a) in enumerate(turn.tool_calls)
        ]
        invalid = [
            InvalidToolCall(
                id=f"bad_{self._index}_{i}", name=n, raw_arguments=raw, error="arguments are not valid JSON"
            )
            for i, (n, raw) in enumerate(turn.invalid_tool_calls)
        ]
        chars = sum(len(str(m.model_dump())) for m in request.messages)
        yield Completed(
            message=AssistantMessage(
                text=turn.text,
                reasoning=turn.reasoning,
                tool_calls=calls,
                invalid_tool_calls=invalid,
                model=self.model,
                stop=turn.stop or ("tool_calls" if calls else "stop"),
                usage=Usage(
                    input_tokens=turn.input_tokens if turn.input_tokens is not None else chars // 4,
                    output_tokens=turn.output_tokens,
                ),
            )
        )
