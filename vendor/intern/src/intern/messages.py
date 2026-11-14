"""Conversation messages and the tool result envelope.

These types are shared by the journal, the providers and the protocol, so they import nothing else
from the package.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, JsonValue

JsonObject = dict[str, JsonValue]


class Usage(BaseModel):
    """Token usage and cost of one or more model calls, as reported by the provider.

    `input_tokens` is the whole prompt, cache reads and writes included.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def __sub__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            cache_read_tokens=self.cache_read_tokens - other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens - other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens - other.reasoning_tokens,
            cost_usd=self.cost_usd - other.cost_usd,
        )


class System1Usage(BaseModel):
    """What a session's System 1 asks added up to: answered by the backend, failed (unanswered after every retry),
    and the backend's reported tokens and cost. Cached answers count in none of these."""

    asks: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def since(self, earlier: System1Usage) -> System1Usage:
        return System1Usage(**{k: getattr(self, k) - getattr(earlier, k) for k in type(self).model_fields})


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: JsonObject = Field(default_factory=dict)


class InvalidToolCall(BaseModel):
    """A tool call whose arguments could not be parsed. It is never sent back to the model."""

    id: str
    name: str
    raw_arguments: str
    error: str


class ResourceRef(BaseModel):
    """Something a tool saw, by pointer plus content hash, so memory can cite it without a copy."""

    uri: str
    sha256: str | None = None


class ImagePart(BaseModel):
    """An image a tool showed the model. The journal keeps the stored file's path, never the bytes;
    providers read the file when they build a request."""

    path: str = Field(description="The stored, already resized image the provider sends.")
    mime: str
    width: int
    height: int
    source: str = Field(description="What it shows, in the model's terms, e.g. `report.pdf page 3`.")
    sha256: str


class FileDiff(BaseModel):
    """A file a tool changed, as whole texts before and after, for clients to show; never sent to the model."""

    path: str = Field(description="Absolute path.")
    old_text: str | None = Field(description="The text before; None when the file was created.")
    new_text: str


class ToolError(BaseModel):
    what_failed: str
    argument: str | None = None
    next_action: str = Field(description="Small models recover only when this names the action to take.")
    retry_safe: bool = True


class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    status: Literal["ok", "error", "refused"]
    content: str = Field(description="Bounded; what the model sees.")
    observation_id: str | None = Field(
        default=None, description="Journal entry holding the full output when `content` was bounded."
    )
    error: ToolError | None = None
    resources: list[ResourceRef] = Field(default_factory=list)
    images: list[ImagePart] = Field(default_factory=list, description="Images the tool showed the model.")
    duration_ms: int = 0
    omitted_bytes: int = Field(default=0, ge=0, description="Bytes omitted from the executor's in-memory excerpt.")
    capture_omitted_bytes: int = Field(default=0, ge=0, description="Bytes beyond the command archive's disk limit.")


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str
    source: Literal["user", "harness"] = Field(
        default="user", description="`harness` marks nudges and notices the engine added itself."
    )


StopKind = Literal["stop", "length", "tool_calls", "error", "aborted"]


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    text: str = ""
    reasoning: str | None = None
    reasoning_blocks: list[JsonObject] = Field(
        default_factory=list,
        description="Signed reasoning (Anthropic thinking blocks), replayed verbatim to the model that wrote it.",
    )
    tool_calls: list[ToolCall] = Field(default_factory=list)
    invalid_tool_calls: list[InvalidToolCall] = Field(default_factory=list)
    model: str = Field(default="", description="`provider/model` that wrote it; reasoning replays only to it.")
    stop: StopKind = "stop"
    usage: Usage = Field(default_factory=Usage)
    duration_ms: int = 0
    error: str | None = None


class ToolResultMessage(BaseModel):
    role: Literal["tool"] = "tool"
    result: ToolResult


Message = Annotated[SystemMessage | UserMessage | AssistantMessage | ToolResultMessage, Field(discriminator="role")]
