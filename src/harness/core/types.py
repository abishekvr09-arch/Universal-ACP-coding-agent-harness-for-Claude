"""Core type contracts for the harness.

These are the load-bearing data shapes the loop, providers, tools, and hooks all
agree on. Kept deliberately small (the "narrow waist"): capability lives at the
edges (tools/providers/hooks), not here.

Design notes baked into these types:
- `ProviderProfile` is PURE DATA (frozen dataclass); behavior lives in the
  separate `ProviderHooks` Protocol. Data/behavior split is unambiguous.
- Every `tool_use` must get exactly one `tool_result` — `error_result()` is the
  helper that preserves that invariant on denial/cancel/exception.
- `Deny` is a concrete type (not a bare sentinel) so `before_tool` can attach a
  human-readable reason that flows into the synthetic error result.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #

# A CancelToken is just a threading.Event: set() to cancel, checked
# cooperatively by the loop, the provider stream, and long-running tools.
CancelToken = threading.Event


# --------------------------------------------------------------------------- #
# Content blocks
# --------------------------------------------------------------------------- #


@dataclass
class TextContent:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ImageContent:
    # base64-encoded image data + media type, Anthropic-shaped.
    data: str
    media_type: str
    type: Literal["image"] = "image"


Content = TextContent | ImageContent


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

ExecutionMode = Literal["parallel", "sequential"]


@dataclass
class Tool:
    """A self-describing tool: schema (what the model sees) + handler (the code).

    Tool *declarations* (name/description/input_schema) are frozen at conversation
    start — they hash into the prompt cache. Tool *handlers* may be lazy-loaded
    (handler=None until first call, then resolved); the model never sees the
    difference.
    """

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema for the tool's arguments
    handler: Callable[..., "ToolResult"] | None = None  # None => lazy, resolve on first call
    parallel_safe: bool = True
    requires_approval: bool = False
    execution_mode: ExecutionMode = "parallel"
    tags: tuple[str, ...] = ()  # e.g. ("execute",) marks a code/exec tool for budget refund


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str  # the tool_use block id; the matching tool_result must echo it
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The outcome of one tool call. Maps onto an Anthropic tool_result block."""

    content: list[Content]
    is_error: bool = False  # maps to Anthropic is_error:true — keep failures explicit
    terminate: bool = False  # hint: if ALL tools in a batch terminate, the loop stops


def error_result(call_id: str, message: str) -> ToolResult:
    """Synthetic result for denial / cancellation / tool exception.

    Preserves the one-tool_result-per-tool_use invariant: every tool_use block
    the model emitted must get a result, or the next provider call 400s with a
    tool_use/tool_result mismatch. `call_id` is accepted for symmetry/clarity at
    call sites even though the loop pairs results with calls positionally.
    """
    return ToolResult(content=[TextContent(message)], is_error=True)


# --------------------------------------------------------------------------- #
# Hooks
# --------------------------------------------------------------------------- #


@dataclass
class Deny:
    """Returned by `before_tool` to block a call. Concrete (not a bare sentinel)
    so the reason flows into the synthetic error_result the loop emits."""

    reason: str = "denied by approval hook"


@runtime_checkable
class Hook(Protocol):
    """Middleware on the loop's lifecycle. All methods are optional in practice;
    the loop calls them defensively. A Hook implementation may define any subset.
    """

    def before_model(self, messages: list[Any]) -> list[Any]:
        """Fires after ephemeral injection, before cache breakpoints. Runs on the
        per-call message COPY — never the live list. May inspect or transform."""
        ...

    def before_tool(self, call: ToolCall) -> ToolCall | Deny:
        """Sees the REPAIRED call. Return the (possibly modified) call to allow,
        or a Deny to block. What's approved here is exactly what runs."""
        ...

    def after_tool(self, call: ToolCall, result: ToolResult) -> None:
        """Observability. Receives the SAME (repaired) call object before_tool
        saw, paired with its result — never the original pre-repair call."""
        ...


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderProfile:
    """PURE DATA describing a model backend. No behavior — that's ProviderHooks.

    `reasoning_in_extra_body` captures the one real cross-provider disagreement:
    some put reasoning/thinking config in the request's extra_body, others as a
    top-level kwarg. The provider's build_api_kwargs() returns both.
    """

    id: str
    supported_models: tuple[str, ...]
    max_tokens_by_model: dict[str, int] = field(default_factory=dict)
    supports_thinking: bool = True
    supports_prompt_cache: bool = True
    reasoning_in_extra_body: bool = True


@runtime_checkable
class ProviderHooks(Protocol):
    """Optional behavior overrides — one implementation per provider."""

    def prepare_messages(self, messages: list[Any]) -> list[Any]:
        ...

    def build_api_kwargs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Returns (extra_body, top_level_kwargs) — the reasoning-config split."""
        ...

    def fetch_models(self) -> list[str]:
        ...


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


# end_turn/tool_use/max_tokens/stop_sequence/pause_turn/refusal are Anthropic's
# documented stop reasons; cancelled/error are harness-internal (cooperative
# cancel, provider failure). `refusal` MUST be preserved — a safety decline is a
# 200 with stop_reason="refusal", not an error, and the caller needs the signal.
StopReason = Literal[
    "end_turn", "tool_use", "max_tokens", "stop_sequence",
    "pause_turn", "refusal", "cancelled", "error",
]


@dataclass
class Response:
    """A normalized assistant turn returned by a provider."""

    content: list[Content]  # the raw assistant content blocks (text, thinking, etc.)
    tool_calls: list[ToolCall]
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)
    raw: Any = None  # the provider's native message object, for round-tripping


# on_chunk receives text deltas as they stream from the model, for live UIs
# (e.g. ACP AgentMessageChunk). Non-streaming callers pass None and it's ignored.
ChunkCallback = Callable[[str], None]


@runtime_checkable
class Provider(Protocol):
    profile: ProviderProfile  # data
    hooks: ProviderHooks  # behavior

    def stream(
        self,
        system: str,
        messages: list[Any],
        tools: list[Tool],
        cancel: CancelToken | None = None,
        on_chunk: "ChunkCallback | None" = None,
    ) -> Response:
        ...
