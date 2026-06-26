"""Anthropic (Claude) provider.

Implements the `Provider` contract: `profile` (data) + `hooks` (behavior) +
`stream()`. Converts harness tools to the Anthropic tool format, streams the
response (streaming by default — SSE-only gateways crash `.create()` callers),
and normalizes the result into a `Response`.

Message format: the harness keeps messages as Anthropic-shaped dicts already
(Claude-first), so conversion here is mostly tool formatting, system marking,
and pulling tool_use blocks out of the final message into normalized ToolCalls.
"""

from __future__ import annotations

import os
from typing import Any

from harness.core.types import (
    CancelToken,
    Response,
    TextContent,
    Tool,
    ToolCall,
    Usage,
)
from harness.providers.cache import mark_system

DEFAULT_MODEL = "claude-opus-4-8"
_MAX_TOKENS = {
    "claude-opus-4-8": 32000,
    "claude-sonnet-4-6": 64000,
    "claude-haiku-4-5": 32000,
}


class CancelledError(RuntimeError):
    """Raised by the provider when the cancel token trips mid-stream."""


class _AnthropicHooks:
    """ProviderHooks implementation for Anthropic. Minimal for MVP."""

    def __init__(self, profile: "ClaudeProfile") -> None:
        self._profile = profile

    def prepare_messages(self, messages: list[Any]) -> list[Any]:
        return messages

    def build_api_kwargs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (extra_body, top_level_kwargs). For Anthropic, adaptive thinking
        + effort go top-level; extra_body stays empty. The split exists so other
        providers (which expect reasoning config in extra_body) can differ."""
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if self._profile.supports_thinking:
            top_level["thinking"] = {"type": "adaptive"}
        return extra_body, top_level

    def fetch_models(self) -> list[str]:
        return list(self._profile.supported_models)


# ProviderProfile is frozen data; we subclass-by-instance via a thin wrapper so
# the provider can carry model defaults without making the profile mutable.
from dataclasses import dataclass, field  # noqa: E402

from harness.core.types import ProviderProfile  # noqa: E402


@dataclass(frozen=True)
class ClaudeProfile(ProviderProfile):
    pass


def default_profile() -> ClaudeProfile:
    return ClaudeProfile(
        id="anthropic",
        supported_models=tuple(_MAX_TOKENS.keys()),
        max_tokens_by_model=dict(_MAX_TOKENS),
        supports_thinking=True,
        supports_prompt_cache=True,
        reasoning_in_extra_body=False,  # Anthropic: thinking config is top-level
    )


def to_anthropic_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Harness Tool -> Anthropic tool schema. JSON Schema passes through as-is."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def _text_delta(event: Any) -> str | None:
    """Pull a text delta out of an Anthropic stream event, if it is one.
    Anthropic emits `content_block_delta` with a `text_delta` carrying `.text`."""
    if getattr(event, "type", None) != "content_block_delta":
        return None
    delta = getattr(event, "delta", None)
    if delta is not None and getattr(delta, "type", None) == "text_delta":
        return getattr(delta, "text", None)
    return None


def _normalize_final_message(msg: Any) -> Response:
    """Anthropic final message -> normalized Response."""
    content: list[Any] = []
    tool_calls: list[ToolCall] = []
    for block in msg.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            content.append(TextContent(block.text))
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            )
        else:
            # thinking / redacted_thinking / other — keep raw for round-tripping
            content.append(block)

    usage = Usage(
        input_tokens=getattr(msg.usage, "input_tokens", 0),
        output_tokens=getattr(msg.usage, "output_tokens", 0),
        cache_read_tokens=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
    )
    stop = msg.stop_reason or "end_turn"
    # Preserve refusal/pause_turn verbatim — collapsing them to end_turn hides a
    # safety decline and a server-tool pause. Only genuinely unknown values fall back.
    if stop not in ("end_turn", "tool_use", "max_tokens", "stop_sequence", "pause_turn", "refusal"):
        stop = "end_turn"
    return Response(
        content=content, tool_calls=tool_calls, stop_reason=stop, usage=usage, raw=msg
    )


class ClaudeProvider:
    """Concrete `Provider` for the Anthropic Messages API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        profile: ClaudeProfile | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.profile = profile or default_profile()
        self.hooks = _AnthropicHooks(self.profile)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = client  # injectable for tests; lazily built otherwise

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # imported lazily so the package loads without it

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def stream(
        self,
        system: str,
        messages: list[Any],
        tools: list[Tool],
        cancel: CancelToken | None = None,
        on_chunk: Any = None,
    ) -> Response:
        messages = self.hooks.prepare_messages(messages)
        extra_body, top_level = self.hooks.build_api_kwargs()

        max_tokens = self.profile.max_tokens_by_model.get(self.model, 8192)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": mark_system(system) if system else [],
            "messages": messages,
            **top_level,
        }
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        if extra_body:
            kwargs["extra_body"] = extra_body

        client = self._get_client()

        # Streaming by default — aggregate to the final message. Check the cancel
        # token between events so a long generation can be torn down promptly, and
        # forward text deltas to on_chunk for live UIs (ACP AgentMessageChunk).
        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                if cancel is not None and cancel.is_set():
                    stream.close()
                    raise CancelledError("stream cancelled")
                if on_chunk is not None:
                    delta = _text_delta(event)
                    if delta:
                        on_chunk(delta)
            final = stream.get_final_message()

        return _normalize_final_message(final)
