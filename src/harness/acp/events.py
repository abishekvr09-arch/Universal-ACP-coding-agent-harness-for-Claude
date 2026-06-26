"""Mapping harness types → ACP `session_update` payloads and tool kinds.

Thin, dependency-isolating layer over the `agent-client-protocol` SDK helpers, so
the rest of the harness never imports `acp` directly and the pinned method/shape
knowledge lives in one place. (Verified against agent-client-protocol 0.10.1.)
"""

from __future__ import annotations

from typing import Any

import acp
from acp import schema

from harness.core.types import ToolCall, ToolResult

# Map our tool `tags` onto ACP ToolKind. Default: "other".
_TAG_TO_KIND = {
    "read": "read",
    "edit": "edit",
    "search": "search",
    "execute": "execute",
    "think": "think",
    "fetch": "fetch",
}

# Our internal stop_reason -> ACP PromptResponse stop_reason.
# ACP allows: end_turn, max_tokens, max_turn_requests, refusal, cancelled.
_STOP_MAP = {
    "end_turn": "end_turn",
    "tool_use": "end_turn",  # internal-only; by the time we answer, the turn ended
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
    "pause_turn": "end_turn",  # server-tool pause; no server tools in MVP
    "refusal": "refusal",  # safety decline — surface it to the editor
    "cancelled": "cancelled",
    "error": "refusal",
}


def tool_kind(tags: tuple[str, ...]) -> str:
    for t in tags:
        if t in _TAG_TO_KIND:
            return _TAG_TO_KIND[t]
    return "other"


def map_stop_reason(internal: str) -> str:
    return _STOP_MAP.get(internal, "end_turn")


def message_chunk(text: str) -> Any:
    """An AgentMessageChunk update for streamed assistant text."""
    return acp.update_agent_message_text(text)


def thought_chunk(text: str) -> Any:
    return acp.update_agent_thought_text(text)


# ToolCallStatus / ToolKind / PermissionOptionKind are Literal string aliases in
# the SDK (not enums) — pass the string values directly.
def tool_call_start(call: ToolCall, kind: str) -> Any:
    """A ToolCallStart update (status=in_progress) when a tool begins."""
    return acp.start_tool_call(
        tool_call_id=call.id,
        title=call.name,
        kind=kind,
        status="in_progress",
        raw_input=call.arguments,
    )


def tool_call_done(call: ToolCall, result: ToolResult) -> Any:
    """A ToolCallProgress update marking completion or failure."""
    status = "failed" if result.is_error else "completed"
    text = " ".join(
        getattr(b, "text", "") for b in result.content if getattr(b, "type", None) == "text"
    )
    return acp.update_tool_call(
        tool_call_id=call.id,
        status=status,
        content=[acp.tool_content(acp.text_block(text))] if text else None,
    )


def permission_options() -> list[Any]:
    """The standard allow/deny options offered for a gated tool call."""
    return [
        schema.PermissionOption(kind="allow_once", name="Allow once", option_id="allow_once"),
        schema.PermissionOption(
            kind="allow_always", name="Allow for session", option_id="allow_always"
        ),
        schema.PermissionOption(kind="reject_once", name="Deny", option_id="reject_once"),
    ]


def permission_tool_call(call: ToolCall, kind: str) -> Any:
    """The ToolCallUpdate passed to client.request_permission."""
    return schema.ToolCallUpdate(
        tool_call_id=call.id,
        title=call.name,
        kind=kind,
        raw_input=call.arguments,
    )


def is_allow_outcome(outcome: Any) -> bool:
    """True if a RequestPermissionResponse.outcome is an allow (vs deny)."""
    return isinstance(outcome, schema.AllowedOutcome)
