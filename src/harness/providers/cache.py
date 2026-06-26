"""Prompt-cache breakpoint placement — the `system_and_3` strategy.

Cache discipline is Law 1. The cached prefix is `tools` + `system` + the message
history; we place `cache_control: {type: ephemeral}` breakpoints so a GROWING
conversation keeps hitting cache: one on the system prompt, and a sliding window
on the last few messages.

These markers are ANTHROPIC-SHAPED. A non-Anthropic provider's message
conversion must strip or translate them — the marker placement is provider-
agnostic (a concept: "cache the stable prefix + recent tail"), the marker SYNTAX
is not.

Pure functions. Always operate on a DEEP COPY — never mutate the live list.
"""

from __future__ import annotations

import copy
from typing import Any

# Anthropic allows up to 4 cache breakpoints. We use system + 3 message points.
MAX_MESSAGE_BREAKPOINTS = 3

_EPHEMERAL = {"type": "ephemeral"}


def _mark_last_block(message: dict[str, Any]) -> None:
    """Attach cache_control to the LAST content block of a message. The cache
    boundary is the end of that block, so everything up to it is cacheable."""
    content = message.get("content")
    if isinstance(content, str):
        # normalize string content to a single text block so we can mark it
        message["content"] = [{"type": "text", "text": content, "cache_control": _EPHEMERAL}]
        return
    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = _EPHEMERAL


def mark_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark cache breakpoints IN PLACE on the last `MAX_MESSAGE_BREAKPOINTS` messages
    (sliding tail window) and return the same list. The caller MUST own `messages`
    (i.e. it is already isolated from canonical history) — the loop deep-copies once
    at the top of the turn, so marking in place there costs no second copy. The system
    prompt is marked separately by the provider (see `mark_system`)."""
    for msg in messages[-MAX_MESSAGE_BREAKPOINTS:]:
        _mark_last_block(msg)
    return messages


def apply_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pure variant: return a DEEP COPY with breakpoints marked, leaving `messages`
    untouched. For callers that don't already own an isolated copy."""
    return mark_cache_breakpoints(copy.deepcopy(messages))


def mark_system(system: str) -> list[dict[str, Any]]:
    """Convert a system string into Anthropic's structured-system form with a
    cache breakpoint, so the (byte-stable) system prefix is cached."""
    return [{"type": "text", "text": system, "cache_control": _EPHEMERAL}]
