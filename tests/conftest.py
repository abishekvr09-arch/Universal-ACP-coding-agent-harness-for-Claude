"""Shared test fixtures — re-exports the scripted fake provider, which now lives in
`harness.testing` so `acp/server.py` can share the SAME fake (one fake, not two).
See the promote-the-fake decision: tests import from here unchanged."""

from __future__ import annotations

from harness.testing import (
    FakeProvider,
    assistant_text,
    assistant_tool_use,
    echo_tool,
    tool_results,
    tool_use,
)

__all__ = [
    "FakeProvider",
    "assistant_text",
    "assistant_tool_use",
    "echo_tool",
    "tool_results",
    "tool_use",
]
