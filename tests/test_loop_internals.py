"""Regression tests for two audit findings:

1. A TypeError raised INSIDE a handler body must NOT be swallowed/retried as an
   arity mismatch (the old broad `except TypeError` masked real bugs).
2. Promoting plaintext tool calls strips the matched span from the text, so the
   raw=None history fallback never double-emits prose + structured tool_use.
"""

from __future__ import annotations

from conftest import FakeProvider, assistant_text, assistant_tool_use, tool_use

from harness.core.loop import Agent, AgentConfig, _accepts_cancel, _assistant_message
from harness.core.repair import promote_plaintext_tool_calls
from harness.core.types import Response, TextContent, Tool, ToolResult


def test_accepts_cancel_detection():
    def with_cancel(cancel=None, **kw):
        return None

    def with_kwargs(**kw):
        return None

    def without(path=None):
        return None

    assert _accepts_cancel(with_cancel) is True
    assert _accepts_cancel(with_kwargs) is True  # **kwargs can receive cancel
    assert _accepts_cancel(without) is False


def test_internal_typeerror_is_surfaced_not_masked():
    # Handler accepts cancel (so no arity fallback), but raises TypeError inside.
    def boom(cancel=None, **kw):
        raise TypeError("genuine bug: NoneType is not subscriptable")

    tool = Tool(name="boom", description="x", input_schema={"type": "object"}, handler=boom)
    prov = FakeProvider([assistant_tool_use(tool_use("boom")), assistant_text("ok")])
    msgs = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    Agent(AgentConfig(provider=prov, tools=[tool])).run(msgs)
    results = [
        b
        for m in msgs
        if m["role"] == "user"
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    # exactly one result, is_error, and it carries the REAL message (not retried away)
    assert len(results) == 1 and results[0]["is_error"] is True
    assert "genuine bug" in results[0]["content"][0]["text"]


def test_handler_without_cancel_still_runs_once():
    calls = {"n": 0}

    def no_cancel(path=None):
        calls["n"] += 1
        return ToolResult(content=[TextContent(f"read {path}")])

    tool = Tool(
        name="read", description="x",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        handler=no_cancel,
    )
    prov = FakeProvider([assistant_tool_use(tool_use("read", path="a")), assistant_text("ok")])
    Agent(AgentConfig(provider=prov, tools=[tool])).run(
        [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    )
    assert calls["n"] == 1  # ran exactly once, not retried


def test_promotion_strips_matched_span_keeps_prose():
    r = Response(
        content=[TextContent('Let me read it. [read]\n{"path":"x"}')],
        tool_calls=[],
        stop_reason="end_turn",
    )
    promote_plaintext_tool_calls(r, {"read"})
    assert r.stop_reason == "tool_use" and r.tool_calls[0].name == "read"
    # the bracket call is gone from the text; surrounding prose kept
    remaining = "".join(b.text for b in r.content if isinstance(b, TextContent))
    assert "read it" in remaining and "[read]" not in remaining


def test_assistant_message_fallback_no_double_emit():
    # raw=None forces the reconstruction fallback. After promotion+strip, the
    # rebuilt assistant message must have exactly ONE tool_use and no leftover
    # tool-call syntax in any text block.
    r = Response(
        content=[TextContent('[read]\n{"path":"x"}')], tool_calls=[], stop_reason="end_turn"
    )
    promote_plaintext_tool_calls(r, {"read"})
    msg = _assistant_message(r)  # raw is None -> fallback path
    blocks = msg["content"]
    tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
    texts = [b for b in blocks if b.get("type") == "text"]
    assert len(tool_uses) == 1
    assert all("[read]" not in b["text"] for b in texts)
