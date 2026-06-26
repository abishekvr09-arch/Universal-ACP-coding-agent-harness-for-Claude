"""Importable test affordances — the single home for the scripted fake provider.

Imported by the test suite (via `conftest`) AND, lazily + env-gated, by
`acp/server.py` (`HARNESS_PROVIDER=fake`) for offline ACP verification. One fake,
not two. This module is NEVER loaded in normal operation — the shipped entry point
only imports it when explicitly asked for the fake.
"""

from __future__ import annotations

from harness.core.types import Response, TextContent, Tool, ToolCall, ToolResult


class FakeProvider:
    """Returns a pre-scripted list of Responses, one per stream() call. Records
    the messages it was handed each turn so tests can assert on history shape."""

    profile = None
    hooks = None

    def __init__(self, script: list[Response]) -> None:
        self.script = list(script)
        self.seen_messages: list[list[dict]] = []

    def stream(self, system, messages, tools, cancel=None) -> Response:
        self.seen_messages.append([dict(m) for m in messages])
        if not self.script:
            return Response(content=[TextContent("(end)")], tool_calls=[], stop_reason="end_turn")
        return self.script.pop(0)


def tool_use(name: str, **args) -> ToolCall:
    return ToolCall(id=f"id_{name}_{len(args)}", name=name, arguments=args)


def assistant_tool_use(*calls: ToolCall) -> Response:
    return Response(content=[], tool_calls=list(calls), stop_reason="tool_use")


def assistant_text(text: str) -> Response:
    return Response(content=[TextContent(text)], tool_calls=[], stop_reason="end_turn")


def echo_tool(name: str = "echo", *, parallel_safe: bool = True, tags: tuple = ()) -> Tool:
    def handler(cancel=None, **kwargs):
        return ToolResult(content=[TextContent(f"{name}:{kwargs}")])

    return Tool(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object"},
        handler=handler,
        parallel_safe=parallel_safe,
        tags=tags,
    )


def tool_results(messages: list[dict]) -> list[dict]:
    """Pull all tool_result blocks out of the user messages in a history."""
    return [
        b
        for m in messages
        if m.get("role") == "user"
        for b in m.get("content", [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]


# --------------------------------------------------------------------------- #
# Scenario provider — for the offline ACP round-trip probe (HARNESS_PROVIDER=fake)
# --------------------------------------------------------------------------- #


class _ScenarioProvider:
    """Deterministic, state-driven (not consumed-list, so it re-runs every prompt):
    turn 1 → stream some text + call the `echo` tool; turn 2 (after the tool_result)
    → stream `pong`. Drives `on_chunk` so the streaming path is real. This forces a
    tool call on every prompt, making the tool_result-invariant check deterministic."""

    profile = None
    hooks = None

    def stream(self, system, messages, tools, cancel=None, on_chunk=None) -> Response:
        last = messages[-1] if messages else {}
        content = last.get("content") if isinstance(last, dict) else None
        has_tool_result = isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
        if has_tool_result:
            resp = Response(content=[TextContent("pong")], tool_calls=[], stop_reason="end_turn")
        else:
            resp = Response(
                content=[TextContent("working ")],
                tool_calls=[ToolCall(id="call_echo_1", name="echo", arguments={"text": "ping"})],
                stop_reason="tool_use",
            )
        if on_chunk is not None:
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    on_chunk(block.text)
        return resp


def build_fake_provider() -> _ScenarioProvider:
    return _ScenarioProvider()


class _BlockingProvider:
    """Blocks in stream() until the CancelToken trips, then RETURNS (does not raise)
    so the loop's top-of-iteration cancel check breaks cleanly into stop_reason
    "cancelled" (loop.py:182). Used to verify cancellation across the ACP wire:
    a `session/cancel` from the client sets the token mid-turn."""

    profile = None
    hooks = None

    def stream(self, system, messages, tools, cancel=None, on_chunk=None) -> Response:
        if cancel is not None:
            while not cancel.is_set():
                cancel.wait(0.05)  # cooperative; the worker thread sees the Event
        return Response(content=[TextContent("")], tool_calls=[], stop_reason="end_turn")


def build_blocking_provider() -> _BlockingProvider:
    return _BlockingProvider()


def fake_tools() -> list[Tool]:
    """A single read-only `echo` tool (requires_approval=False → not gated), so the
    scenario's tool call runs end to end without a permission round-trip."""

    def handler(cancel=None, **kwargs):
        return ToolResult(content=[TextContent(f"echo:{kwargs}")])

    return [
        Tool(
            name="echo",
            description="echo (fake)",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=handler,
            parallel_safe=True,
            requires_approval=False,
        )
    ]
