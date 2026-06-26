"""The tool_result invariant — the property that must never regress.

Every tool_use block the model emits gets exactly one tool_result, regardless of
denial, cancellation, unknown tool, or a tool that raises. A violation means the
next provider call 400s.
"""

from __future__ import annotations

from conftest import (
    FakeProvider,
    assistant_text,
    assistant_tool_use,
    echo_tool,
    tool_results,
    tool_use,
)

from harness.core.loop import Agent, AgentConfig
from harness.core.types import CancelToken, Deny, TextContent, Tool, ToolResult


def _run(provider, tools, hooks=()):
    msgs = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    Agent(AgentConfig(provider=provider, tools=tools, hooks=list(hooks))).run(msgs)
    return msgs


def test_normal_call_emits_one_result():
    prov = FakeProvider([assistant_tool_use(tool_use("echo", x=1)), assistant_text("done")])
    msgs = _run(prov, [echo_tool()])
    assert len(tool_results(msgs)) == 1
    assert tool_results(msgs)[0]["is_error"] is False


def test_denial_still_emits_result():
    class DenyAll:
        def before_tool(self, call):
            return Deny("nope")

    prov = FakeProvider([assistant_tool_use(tool_use("echo", x=1)), assistant_text("ok")])
    msgs = _run(prov, [echo_tool()], hooks=[DenyAll()])
    results = tool_results(msgs)
    assert len(results) == 1 and results[0]["is_error"] is True
    assert "nope" in results[0]["content"][0]["text"]


def test_mixed_batch_one_result_per_call():
    class DenyB:
        def before_tool(self, call):
            return Deny("no b") if call.arguments.get("k") == "b" else call

    prov = FakeProvider(
        [
            assistant_tool_use(tool_use("echo", k="a"), tool_use("echo", k="b")),
            assistant_text("fin"),
        ]
    )
    msgs = _run(prov, [echo_tool()], hooks=[DenyB()])
    results = tool_results(msgs)
    assert len(results) == 2
    assert sorted(r["is_error"] for r in results) == [False, True]


def test_unknown_tool_emits_error_result():
    prov = FakeProvider([assistant_tool_use(tool_use("ghost")), assistant_text("ok")])
    msgs = _run(prov, [echo_tool()])  # no 'ghost' tool registered
    results = tool_results(msgs)
    assert len(results) == 1 and results[0]["is_error"] is True
    assert "unknown tool" in results[0]["content"][0]["text"]


def test_raising_tool_emits_error_not_crash():
    def boom(cancel=None, **kw):
        raise RuntimeError("kaboom")

    raiser = Tool(name="boom", description="x", input_schema={"type": "object"}, handler=boom)
    prov = FakeProvider([assistant_tool_use(tool_use("boom")), assistant_text("ok")])
    msgs = _run(prov, [raiser])
    results = tool_results(msgs)
    assert len(results) == 1 and results[0]["is_error"] is True
    assert "kaboom" in results[0]["content"][0]["text"]


def test_mid_batch_cancel_fills_remaining_results():
    # A cancel token already set: every call short-circuits to a 'cancelled'
    # result, but the invariant still holds (N tool_use -> N tool_result).
    token = CancelToken()

    class SetCancelBefore:
        def before_model(self, messages):
            token.set()
            return messages

    prov = FakeProvider(
        [assistant_tool_use(tool_use("echo", i=1), tool_use("echo", i=2))]
    )
    msgs = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    agent = Agent(AgentConfig(provider=prov, tools=[echo_tool()], hooks=[SetCancelBefore()]))
    agent.run(msgs, cancel=token)
    results = tool_results(msgs)
    assert len(results) == 2
    assert all(r["is_error"] for r in results)
    assert all("cancelled" in r["content"][0]["text"] for r in results)


def test_after_tool_sees_repaired_call_object():
    # before_tool mutates the call; after_tool must see the SAME mutated object.
    seen = {}

    class Mutate:
        def before_tool(self, call):
            call.arguments["touched"] = True
            return call

        def after_tool(self, call, result):
            seen["args"] = dict(call.arguments)

    prov = FakeProvider([assistant_tool_use(tool_use("echo", x=1)), assistant_text("done")])
    _run(prov, [echo_tool()], hooks=[Mutate()])
    assert seen["args"].get("touched") is True
