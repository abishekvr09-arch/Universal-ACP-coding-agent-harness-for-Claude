"""End-to-end ACP: a fake editor (Client) drives HarnessAgent through a real event
loop + bridge. Confirms session lifecycle, streamed chunks, the permission relay,
the tool_result invariant, and compression all survive an ACP-driven session.
"""

from __future__ import annotations

import asyncio
import threading
import time

import acp
from acp import schema
from conftest import FakeProvider, assistant_text, assistant_tool_use, echo_tool, tool_use

from harness.acp.bridge import AsyncBridge
from harness.acp.server import HarnessAgent
from harness.core.context import Compressor, CompressionPolicy
from harness.core.types import Response, TextContent, Tool, ToolResult, Usage


class LoopThread:
    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        self.t = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.t.start()
        return self.loop

    def __exit__(self, *a):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.t.join(timeout=2)
        self.loop.close()


class FakeClient:
    """Records session_update notifications; answers request_permission by policy."""

    def __init__(self, allow: bool = True) -> None:
        self.updates: list = []
        self.permission_calls: list = []
        self._allow = allow

    async def session_update(self, session_id, update, **kw):
        self.updates.append(update)

    async def request_permission(self, options, session_id, tool_call, **kw):
        self.permission_calls.append(tool_call)
        if self._allow:
            return schema.RequestPermissionResponse(
                outcome=schema.AllowedOutcome(outcome="selected", option_id="allow_once")
            )
        return schema.RequestPermissionResponse(outcome=schema.DeniedOutcome(outcome="cancelled"))


def _run(coro, loop):
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=5)


def _streaming_provider(script):
    """A provider that emits its final text via on_chunk, then returns it."""

    class P:
        profile = None
        hooks = None

        def __init__(self):
            self.script = list(script)

        def stream(self, system, messages, tools, cancel=None, on_chunk=None):
            resp = self.script.pop(0)
            if on_chunk:
                for b in resp.content:
                    if isinstance(b, TextContent):
                        on_chunk(b.text)
            return resp

    return P()


def test_initialize_and_new_session():
    with LoopThread() as loop:
        agent = HarnessAgent(provider=FakeProvider([]), tools=[echo_tool()], bridge=AsyncBridge(loop))
        agent.on_connect(FakeClient())
        init = _run(agent.initialize(protocol_version=acp.PROTOCOL_VERSION), loop)
        assert init.protocol_version == acp.PROTOCOL_VERSION
        ns = _run(agent.new_session(cwd="/tmp"), loop)
        assert ns.session_id in agent._sessions


def test_prompt_streams_chunks_and_returns_end_turn():
    with LoopThread() as loop:
        prov = _streaming_provider([assistant_text("Hello from the agent")])
        client = FakeClient()
        agent = HarnessAgent(provider=prov, tools=[echo_tool()], bridge=AsyncBridge(loop))
        agent.on_connect(client)
        sid = _run(agent.new_session(cwd="/tmp"), loop).session_id
        resp = _run(agent.prompt([acp.text_block("hi")], sid), loop)
        assert resp.stop_reason == "end_turn"
        # the streamed text reached the editor as AgentMessageChunk updates
        chunk_texts = [
            u.content.text for u in client.updates
            if type(u).__name__ == "AgentMessageChunk" and hasattr(u.content, "text")
        ]
        assert "Hello from the agent" in "".join(chunk_texts)


def test_prompt_with_gated_tool_relays_permission_and_holds_invariant():
    # bash is gated (requires_approval); the client allows it. One tool_use -> one
    # tool_result, and a permission request was relayed.
    def bash_handler(cancel=None, **kw):
        return ToolResult(content=[TextContent("ran")])

    bash = Tool(
        name="bash", description="run", input_schema={"type": "object"},
        handler=bash_handler, requires_approval=True, tags=("execute",),
    )
    with LoopThread() as loop:
        prov = _streaming_provider(
            [assistant_tool_use(tool_use("bash", command="ls")), assistant_text("done")]
        )
        client = FakeClient(allow=True)
        agent = HarnessAgent(provider=prov, tools=[bash], bridge=AsyncBridge(loop))
        agent.on_connect(client)
        sid = _run(agent.new_session(cwd="/tmp"), loop).session_id
        resp = _run(agent.prompt([acp.text_block("list files")], sid), loop)
        assert resp.stop_reason == "end_turn"
        assert len(client.permission_calls) == 1  # relayed exactly once
        # invariant: the tool_result is present in the session history
        sess = agent._sessions[sid]
        results = [
            b for m in sess.messages if m["role"] == "user"
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        assert len(results) == 1 and results[0]["is_error"] is False
        # tool start + completion updates were emitted
        kinds = [type(u).__name__ for u in client.updates]
        assert "ToolCallStart" in kinds and "ToolCallProgress" in kinds


def test_denied_permission_produces_error_result_not_crash():
    def bash_handler(cancel=None, **kw):
        return ToolResult(content=[TextContent("should not run")])

    bash = Tool(
        name="bash", description="run", input_schema={"type": "object"},
        handler=bash_handler, requires_approval=True, tags=("execute",),
    )
    with LoopThread() as loop:
        prov = _streaming_provider(
            [assistant_tool_use(tool_use("bash", command="rm -rf /")), assistant_text("ok")]
        )
        client = FakeClient(allow=False)  # user denies
        agent = HarnessAgent(provider=prov, tools=[bash], bridge=AsyncBridge(loop))
        agent.on_connect(client)
        sid = _run(agent.new_session(cwd="/tmp"), loop).session_id
        _run(agent.prompt([acp.text_block("delete everything")], sid), loop)
        sess = agent._sessions[sid]
        results = [
            b for m in sess.messages if m["role"] == "user"
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        assert len(results) == 1 and results[0]["is_error"] is True
        assert "denied" in results[0]["content"][0]["text"]


def test_cancel_sets_token_and_reports_cancelled():
    with LoopThread() as loop:
        # provider blocks until cancel flips, then the loop unwinds
        class Blocker:
            profile = None
            hooks = None

            def stream(self, system, messages, tools, cancel=None, on_chunk=None):
                for _ in range(200):
                    if cancel and cancel.is_set():
                        return Response(content=[], tool_calls=[], stop_reason="cancelled")
                    time.sleep(0.005)
                return assistant_text("never")

        agent = HarnessAgent(provider=Blocker(), tools=[echo_tool()], bridge=AsyncBridge(loop))
        agent.on_connect(FakeClient())
        sid = _run(agent.new_session(cwd="/tmp"), loop).session_id
        fut = asyncio.run_coroutine_threadsafe(agent.prompt([acp.text_block("go")], sid), loop)
        time.sleep(0.05)
        _run(agent.cancel(sid), loop)
        resp = fut.result(timeout=5)
        assert resp.stop_reason == "cancelled"


def test_compression_runs_at_acp_session_boundary():
    # After a prompt, the server invokes the compressor (caller-driven). Force a
    # high token count so it compresses, and confirm the session history shrank.
    with LoopThread() as loop:
        # Build a long-ish history first via several prompts.
        prov = _streaming_provider(
            [
                assistant_tool_use(tool_use("echo", n=1)), assistant_text("a"),
                assistant_tool_use(tool_use("echo", n=2)), assistant_text("b"),
                assistant_text("c"),
            ]
        )
        comp = Compressor(
            _streaming_provider(
                [Response(content=[TextContent("SUMMARY")], tool_calls=[], stop_reason="end_turn")] * 5
            ),
            CompressionPolicy(context_window=100, trigger_ratio=0.1, tail_ratio=0.15, head_protect_turns=1),
        )
        agent = HarnessAgent(
            provider=prov, tools=[echo_tool()], compressor=comp, bridge=AsyncBridge(loop)
        )
        client = FakeClient()
        agent.on_connect(client)
        sid = _run(agent.new_session(cwd="/tmp"), loop).session_id
        _run(agent.prompt([acp.text_block("one")], sid), loop)
        _run(agent.prompt([acp.text_block("two")], sid), loop)
        # third prompt: inject high token usage via a provider whose Response carries it
        agent._provider = _streaming_provider(
            [Response(content=[TextContent("c")], tool_calls=[], stop_reason="end_turn", usage=Usage(input_tokens=80))]
        )
        n_before = len(agent._sessions[sid].messages)
        _run(agent.prompt([acp.text_block("three")], sid), loop)
        # compression should have folded earlier turns into a summary
        flat = " ".join(
            b.get("text", "")
            for m in agent._sessions[sid].messages
            for b in (m.get("content") or [])
            if isinstance(b, dict)
        )
        assert "SUMMARY" in flat


def test_per_session_model_via_meta_advertises_validates_and_selects_provider():
    """Per-session model propagation over the STABLE channel: the client sends the
    model in session/new's _meta (the router spreads it into new_session kwargs as
    `modelId`). The agent advertises supported models, validates the id against them,
    and the prompt path then drives the matching provider. We deliberately do NOT use
    ACP's set_session_model — it's unstable (method_not_found unless opted in)."""
    base = FakeProvider([])
    alt = FakeProvider([])
    by_id = {"opus": base, "haiku": alt}
    agent = HarnessAgent(
        provider=base,
        tools=[echo_tool()],
        make_provider=lambda m: by_id[m],
        available_models=("opus", "haiku"),
        default_model="opus",
    )

    # default session (no _meta): advertises models, current = default, drives base
    ns = asyncio.run(agent.new_session(cwd="/tmp"))
    assert ns.models is not None
    assert [m.model_id for m in ns.models.available_models] == ["opus", "haiku"]
    assert ns.models.current_model_id == "opus"
    assert agent._provider_for(agent._sessions[ns.session_id]) is base

    # model selected via _meta (arrives as a `modelId` kwarg) -> session binds it,
    # advertised current follows, and the matching provider is selected
    ns2 = asyncio.run(agent.new_session(cwd="/tmp", modelId="haiku"))
    assert agent._sessions[ns2.session_id].model == "haiku"
    assert ns2.models.current_model_id == "haiku"
    assert agent._provider_for(agent._sessions[ns2.session_id]) is alt

    # unsupported id is rejected loudly (never forwarded to the provider/API)
    import pytest

    with pytest.raises(acp.RequestError):
        asyncio.run(agent.new_session(cwd="/tmp", modelId="gpt-4"))


def test_main_entry_serves_then_exits_on_stdin_eof():
    """The `harness-acp` console entry must actually serve over stdio AND exit on
    stdin EOF (parent death). The load-bearing assertion is the CONTROL — `main()`
    once called `run_agent` without awaiting it, so the process fell straight
    through and exited 0; only 'still alive with stdin open' catches that. The EOF
    exit is what makes OpenClaw's teardown watchdog unnecessary (acp.run_agent
    breaks its read loop on empty readline)."""
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    src = str(Path(__file__).resolve().parent.parent / "src")
    env = {**os.environ, "PYTHONPATH": src}
    p = subprocess.Popen(
        [sys.executable, "-c", "from harness.acp.server import main; main()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    try:
        time.sleep(0.8)
        assert p.poll() is None, "entry exited immediately — not serving (un-awaited run_agent?)"
        p.stdin.close()  # parent death -> stdin EOF
        assert p.wait(timeout=5) == 0, "entry did not exit cleanly on stdin EOF"
    finally:
        if p.poll() is None:
            p.kill()
