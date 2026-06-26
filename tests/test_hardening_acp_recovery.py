"""Adversarial hardening — Bucket B: ACP store-backed recovery.

The ACP server kept sessions in memory only, so a process crash lost them. These
acceptance tests prove a fresh process can `session/load` a prior session (system
restored byte-for-byte + full message log), continue it, and that a turn interrupted
by the crash is reconciled on resume — closing the gap that bucket A's force-kill /
shutdown probes would otherwise hit.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading

import acp
import pytest
from conftest import assistant_text, echo_tool

from harness.acp.bridge import AsyncBridge
from harness.acp.server import HarnessAgent
from harness.core.types import TextContent
from harness.session.store import SessionStore


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
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kw):
        self.updates.append(update)

    async def request_permission(self, options, session_id, tool_call, **kw):
        return acp.schema.RequestPermissionResponse(
            outcome=acp.schema.AllowedOutcome(outcome="selected", option_id="allow_once")
        )


def _run(coro, loop):
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=5)


def _streaming_provider(script):
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


def _dangling(messages):
    result_ids = {
        b.get("tool_use_id")
        for m in messages if m.get("role") == "user"
        for b in (m.get("content") or []) if isinstance(b, dict) and b.get("type") == "tool_result"
    }
    return [
        b.get("id")
        for m in messages if m.get("role") == "assistant"
        for b in (m.get("content") or []) if isinstance(b, dict) and b.get("type") == "tool_use"
        and b.get("id") not in result_ids
    ]


def _agent(loop, store, script):
    a = HarnessAgent(
        provider=_streaming_provider(script), tools=[echo_tool()],
        bridge=AsyncBridge(loop), store=store, system="You are a coding agent.",
    )
    a.on_connect(FakeClient())
    return a


def test_acp_session_persists_and_resumes_in_a_fresh_agent():
    with LoopThread() as loop, tempfile.TemporaryDirectory() as d:
        store = SessionStore(os.path.join(d, "s.db"))

        a1 = _agent(loop, store, [assistant_text("hi there")])
        sid = _run(a1.new_session(cwd="/tmp"), loop).session_id
        _run(a1.prompt([acp.text_block("hello")], sid), loop)
        assert len(store.load_messages(sid)) == 2  # user + assistant persisted

        # a FRESH process: new agent, same store, no in-memory carryover
        a2 = _agent(loop, store, [assistant_text("welcome back")])
        assert sid not in a2._sessions
        _run(a2.load_session(cwd="/tmp", session_id=sid), loop)
        assert sid in a2._sessions
        assert len(a2._sessions[sid].messages) == 2          # history restored
        assert a2._sessions[sid].system == "You are a coding agent."  # system byte-for-byte

        _run(a2.prompt([acp.text_block("more")], sid), loop)
        assert len(store.load_messages(sid)) == 4            # conversation continued durably


def test_acp_load_unknown_session_is_rejected():
    with LoopThread() as loop, tempfile.TemporaryDirectory() as d:
        store = SessionStore(os.path.join(d, "s.db"))
        a = _agent(loop, store, [])
        with pytest.raises(acp.RequestError):
            _run(a.load_session(cwd="/tmp", session_id="sess-does-not-exist"), loop)


def test_acp_load_without_store_is_rejected():
    with LoopThread() as loop:
        a = HarnessAgent(provider=_streaming_provider([]), tools=[echo_tool()], bridge=AsyncBridge(loop))
        a.on_connect(FakeClient())
        with pytest.raises(acp.RequestError):
            _run(a.load_session(cwd="/tmp", session_id="whatever"), loop)


def test_acp_resume_reconciles_an_interrupted_turn():
    with LoopThread() as loop, tempfile.TemporaryDirectory() as d:
        store = SessionStore(os.path.join(d, "s.db"))
        # craft a crashed session directly: user + assistant(tool_use), no tool_result
        store.create_session("crashed", "You are a coding agent.")
        store.append_message("crashed", {"role": "user", "content": [{"type": "text", "text": "delete tmp"}]})
        store.append_message("crashed", {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_Z", "name": "echo", "input": {}},
        ]})
        assert _dangling(store.load_messages("crashed")) == ["toolu_Z"]  # gap present

        a = _agent(loop, store, [assistant_text("ok")])
        _run(a.load_session(cwd="/tmp", session_id="crashed"), loop)
        _run(a.prompt([acp.text_block("did it work?")], "crashed"), loop)

        # the orphan was reconciled and persisted — a third restart loads clean
        assert _dangling(store.load_messages("crashed")) == []


def test_acp_prompt_wires_tool_timeout_into_agent():
    """The ACP server's prompt() passes resolve_tool_timeout() into AgentConfig — the
    second shipping driver, proven by probe (not just asserted from reading the code)."""
    from unittest.mock import patch
    from harness.core.loop import Agent

    old = os.environ.pop("HARNESS_TOOL_TIMEOUT", None)
    try:
        captured = {}
        original_init = Agent.__init__

        def spy_init(self, config):
            captured["tool_timeout"] = config.tool_timeout
            original_init(self, config)

        with LoopThread() as loop, tempfile.TemporaryDirectory() as d:
            store = SessionStore(os.path.join(d, "s.db"))
            a = _agent(loop, store, [assistant_text("hi")])
            sid = _run(a.new_session(cwd="/tmp"), loop).session_id
            with patch.object(Agent, "__init__", spy_init):
                _run(a.prompt([acp.text_block("go")], sid), loop)
        assert captured["tool_timeout"] == 900.0
    finally:
        if old is not None:
            os.environ["HARNESS_TOOL_TIMEOUT"] = old
