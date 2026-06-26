"""Adversarial hardening — Bucket C: provider + I/O fault injection.

Fail-closed at the provider/stream boundary: a fault RAISES and the loop appends
NOTHING corrupt — history stays at the pre-turn state (reconcilable), never a
half-written turn or a silent swallow.

Coverage map (rung 2 — REUSE, don't rewrite; these faults are already sealed):
  - disk-full / WAL exhaustion on write  -> Bucket B
      test_hardening_resume.py::test_persist_failure_on_assistant_turn_is_fail_closed
      test_hardening_resume.py::test_store_append_propagates_write_failure_no_silent_drop
  - stdin EOF -> clean exit               -> test_acp_server.py::test_main_entry_serves_then_exits_on_stdin_eof
  - provider stream raises (disconnect)   -> test_hardening_lifecycle.py::test_provider_disconnect_aborts_cleanly_with_no_orphan

NEW here: 500 at request time, fault MID-stream (after partial deltas), malformed
final payload (normalizer fail-closed), stream-cancel severs the connection, and a
broken pipe on the write side cannot wedge or corrupt the loop (fire-and-forget emit).

Honest non-faults (NOT testable as a "raise" at our layer — documented, not faked):
  - "duplicate network responses": the loop calls provider.stream() exactly once per
    turn and appends exactly one Response — there is no path that ingests a response
    twice. HTTP request/response is 1:1; dedup is the SDK/transport's concern.
  - "kill -9": SIGKILL is uncatchable — it raises nothing. Its *recovery* is the
    Bucket B crash-resume reconciliation (already proven). Nothing to assert here.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from conftest import assistant_text, echo_tool

from harness.core.loop import Agent, AgentConfig, reconcile_dangling_tool_calls
from harness.core.types import CancelToken
from harness.providers.claude import CancelledError, ClaudeProvider

USER = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]


def _msgs():
    return [dict(m) for m in USER]


def _delta(text):
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)
    )


def _good_final(text="ok"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    )


class _Stream:
    """Context-manager stream: yields `events`, optionally raises mid-iter or in
    get_final_message; records close()."""

    def __init__(self, events, final=None, iter_raises=None, final_raises=None):
        self._events, self._final = events, final
        self._iter_raises, self._final_raises = iter_raises, final_raises
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for e in self._events:
            yield e
        if self._iter_raises is not None:
            raise self._iter_raises

    def get_final_message(self):
        if self._final_raises is not None:
            raise self._final_raises
        return self._final

    def close(self):
        self.closed = True


class _Client:
    """Shape-matches the anthropic client: `client.messages.stream(**kwargs)`."""

    def __init__(self, stream=None, stream_raises=None):
        self._stream, self._stream_raises = stream, stream_raises
        self.messages = self

    def stream(self, **kwargs):
        if self._stream_raises is not None:
            raise self._stream_raises
        return self._stream


def _provider(client):
    return ClaudeProvider(model="claude-opus-4-8", api_key="x", client=client)


# ---- 1. Provider network faults --------------------------------------------
def test_provider_500_at_request_propagates_no_corrupt_history():
    class APIStatus500(Exception):
        ...

    n0 = threading.active_count()
    prov = _provider(_Client(stream_raises=APIStatus500("HTTP 500")))
    msgs = _msgs()
    with pytest.raises(APIStatus500):
        Agent(AgentConfig(provider=prov, tools=[])).run(msgs)
    assert [m["role"] for m in msgs] == ["user"]      # nothing appended
    assert reconcile_dangling_tool_calls(msgs) == 0   # no orphan to reconcile
    assert threading.active_count() == n0             # no zombie thread


def test_provider_fault_mid_stream_propagates_no_corrupt_history():
    """A fault AFTER partial deltas: the deltas flowed to on_chunk, but the turn is
    NOT committed (no half-written assistant turn in canonical history)."""
    class MidStreamReset(Exception):
        ...

    stream = _Stream([_delta("partial ")], iter_raises=MidStreamReset("reset mid-stream"))
    prov = _provider(_Client(stream=stream))
    seen: list[str] = []
    msgs = _msgs()
    with pytest.raises(MidStreamReset):
        Agent(AgentConfig(provider=prov, tools=[], on_chunk=seen.append)).run(msgs)
    assert seen == ["partial "]                        # partial delta did stream out
    assert [m["role"] for m in msgs] == ["user"]       # but the turn was NOT committed
    assert reconcile_dangling_tool_calls(msgs) == 0


def test_malformed_final_payload_fails_closed_in_normalizer():
    """A structurally malformed final message (no `.content`) makes the normalizer
    raise — the loop appends nothing, history stays clean."""
    bad_final = SimpleNamespace(stop_reason="end_turn")  # missing .content / .usage
    prov = _provider(_Client(stream=_Stream([], final=bad_final)))
    msgs = _msgs()
    with pytest.raises(AttributeError):
        Agent(AgentConfig(provider=prov, tools=[])).run(msgs)
    assert [m["role"] for m in msgs] == ["user"]


# ---- 2. Stream faults ------------------------------------------------------
def test_stream_cancel_severs_the_connection_synchronously():
    """A slow/stuck stream is severed via the cancel token: the provider closes the
    stream and raises CancelledError, stopping at the cancel point. The stream loop is
    SYNCHRONOUS (no worker thread), so a clean sever leaves no zombie."""
    token = CancelToken()

    class CancelMidStream:
        def __init__(self):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            yield _delta("x")
            token.set()          # trip cancel before the next event is processed
            yield _delta("y")

        def get_final_message(self):
            return _good_final()

        def close(self):
            self.closed = True

    stream = CancelMidStream()
    prov = _provider(_Client(stream=stream))
    chunks: list[str] = []
    n0 = threading.active_count()
    with pytest.raises(CancelledError):
        prov.stream("sys", [{"role": "user", "content": "hi"}], [], cancel=token, on_chunk=chunks.append)
    assert stream.closed is True                       # connection severed
    assert chunks == ["x"]                             # stopped at the cancel point
    assert threading.active_count() == n0              # no zombie thread


# ---- 3. I/O fault: broken pipe on the write side ---------------------------
def test_broken_pipe_on_session_update_does_not_wedge_or_corrupt():
    """`session_update` chunks go through `bridge.emit` (fire-and-forget; the
    run_coroutine_threadsafe future is never retrieved — bridge.py:45-48). So a broken
    pipe on a chunk WRITE is isolated to its own task: the worker (agent.run) is
    untouched, the prompt completes, the session stays intact. Parent-death teardown is
    the read side (stdin EOF), proven in test_acp_server::test_main_entry_*."""
    import asyncio

    import acp

    from harness.acp.bridge import AsyncBridge
    from harness.acp.server import HarnessAgent
    from harness.core.types import TextContent

    class _StreamingProv:
        profile = None
        hooks = None

        def __init__(self, resp):
            self._resp = resp

        def stream(self, system, messages, tools, cancel=None, on_chunk=None):
            if on_chunk:
                for b in self._resp.content:
                    if isinstance(b, TextContent):
                        on_chunk(b.text)
            return self._resp

    class _BrokenPipeClient:
        async def session_update(self, session_id, update, **kw):
            raise BrokenPipeError("editor pipe closed")

        async def request_permission(self, options, session_id, tool_call, **kw):
            raise AssertionError("no permission expected for this tool")

    loop = asyncio.new_event_loop()
    th = threading.Thread(target=loop.run_forever, daemon=True)
    th.start()
    try:
        agent = HarnessAgent(
            provider=_StreamingProv(assistant_text("hello despite the broken pipe")),
            tools=[echo_tool()],
            bridge=AsyncBridge(loop),
        )
        agent.on_connect(_BrokenPipeClient())
        sid = asyncio.run_coroutine_threadsafe(agent.new_session(cwd="/tmp"), loop).result(5).session_id
        resp = asyncio.run_coroutine_threadsafe(agent.prompt([acp.text_block("hi")], sid), loop).result(5)
        assert resp.stop_reason == "end_turn"                 # prompt completed despite write failures
        sess = agent._sessions[sid]
        assert any(m["role"] == "assistant" for m in sess.messages)  # session intact, not corrupt
    finally:
        loop.call_soon_threadsafe(loop.stop)
        th.join(timeout=2)
        loop.close()
