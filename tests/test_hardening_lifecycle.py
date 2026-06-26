"""Adversarial hardening — Bucket A: tool-lifecycle invariants under fault.

Proves ∀ tool_use(id): exactly one tool_result(id), and that the loop never hangs or
crashes, under each adversarial condition. Coverage map (7 conditions):

  1. Timeout                 -> here: bash self-limit + loop backstop (serial/parallel)
  2. Mid-batch cancellation  -> test_invariant.py::test_mid_batch_cancel_fills_remaining_results
                                + here: cancel DURING tool execution
  3. Hook denial             -> test_invariant.py::test_denial_still_emits_result
  4. Tool exception          -> test_invariant.py::test_raising_tool_emits_error_not_crash
                                + here: isolation in a PARALLEL batch
  5. Process SIGINT          -> here: _install_sigint wires SIGINT -> cancel (cooperative)
  6. Provider disconnect     -> here: stream raises -> clean abort, no orphan
  7. Shutdown during exec    -> Bucket B (crash-resume reconcile); graceful = #2/#5
"""
from __future__ import annotations

import io
import signal
import sys
import threading

import pytest
from conftest import FakeProvider, assistant_text, assistant_tool_use, echo_tool, tool_results, tool_use

from harness.core.loop import Agent, AgentConfig, reconcile_dangling_tool_calls
from harness.core.types import CancelToken, TextContent, Tool, ToolResult

USER = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]


def _msgs():
    return [dict(m) for m in USER]


# ---- 1. Timeout ------------------------------------------------------------
def test_bash_tool_self_limits_on_timeout():
    """The real bash tool kills a hanging command via its subprocess timeout and
    returns an error result — no hang, no orphan."""
    from harness.tools.bash import bash_handler

    cmd = f'"{sys.executable}" -c "import time; time.sleep(5)"'
    res = bash_handler(command=cmd, timeout=200)  # 200 ms
    assert res.is_error
    assert "timed out" in res.content[0].text


def test_timeout_backstop_serial_hung_tool_holds_invariant():
    """A non-cooperative tool that hangs (ignores cancel, no internal timeout) is
    bounded by the loop backstop: a timeout error_result is emitted, the loop returns,
    the invariant holds."""
    release = threading.Event()

    def hang(cancel=None, **kw):
        release.wait(10)  # ignores `cancel` on purpose
        return ToolResult(content=[TextContent("late")])

    hung = Tool(name="hung", description="x", input_schema={"type": "object"},
                handler=hang, parallel_safe=False, execution_mode="sequential")
    prov = FakeProvider([assistant_tool_use(tool_use("hung")), assistant_text("done")])
    msgs = _msgs()
    try:
        Agent(AgentConfig(provider=prov, tools=[hung], tool_timeout=0.2)).run(msgs)
        results = tool_results(msgs)
        assert len(results) == 1 and results[0]["is_error"] is True
        assert "timed out" in results[0]["content"][0]["text"]
    finally:
        release.set()  # let the leaked worker exit cleanly


def test_timeout_backstop_parallel_mixes_ok_and_timeout():
    """In a parallel batch, a hung call times out while a fast call still succeeds —
    one result each, in order."""
    release = threading.Event()

    def hang(cancel=None, **kw):
        release.wait(10)
        return ToolResult(content=[TextContent("late")])

    hung = Tool(name="hung", description="x", input_schema={"type": "object"},
                handler=hang, parallel_safe=True)
    prov = FakeProvider(
        [assistant_tool_use(tool_use("ok"), tool_use("hung")), assistant_text("done")]
    )
    msgs = _msgs()
    try:
        Agent(AgentConfig(provider=prov, tools=[echo_tool("ok"), hung], tool_timeout=0.3)).run(msgs)
        results = tool_results(msgs)
        assert len(results) == 2
        assert any(not r["is_error"] for r in results)                          # ok succeeded
        assert any("timed out" in r["content"][0]["text"] for r in results)     # hung timed out
    finally:
        release.set()


# ---- 2. Cancellation during execution --------------------------------------
def test_cancel_during_tool_execution_holds_invariant():
    """A cooperative long-running tool sees the cancel set mid-execution, returns, and
    the turn closes with exactly one tool_result."""
    started = threading.Event()

    def coop(cancel=None, **kw):
        started.set()
        while cancel is not None and not cancel.is_set():
            cancel.wait(0.01)
        return ToolResult(content=[TextContent("stopped")])

    tool = Tool(name="coop", description="x", input_schema={"type": "object"},
                handler=coop, parallel_safe=False, execution_mode="sequential")
    prov = FakeProvider([assistant_tool_use(tool_use("coop")), assistant_text("done")])
    msgs = _msgs()
    token = CancelToken()
    th = threading.Thread(
        target=lambda: Agent(AgentConfig(provider=prov, tools=[tool])).run(msgs, cancel=token)
    )
    th.start()
    assert started.wait(2), "tool never started"
    token.set()
    th.join(5)
    assert not th.is_alive()
    assert len(tool_results(msgs)) == 1


# ---- 4. Tool exception isolated in a parallel batch ------------------------
def test_parallel_batch_isolates_a_raising_tool():
    def boom(cancel=None, **kw):
        raise RuntimeError("kaboom")

    raiser = Tool(name="boom", description="x", input_schema={"type": "object"},
                  handler=boom, parallel_safe=True)
    prov = FakeProvider(
        [assistant_tool_use(tool_use("ok"), tool_use("boom")), assistant_text("done")]
    )
    msgs = _msgs()
    Agent(AgentConfig(provider=prov, tools=[echo_tool("ok"), raiser])).run(msgs)
    results = tool_results(msgs)
    assert len(results) == 2
    assert sorted(r["is_error"] for r in results) == [False, True]
    assert any("kaboom" in r["content"][0]["text"] for r in results)


# ---- 5. Process SIGINT -> cooperative cancel -------------------------------
def test_sigint_handler_sets_cancel_token():
    """The CLI's Ctrl-C handler requests cooperative cancellation (first press), which
    drives the same invariant-preserving unwind as #2."""
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signals are only settable on the main thread")
    from harness.cli import _install_sigint

    original = signal.getsignal(signal.SIGINT)
    try:
        token = CancelToken()
        err = io.StringIO()
        _install_sigint(token, err)
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)  # simulate first Ctrl-C
        assert token.is_set()
        assert "cancel" in err.getvalue().lower()
    finally:
        signal.signal(signal.SIGINT, original)


# ---- 6. Provider disconnect ------------------------------------------------
def test_provider_disconnect_aborts_cleanly_with_no_orphan():
    """If the provider stream raises (connection reset) the turn aborts before any
    tool_use is recorded — the exception propagates and the history has no orphan."""
    class Disconnecting:
        profile = None
        hooks = None

        def stream(self, system, messages, tools, cancel=None):
            raise ConnectionError("provider connection reset")

    msgs = _msgs()
    with pytest.raises(ConnectionError):
        Agent(AgentConfig(provider=Disconnecting(), tools=[echo_tool()])).run(msgs)
    assert [m["role"] for m in msgs] == ["user"]      # nothing partial appended
    assert reconcile_dangling_tool_calls(msgs) == 0   # and nothing to reconcile


# ---- 7. resolve_tool_timeout wiring -----------------------------------------

def test_resolve_tool_timeout_default():
    """With no env override, resolve_tool_timeout returns the 900s default."""
    from harness.core.loop import resolve_tool_timeout
    import os
    env = os.environ.pop("HARNESS_TOOL_TIMEOUT", None)
    try:
        assert resolve_tool_timeout() == 900.0
    finally:
        if env is not None:
            os.environ["HARNESS_TOOL_TIMEOUT"] = env


def test_resolve_tool_timeout_custom():
    """HARNESS_TOOL_TIMEOUT overrides the default."""
    from harness.core.loop import resolve_tool_timeout
    import os
    old = os.environ.get("HARNESS_TOOL_TIMEOUT")
    os.environ["HARNESS_TOOL_TIMEOUT"] = "300"
    try:
        assert resolve_tool_timeout() == 300.0
    finally:
        if old is None:
            os.environ.pop("HARNESS_TOOL_TIMEOUT", None)
        else:
            os.environ["HARNESS_TOOL_TIMEOUT"] = old


def test_resolve_tool_timeout_zero_disables():
    """HARNESS_TOOL_TIMEOUT=0 disables the backstop (returns None)."""
    from harness.core.loop import resolve_tool_timeout
    import os
    old = os.environ.get("HARNESS_TOOL_TIMEOUT")
    os.environ["HARNESS_TOOL_TIMEOUT"] = "0"
    try:
        assert resolve_tool_timeout() is None
    finally:
        if old is None:
            os.environ.pop("HARNESS_TOOL_TIMEOUT", None)
        else:
            os.environ["HARNESS_TOOL_TIMEOUT"] = old


def test_resolve_tool_timeout_invalid_falls_back_to_default():
    """Fail-closed: an UNPARSABLE HARNESS_TOOL_TIMEOUT (a typo like '900s') keeps the
    production backstop at its default — it must NOT silently disable the floor. Only an
    explicit numeric <= 0 disables (see test_resolve_tool_timeout_zero_disables)."""
    from harness.core.loop import resolve_tool_timeout
    import os
    old = os.environ.get("HARNESS_TOOL_TIMEOUT")
    os.environ["HARNESS_TOOL_TIMEOUT"] = "900s"
    try:
        assert resolve_tool_timeout() == 900.0
    finally:
        if old is None:
            os.environ.pop("HARNESS_TOOL_TIMEOUT", None)
        else:
            os.environ["HARNESS_TOOL_TIMEOUT"] = old


def test_serial_timeout_fail_stops_remaining_sequential_calls():
    """A serial tool that times out must NOT let the next sequential tool start — its
    (non-killable) thread may still be mutating state. The remaining serial calls are
    skipped with a synthetic result: invariant preserved AND sequential isolation held.
    (Regression guard for the Finding-1 overlap bug.)"""
    first_in = threading.Event()
    release = threading.Event()
    state = {"first_done": False, "second_ran_while_first_alive": False}

    def hang(cancel=None, **kw):
        first_in.set()
        release.wait(10)  # ignores cancel on purpose
        state["first_done"] = True
        return ToolResult(content=[TextContent("late")])

    def second(cancel=None, **kw):
        if not state["first_done"]:
            state["second_ran_while_first_alive"] = True
        return ToolResult(content=[TextContent("second ran")])

    t1 = Tool(name="t1", description="x", input_schema={"type": "object"},
              handler=hang, parallel_safe=False, execution_mode="sequential")
    t2 = Tool(name="t2", description="x", input_schema={"type": "object"},
              handler=second, parallel_safe=False, execution_mode="sequential")
    prov = FakeProvider(
        [assistant_tool_use(tool_use("t1"), tool_use("t2")), assistant_text("done")]
    )
    msgs = _msgs()
    try:
        Agent(AgentConfig(provider=prov, tools=[t1, t2], tool_timeout=0.2)).run(msgs)
        results = tool_results(msgs)
        assert len(results) == 2                                    # invariant: one each
        assert results[0]["is_error"] and "timed out" in results[0]["content"][0]["text"]
        assert results[1]["is_error"] and "skipped" in results[1]["content"][0]["text"]
        assert state["second_ran_while_first_alive"] is False       # isolation held
    finally:
        release.set()


def test_cli_wires_tool_timeout_into_agent():
    """The CLI's run() passes resolve_tool_timeout() into AgentConfig — proving the
    shipping driver actually uses the backstop."""
    import argparse, os, tempfile
    from unittest.mock import patch
    from harness.core.loop import resolve_tool_timeout

    old = os.environ.pop("HARNESS_TOOL_TIMEOUT", None)
    try:
        captured = {}
        original_init = Agent.__init__

        def spy_init(self, config):
            captured["tool_timeout"] = config.tool_timeout
            original_init(self, config)

        prov = FakeProvider([assistant_text("hi")])
        from harness.session.store import SessionStore
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(os.path.join(td, "s.db"))

            args = argparse.Namespace(
                prompt="test", session_id=None, model="opus", system=None,
                max_iterations=1, yes=True, no_color=True, db=os.path.join(td, "s.db"),
                mcp=None,
            )
            with patch.object(Agent, "__init__", spy_init):
                from harness.cli import run
                run(args, provider=prov, store=store, install_signals=False,
                    interactive=False, tools=[echo_tool()])

            assert captured["tool_timeout"] == 900.0
    finally:
        if old is not None:
            os.environ["HARNESS_TOOL_TIMEOUT"] = old
