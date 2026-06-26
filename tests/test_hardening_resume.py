"""Adversarial hardening — Bucket B: the tool_result invariant across a crash/resume.

These are ACCEPTANCE tests for the cross-restart reconciliation (not an audit of prior
behavior — before this work they failed; see learning-log L9). The hazard: the loop
persists the assistant tool_use turn BEFORE running tools and the tool_result turn
AFTER, so a crash between the two writes leaves a dangling tool_use that would 400 the
resumed request. `reconcile_dangling_tool_calls` folds synthetic 'interrupted' results
in — idempotently, re-executing nothing.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile

import pytest

from harness.core.loop import Agent, AgentConfig, reconcile_dangling_tool_calls
from harness.core.types import TextContent, Tool, ToolResult
from harness.session.store import SessionStore
from harness.testing import FakeProvider, assistant_text, assistant_tool_use, tool_use


def _dangling(messages):
    result_ids = {
        b.get("tool_use_id")
        for m in messages if m.get("role") == "user"
        for b in (m.get("content") or []) if isinstance(b, dict) and b.get("type") == "tool_result"
    }
    use_ids = [
        b.get("id")
        for m in messages if m.get("role") == "assistant"
        for b in (m.get("content") or []) if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    return [u for u in use_ids if u not in result_ids]


def _has_consecutive_same_role(messages):
    return any(messages[i].get("role") == messages[i + 1].get("role") for i in range(len(messages) - 1))


# ---- the core reconciliation -----------------------------------------------
def test_reconcile_fixes_dangling_tool_use_at_tail():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "delete tmp"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "running"},
            {"type": "tool_use", "id": "toolu_ABC", "name": "bash", "input": {"cmd": "rm -rf tmp"}},
        ]},
    ]  # crashed before the tool_result turn
    n = reconcile_dangling_tool_calls(messages)
    assert n == 1
    assert _dangling(messages) == []
    last = messages[-1]
    assert last["role"] == "user"
    tr = last["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "toolu_ABC" and tr["is_error"] is True
    assert not _has_consecutive_same_role(messages)


def test_reconcile_is_idempotent():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]},
    ]
    assert reconcile_dangling_tool_calls(messages) == 1
    snap = json.dumps(messages)
    assert reconcile_dangling_tool_calls(messages) == 0  # nothing left to fix
    assert json.dumps(messages) == snap  # unchanged on the second pass


def test_reconcile_noop_on_completed_history():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},  # end_turn, no tool_use
    ]
    snap = json.dumps(messages)
    assert reconcile_dangling_tool_calls(messages) == 0
    assert json.dumps(messages) == snap


def test_reconcile_folds_into_following_user_prompt():
    """Resume order: the new user prompt is already appended after the dangling
    assistant. Synthetic results prepend INTO that prompt (tool_result then text) —
    valid + alternating, no extra message."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "delete tmp"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_X", "name": "bash", "input": {}}]},
        {"role": "user", "content": [{"type": "text", "text": "did it work?"}]},  # next prompt
    ]
    n = reconcile_dangling_tool_calls(messages)
    assert n == 1
    assert _dangling(messages) == []
    assert len(messages) == 3  # folded, not inserted
    blocks = messages[-1]["content"]
    assert blocks[0]["type"] == "tool_result" and blocks[0]["tool_use_id"] == "toolu_X"
    assert blocks[-1]["type"] == "text" and blocks[-1]["text"] == "did it work?"
    assert not _has_consecutive_same_role(messages)


# ---- end-to-end via the SQLite store (the real CLI resume flow) -------------
def test_crash_resume_via_store_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        store = SessionStore(os.path.join(d, "s.db"))
        store.create_session("sess", "sys")
        store.append_message("sess", {"role": "user", "content": [{"type": "text", "text": "delete tmp"}]})
        store.append_message("sess", {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_ABC", "name": "bash", "input": {"cmd": "rm -rf tmp"}},
        ]})  # <- SIGKILL: tool_result never written

        # resume exactly as cli.run does
        messages = store.load_messages("sess")
        assert _dangling(messages) == ["toolu_ABC"]  # the gap exists on load
        messages.append({"role": "user", "content": [{"type": "text", "text": "did it work?"}]})
        reconcile_dangling_tool_calls(messages)
        store.append_message("sess", messages[-1])

        assert _dangling(messages) == []
        assert not _has_consecutive_same_role(messages)

        # a SECOND restart loads a clean, already-reconciled history (idempotent)
        reloaded = store.load_messages("sess")
        assert _dangling(reloaded) == []
        assert reconcile_dangling_tool_calls(reloaded) == 0


# ---- run() is a defensive net regardless of caller -------------------------
def test_run_never_sends_an_orphan_to_the_provider():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "x"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_Q", "name": "bash", "input": {}}]},
        {"role": "user", "content": [{"type": "text", "text": "continue"}]},
    ]
    prov = FakeProvider([assistant_text("ok")])
    Agent(AgentConfig(provider=prov, tools=[], system="s")).run(messages)
    assert _dangling(prov.seen_messages[0]) == []   # provider never saw the orphan
    assert _dangling(messages) == []                # canonical fixed in place


def test_interrupted_tool_is_not_re_executed_on_resume():
    calls = {"n": 0}

    def handler(cancel=None, **kwargs):
        calls["n"] += 1
        return ToolResult(content=[TextContent("ran")])

    bash = Tool(name="bash", description="bash", input_schema={"type": "object"}, handler=handler)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "delete tmp"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_R", "name": "bash", "input": {"cmd": "rm -rf tmp"}}]},
        {"role": "user", "content": [{"type": "text", "text": "did it work?"}]},
    ]
    Agent(AgentConfig(provider=FakeProvider([assistant_text("ok")]), tools=[bash], system="s")).run(messages)
    assert calls["n"] == 0                 # the interrupted call is NOT re-run (no dup side effects)
    assert _dangling(messages) == []        # but the invariant is satisfied via the synthetic result


# ---- fail-closed persistence (disk-full / WAL exhaustion class) -------------
def test_persist_failure_on_assistant_turn_is_fail_closed():
    """A write failure when persisting the assistant tool_use turn must (a) propagate
    loudly, (b) NOT execute tools (persist-before-execute), (c) leave in-memory history
    un-advanced (no store/memory divergence)."""
    ran = {"v": False}

    def handler(cancel=None, **kw):
        ran["v"] = True
        return ToolResult(content=[TextContent("ran")])

    def boom(_msg):
        raise sqlite3.OperationalError("database or disk is full")

    bash = Tool(name="bash", description="bash", input_schema={"type": "object"}, handler=handler)
    messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    with pytest.raises(sqlite3.OperationalError):
        Agent(AgentConfig(
            provider=FakeProvider([assistant_tool_use(tool_use("bash"))]),
            tools=[bash], system="s", persist=boom,
        )).run(messages)
    assert ran["v"] is False                                  # tools never executed
    assert all(m.get("role") != "assistant" for m in messages)  # memory did not advance


def test_persist_failure_on_tool_result_leaves_reconcilable_state():
    """If the tool_result write fails AFTER the tool ran, the persisted store keeps the
    assistant tool_use turn (a dangling tool_use) — recoverable on resume, never a
    corrupt/partial record."""
    persisted: list[dict] = []
    ran = {"v": False}

    def handler(cancel=None, **kw):
        ran["v"] = True
        return ToolResult(content=[TextContent("ran")])

    def persist(msg):
        if msg.get("role") == "user" and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in msg.get("content", [])
        ):
            raise sqlite3.OperationalError("disk full")
        persisted.append(msg)

    bash = Tool(name="bash", description="bash", input_schema={"type": "object"}, handler=handler)
    messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    with pytest.raises(sqlite3.OperationalError):
        Agent(AgentConfig(
            provider=FakeProvider([assistant_tool_use(tool_use("bash")), assistant_text("done")]),
            tools=[bash], system="s", persist=persist,
        )).run(messages)
    assert ran["v"] is True                                   # the tool DID run
    assert _dangling(persisted) == ["id_bash_0"]              # store left dangling, not corrupt
    assert reconcile_dangling_tool_calls(persisted) == 1      # and it is reconcilable on resume


def test_store_append_propagates_write_failure_no_silent_drop():
    """The store surfaces a write failure (it never silently swallows) and leaves no
    partial row. We force the failure with PRAGMA query_only (the same OperationalError
    class as disk-full), armed only after schema/session creation."""
    class GatedStore(SessionStore):
        block_writes = False

        def _connect(self):
            conn = super()._connect()
            if self.block_writes:
                conn.execute("PRAGMA query_only=ON")
            return conn

    with tempfile.TemporaryDirectory() as d:
        store = GatedStore(os.path.join(d, "s.db"))
        store.create_session("s", "sys")
        store.block_writes = True
        with pytest.raises(sqlite3.OperationalError):
            store.append_message("s", {"role": "user", "content": [{"type": "text", "text": "x"}]})
        store.block_writes = False
        assert store.load_messages("s") == []  # no partial row persisted
