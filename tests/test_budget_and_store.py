"""Iteration budget, code-only-turn refund, and the SQLite session store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from conftest import FakeProvider, assistant_text, assistant_tool_use, echo_tool, tool_use

from harness.core.budget import IterationBudget
from harness.core.loop import Agent, AgentConfig
from harness.hooks.cost import CostTracker
from harness.core.types import Usage
from harness.session import SessionStore


def test_budget_consume_exhaust_grace():
    b = IterationBudget(max_iterations=2)
    b.consume()
    assert not b.exhausted
    b.consume()
    assert b.exhausted
    assert b.take_grace() is True
    assert b.take_grace() is False
    b.refund()
    assert not b.exhausted


def test_code_only_turn_is_refunded():
    bash_like = echo_tool("runner", parallel_safe=False, tags=("execute",))
    prov = FakeProvider(
        [assistant_tool_use(tool_use("runner", cmd="ls")), assistant_text("fin")]
    )
    agent = Agent(AgentConfig(provider=prov, tools=[bash_like]))
    agent.run([{"role": "user", "content": [{"type": "text", "text": "ls"}]}])
    # 2 model turns, the execute-only turn refunded -> used == 1
    assert agent.budget.used == 1


def test_non_code_turn_not_refunded():
    # tool is NOT execute-tagged -> no refund
    prov = FakeProvider([assistant_tool_use(tool_use("echo", x=1)), assistant_text("fin")])
    agent = Agent(AgentConfig(provider=prov, tools=[echo_tool()]))
    agent.run([{"role": "user", "content": [{"type": "text", "text": "go"}]}])
    assert agent.budget.used == 2


def test_store_uses_wal(tmp_path: Path):
    db = tmp_path / "s.db"
    SessionStore(db)
    mode = sqlite3.connect(db).execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_store_system_roundtrip_and_messages(tmp_path: Path):
    store = SessionStore(tmp_path / "s.db")
    store.create_session("s1", "SYS", model="claude-opus-4-8")
    assert store.get_system("s1") == "SYS"
    assert store.get_system("missing") is None

    persist = store.persist_fn("s1")
    persist({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    persist({"role": "assistant", "content": [{"type": "text", "text": "yo"}]})
    msgs = store.load_messages("s1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"][0]["text"] == "yo"


def test_cost_tracker_pricing_and_cache_discount():
    ct = CostTracker()
    ct.record("claude-opus-4-8", Usage(input_tokens=1_000_000))
    assert abs(ct.usd - 5.0) < 1e-9
    ct.record("claude-opus-4-8", Usage(output_tokens=1_000_000))
    assert abs(ct.usd - 30.0) < 1e-9
    cached = CostTracker()
    cached.record("claude-opus-4-8", Usage(cache_read_tokens=1_000_000))
    assert abs(cached.usd - 0.5) < 1e-9  # 0.1x input price
