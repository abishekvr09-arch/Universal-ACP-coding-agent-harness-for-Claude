"""Context compression (Law 3). The pair-protection rule is the never-regress one."""

from __future__ import annotations

from pathlib import Path

from harness.core.context import (
    CompressionPolicy,
    Compressor,
    is_safe_split_point,
    turn_boundaries,
)
from harness.core.types import Response, TextContent, Usage
from harness.session import SessionStore


# --------------------------------------------------------------------------- #
# History builders
# --------------------------------------------------------------------------- #


def u(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def a_text(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def a_tool(name: str, tid: str) -> dict:
    return {"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": name, "input": {}}]}


def u_result(tid: str, text: str = "ok") -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": [{"type": "text", "text": text}]}],
    }


def sample_history() -> list[dict]:
    # exchange 1 (with a tool pair), exchange 2 (with a tool pair), exchange 3 (text)
    return [
        u("task one"),                 # 0
        a_tool("read", "t1"),          # 1
        u_result("t1", "file body"),   # 2
        a_text("read done"),           # 3
        u("task two"),                 # 4
        a_tool("bash", "t2"),          # 5
        u_result("t2", "ran"),         # 6
        a_text("bash done"),           # 7
        u("task three"),               # 8
        a_text("final answer"),        # 9
    ]


class FakeSummary:
    profile = None
    hooks = None

    def __init__(self, text: str = "SUMMARY") -> None:
        self.text = text
        self.calls = 0

    def stream(self, system, messages, tools, cancel=None) -> Response:
        self.calls += 1
        return Response(content=[TextContent(self.text)], tool_calls=[], stop_reason="end_turn")


def resp(input_tokens: int) -> Response:
    return Response(content=[], tool_calls=[], stop_reason="end_turn", usage=Usage(input_tokens=input_tokens))


def assert_no_orphaned_pairs(messages: list[dict]) -> None:
    for i, m in enumerate(messages):
        has_tr = m.get("role") == "user" and any(
            b.get("type") == "tool_result" for b in m.get("content", []) if isinstance(b, dict)
        )
        if has_tr:
            prev = messages[i - 1] if i > 0 else {}
            assert prev.get("role") == "assistant" and any(
                b.get("type") == "tool_use" for b in prev.get("content", []) if isinstance(b, dict)
            ), f"orphaned tool_result at {i}"


def assert_alternates(messages: list[dict]) -> None:
    for i in range(1, len(messages)):
        assert messages[i]["role"] != messages[i - 1]["role"], f"role collision at {i}"


# --------------------------------------------------------------------------- #
# The split invariant
# --------------------------------------------------------------------------- #


def test_unsafe_split_between_tool_use_and_result():
    h = sample_history()
    assert is_safe_split_point(h, 2) is False  # cuts t1 tool_use(1) from result(2)
    assert is_safe_split_point(h, 6) is False  # cuts t2 tool_use(5) from result(6)


def test_safe_split_on_exchange_boundaries():
    h = sample_history()
    assert is_safe_split_point(h, 0) is True
    assert is_safe_split_point(h, 4) is True
    assert is_safe_split_point(h, 8) is True
    assert is_safe_split_point(h, len(h)) is True


def test_turn_boundaries():
    assert turn_boundaries(sample_history()) == [0, 4, 8, 10]


# --------------------------------------------------------------------------- #
# Gating on REAL tokens, not estimates
# --------------------------------------------------------------------------- #


def test_should_compress_uses_real_tokens():
    c = Compressor(FakeSummary())  # default window 200k, ratio 0.6 -> 120k
    assert c.should_compress(120_001) is True
    assert c.should_compress(120_000) is False


def test_long_history_under_token_threshold_is_not_compressed():
    # A long message list but the API says input_tokens is tiny -> no compression.
    c = Compressor(FakeSummary(), CompressionPolicy(context_window=200_000, trigger_ratio=0.6))
    h = sample_history() * 5
    n = len(h)
    assert c.compress_if_needed(h, resp(input_tokens=10)) is False
    assert len(h) == n  # untouched


# --------------------------------------------------------------------------- #
# Actual compression
# --------------------------------------------------------------------------- #


def _small_policy(**kw) -> CompressionPolicy:
    base = dict(context_window=100, trigger_ratio=0.1, tail_ratio=0.15, head_protect_turns=1)
    base.update(kw)
    return CompressionPolicy(**base)


def test_compression_rewrites_and_preserves_invariants():
    fake = FakeSummary("DENSE SUMMARY")
    c = Compressor(fake, _small_policy())
    h = sample_history()
    before = len(h)
    changed = c.compress_if_needed(h, resp(input_tokens=50))  # 50 > 0.1*100
    assert changed is True
    assert fake.calls == 1
    assert len(h) < before
    # the summary made it into history
    flat = " ".join(
        b.get("text", "") for m in h for b in m.get("content", []) if isinstance(b, dict)
    )
    assert "DENSE SUMMARY" in flat
    # the load-bearing properties
    assert_no_orphaned_pairs(h)
    assert_alternates(h)
    assert h[0]["role"] == "user"  # valid opening for the Messages API


def test_summary_failure_leaves_history_untouched():
    class Boom(FakeSummary):
        def stream(self, *a, **k):
            raise RuntimeError("haiku down")

    c = Compressor(Boom(), _small_policy())
    h = sample_history()
    snapshot = [dict(m) for m in h]
    assert c.compress_if_needed(h, resp(input_tokens=50)) is False
    assert h == snapshot


# --------------------------------------------------------------------------- #
# Anti-thrash + head-protection decay
# --------------------------------------------------------------------------- #


def test_anti_thrash_skips_after_two_small_saves():
    c = Compressor(FakeSummary(), _small_policy(min_save_ratio=0.5))
    # pre-seed two tiny saves on the in-memory state
    c._mem["s"] = [0.05, 0.02]
    h = sample_history()
    assert c.compress_if_needed(h, resp(input_tokens=50), session_id="s") is False


def test_head_protection_decays_with_prior_compactions(tmp_path: Path):
    store = SessionStore(tmp_path / "s.db")
    store.create_session("s", "SYS")
    # simulate 5 prior big-save compactions -> effective head protect decays to 0
    for _ in range(5):
        store.record_compaction("s", before_tokens=1000, after_tokens=100)
    count, saves = store.compaction_stats("s")
    assert count == 5 and all(sv > 0.5 for sv in saves)

    c = Compressor(FakeSummary(), _small_policy(head_protect_turns=1))
    plan = c._plan_cut(sample_history(), prior_compactions=count)
    assert plan is not None
    head_end, _ = plan
    assert head_end == 0  # fully decayed: nothing protected at the head


def test_store_backed_anti_thrash(tmp_path: Path):
    store = SessionStore(tmp_path / "s.db")
    store.create_session("s", "SYS")
    store.record_compaction("s", before_tokens=1000, after_tokens=980)  # 2% save
    store.record_compaction("s", before_tokens=1000, after_tokens=985)  # 1.5% save
    c = Compressor(FakeSummary(), _small_policy(min_save_ratio=0.10))
    h = sample_history()
    assert c.compress_if_needed(h, resp(input_tokens=50), store=store, session_id="s") is False


# --------------------------------------------------------------------------- #
# End-to-end: real loop → compress mid-session → resume on compressed history
# --------------------------------------------------------------------------- #


def test_compression_in_a_live_loop_preserves_invariant():
    from conftest import (
        FakeProvider,
        assistant_text,
        assistant_tool_use,
        echo_tool,
        tool_use,
    )

    from harness.core.loop import Agent, AgentConfig

    tools = [echo_tool()]
    msgs = [u("one")]
    Agent(AgentConfig(
        provider=FakeProvider([assistant_tool_use(tool_use("echo", n=1)), assistant_text("done one")]),
        tools=tools,
    )).run(msgs)
    msgs.append(u("two"))
    Agent(AgentConfig(
        provider=FakeProvider([assistant_tool_use(tool_use("echo", n=2)), assistant_text("done two")]),
        tools=tools,
    )).run(msgs)
    msgs.append(u("three"))
    Agent(AgentConfig(provider=FakeProvider([assistant_text("done three")]), tools=tools)).run(msgs)

    n_before = len(msgs)
    comp = Compressor(FakeSummary("SUMMARY OF EARLIER"), _small_policy())
    assert comp.compress_if_needed(msgs, resp(input_tokens=80)) is True
    assert len(msgs) < n_before
    assert_no_orphaned_pairs(msgs)
    assert_alternates(msgs)
    assert msgs[0]["role"] == "user"

    # resume the loop on the COMPRESSED history — provider must get a clean list
    msgs.append(u("four"))
    prov = FakeProvider([assistant_text("done four")])
    final = Agent(AgentConfig(provider=prov, tools=tools)).run(msgs)
    assert final.stop_reason == "end_turn" and "four" in final.content[0].text
    flat = " ".join(
        b.get("text", "")
        for m in prov.seen_messages[0]
        for b in (m.get("content") or [])
        if isinstance(b, dict)
    )
    assert "SUMMARY OF EARLIER" in flat
