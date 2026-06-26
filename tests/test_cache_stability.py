"""Cache-prefix stability invariants — the audited guarantees (CLAUDE.md Gotcha 17).

These lock down the 7 properties a stable prompt-cache prefix depends on. They are
EMPIRICAL: each runs the real loop / provider / store code and inspects bytes, not
docstrings. If one fails, a cache-key regression (silent cost multiplier) has landed.

The shallow-copy contract (learning-log L8): the loop's per-turn view is a shallow
copy, so ephemeral injectors must REASSIGN content, never mutate it in place. The
characterization test `test_inplace_mutation_leaks_contract` documents *why* the
contract exists; if it ever fails because in-place mutation no longer leaks, the loop
adopted a deeper copy — update Gotcha 17 / L8 rather than just deleting the test.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile

from harness.core.loop import Agent, AgentConfig
from harness.providers.cache import apply_cache_breakpoints, mark_cache_breakpoints, mark_system
from harness.providers.claude import to_anthropic_tools
from harness.session.store import SessionStore
from harness.testing import FakeProvider, assistant_text, echo_tool
from harness.tools import default_tools

EPHEMERAL = {"type": "ephemeral"}


# ---- 1. Prefix immutability ------------------------------------------------
def test_full_turn_does_not_leak_cache_or_ephemeral_into_canonical():
    canonical = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    before = copy.deepcopy(canonical)

    def inject(api_msgs):  # SAFE pattern: reassign content on the copy
        api_msgs[-1] = dict(api_msgs[-1])
        api_msgs[-1]["content"] = [{"type": "text", "text": "RAG"}] + list(api_msgs[-1]["content"])

    Agent(AgentConfig(
        provider=FakeProvider([assistant_text("done")]),
        tools=[], system="sys", inject_ephemeral=inject,
    )).run(canonical)

    assert canonical[0] == before[0]  # original user turn untouched
    blob = json.dumps(canonical)
    assert "cache_control" not in blob
    assert "RAG" not in blob


def test_apply_cache_breakpoints_deepcopies_source():
    src = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    snap = copy.deepcopy(src)
    out = apply_cache_breakpoints(src)
    assert src == snap  # source never mutated
    assert out[-1]["content"][-1].get("cache_control") == EPHEMERAL  # output marked


def test_inplace_mutation_is_isolated():
    """L8 ELEVATED: the loop hands hooks/injectors a DEEP copy, so even an in-place
    mutation of a nested content list cannot leak into canonical history. Isolation is
    now enforced structurally, not a documented contract. (If this regresses — leak
    reappears — the loop stopped deep-copying per turn: see CLAUDE.md Gotcha 17 / L8.)"""
    canonical = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    before = copy.deepcopy(canonical)

    def inject_inplace(api_msgs):  # the OLD footgun — now harmless
        api_msgs[-1]["content"].append({"type": "text", "text": "MUTATION"})

    Agent(AgentConfig(
        provider=FakeProvider([assistant_text("d")]),
        tools=[], system="s", inject_ephemeral=inject_inplace,
    )).run(canonical)
    assert canonical[0] == before[0]                 # user turn byte-identical
    assert "MUTATION" not in json.dumps(canonical)   # nothing leaked


def test_before_model_inplace_mutation_is_isolated():
    """The deep-copy boundary covers hooks too: a before_model that mutates a nested
    content list in place cannot reach canonical history."""
    canonical = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    before = copy.deepcopy(canonical)

    class MutatingHook:
        def before_model(self, messages):
            messages[-1]["content"].append({"type": "text", "text": "HOOKMUT"})
            return messages

    Agent(AgentConfig(
        provider=FakeProvider([assistant_text("ok")]),
        tools=[], system="s", hooks=[MutatingHook()],
    )).run(canonical)
    assert canonical[0] == before[0]
    assert "HOOKMUT" not in json.dumps(canonical)


def test_mark_cache_breakpoints_is_in_place():
    """The loop's marker mutates in place and returns the same object (one deepcopy
    per turn lives in run(), not here)."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    out = mark_cache_breakpoints(msgs)
    assert out is msgs  # same object, marked in place
    assert msgs[-1]["content"][-1].get("cache_control") == EPHEMERAL


# ---- 2. Hook mutation boundary ---------------------------------------------
def test_before_model_changes_stay_in_ephemeral_copy():
    canonical = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]

    class GuardHook:
        def before_model(self, messages):
            return [{"role": "user", "content": [{"type": "text", "text": "GUARDRAIL"}]}] + messages

    prov = FakeProvider([assistant_text("ok")])
    Agent(AgentConfig(provider=prov, tools=[], system="s", hooks=[GuardHook()])).run(canonical)

    assert any("GUARDRAIL" in json.dumps(m) for m in prov.seen_messages[0])  # provider saw it
    assert not any("GUARDRAIL" in json.dumps(m) for m in canonical)  # canonical clean


# ---- 3. No dynamic (timestamp) injection into the prefix -------------------
def test_prefix_has_no_clock_derived_bytes():
    tools = default_tools()
    sys_a, sys_b = mark_system("You are a coding agent."), mark_system("You are a coding agent.")
    tools_a, tools_b = to_anthropic_tools(tools), to_anthropic_tools(tools)
    assert json.dumps(sys_a) == json.dumps(sys_b)
    assert json.dumps(tools_a) == json.dumps(tools_b)


# ---- 4. Input ordering determinism -----------------------------------------
def test_tool_order_is_list_driven_and_stable_across_rebuilds():
    order1 = [t["name"] for t in to_anthropic_tools(default_tools())]
    order2 = [t["name"] for t in to_anthropic_tools(default_tools())]  # a 'resume'
    assert order1 == order2 == ["read", "edit", "bash", "glob", "grep"]


# ---- 5. Map serialization stability (insertion-order, not sort_keys) -------
def test_schema_bytes_stable_and_store_roundtrip_identical():
    assert json.dumps(to_anthropic_tools(default_tools())) == json.dumps(
        to_anthropic_tools(default_tools())
    )
    with tempfile.TemporaryDirectory() as d:
        store = SessionStore(os.path.join(d, "t.db"))
        store.create_session("s", "sys")
        msg = {"role": "assistant", "content": [
            {"type": "text", "text": "a"},
            {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "x", "limit": 5}},
        ]}
        store.append_message("s", msg)
        loaded = store.load_messages("s")
    # byte-identical round-trip => insertion order preserved end-to-end (key order
    # is load-bearing; the harness does NOT normalize via sort_keys on this path)
    assert json.dumps(loaded[0]["content"]) == json.dumps(msg["content"])
    assert json.dumps({"path": "x", "limit": 5}) != json.dumps({"limit": 5, "path": "x"})


# ---- 6. No whitespace canonicalization (byte-for-byte passthrough) ---------
def test_system_passthrough_preserves_whitespace_verbatim():
    weird = "line1\r\n  indented\ttab   \n\n"  # CRLF + trailing ws + blank lines
    assert mark_system(weird)[0]["text"] == weird  # no strip / CRLF-normalize


# ---- 7. Schema not regenerated by handler invocation -----------------------
def test_handler_invocation_does_not_mutate_schema():
    tools = default_tools()
    before = json.dumps(to_anthropic_tools(tools))
    read_tool = next(t for t in tools if t.name == "read")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("hello\nworld\n")
        path = f.name
    try:
        read_tool.handler(path=path)
    finally:
        os.unlink(path)
    assert json.dumps(to_anthropic_tools(tools)) == before
    # passed by reference (no copy/regeneration of the frozen schema)
    assert to_anthropic_tools(tools)[0]["input_schema"] is tools[0].input_schema
