"""MCP client bridge — the architecture-validation suite.

Proves external MCP tools drop into the registry through the three seams with NO
core change: a scripted in-process fake MCP server (no subprocess), offline like
FakeProvider. The end-to-end test runs an MCP tool through the real Agent.run().
"""

from __future__ import annotations

import asyncio
import re

import pytest
from conftest import FakeProvider, assistant_text, assistant_tool_use, tool_use

from harness.core.loop import Agent, AgentConfig
from harness.mcp.client import McpCallResult, McpRuntime, McpToolSpec
from harness.mcp.registry import load_mcp_tools, mcp_tools, namespaced


class FakeMcpClient:
    """Scripted MCP server: fixed tool specs + call results, with injectable
    connect / call failures to exercise isolation paths."""

    def __init__(self, specs, *, connect_error=None, call_error=None, results=None):
        self._specs = specs
        self._connect_error = connect_error
        self._call_error = call_error
        self._results = results or {}
        self.connected = False
        self.closed = False
        self.call_count = 0

    def connect(self, timeout):
        if self._connect_error is not None:
            raise self._connect_error
        self.connected = True

    def list_tools(self):
        return list(self._specs)

    def call_tool(self, name, arguments):
        self.call_count += 1
        if self._call_error is not None:
            raise self._call_error
        return self._results.get(name, McpCallResult(text=f"{name}:{sorted(arguments.items())}"))

    def close(self):
        self.closed = True


def _spec(name="echo", read_only=False):
    return McpToolSpec(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        read_only=read_only,
    )


# --------------------------------------------------------------------------- #
# Conversion: declarations frozen, schema preserved, namespaced
# --------------------------------------------------------------------------- #


_VALID_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def test_namespacing():
    assert namespaced("git", "status") == "mcp__git__status"
    assert namespaced("my server!", "x") == "mcp__my_server__x"  # sanitized


def test_namespaced_name_always_valid_for_pathological_inputs():
    cases = [
        ("git", "status"),            # normal
        ("srv", "search/files"),      # illegal char in tool half
        ("srv", "x" * 70),            # over-long tool half
        ("srv", "naïve tool"),        # non-ASCII (isalnum() lies — must be stripped)
        ("a" * 80, "y"),              # over-long server half
        ("s", "@@@"),                 # all-illegal tool half -> fallback
    ]
    for server, tool in cases:
        name = namespaced(server, tool)
        assert _VALID_TOOL_NAME.match(name), f"{server!r},{tool!r} -> {name!r}"


def test_sanitize_collision_resolved_distinct_and_valid():
    # both sanitize to mcp__srv__get_thing; must become two DISTINCT valid names
    client = FakeMcpClient([_spec("get.thing"), _spec("get/thing")])
    tools = mcp_tools(client, "srv")
    names = [t.name for t in tools]
    assert len(set(names)) == 2  # collision resolved
    assert all(_VALID_TOOL_NAME.match(n) for n in names)
    assert all(n.startswith("mcp__srv__get_thing") for n in names)


def test_namespacing_is_deterministic_across_independent_runs():
    # resume/cache stability: identical inputs -> identical names, regardless of run
    def build():
        c = FakeMcpClient([_spec("get.thing"), _spec("get/thing"), _spec("plain")])
        return [t.name for t in mcp_tools(c, "srv")]

    assert build() == build()


def test_conversion_preserves_schema_and_namespaces():
    client = FakeMcpClient([_spec("read_file")])
    tools = mcp_tools(client, "fs")
    assert len(tools) == 1
    t = tools[0]
    assert t.name == "mcp__fs__read_file"
    assert t.input_schema["properties"]["x"]["type"] == "string"  # JSON Schema intact
    assert t.handler is not None  # closure, NOT None (see registry docstring finding)
    assert "mcp" in t.tags


def test_read_only_tools_not_gated_mutating_gated():
    client = FakeMcpClient([_spec("search", read_only=True), _spec("write", read_only=False)])
    by_name = {t.name: t for t in mcp_tools(client, "s")}
    assert by_name["mcp__s__search"].requires_approval is False
    assert by_name["mcp__s__write"].requires_approval is True


def test_no_mcp_round_trip_until_invoked():
    # the "lazy half": building the Tool must not call the server
    client = FakeMcpClient([_spec("echo")])
    tools = mcp_tools(client, "srv")
    assert client.call_count == 0
    tools[0].handler(x="hi")  # first invocation
    assert client.call_count == 1


# --------------------------------------------------------------------------- #
# Execution + the invariant
# --------------------------------------------------------------------------- #


def test_call_success_returns_tool_result():
    client = FakeMcpClient([_spec("echo")], results={"echo": McpCallResult(text="pong")})
    tool = mcp_tools(client, "srv")[0]
    result = tool.handler(x="ping")
    assert result.content[0].text == "pong" and result.is_error is False


def test_mcp_call_error_becomes_error_result_not_raise():
    client = FakeMcpClient([_spec("echo")], call_error=RuntimeError("server boom"))
    tool = mcp_tools(client, "srv")[0]
    result = tool.handler(x="ping")  # must not raise
    assert result.is_error is True and "server boom" in result.content[0].text


def test_mcp_reported_error_flag_propagates():
    client = FakeMcpClient(
        [_spec("echo")], results={"echo": McpCallResult(text="bad input", is_error=True)}
    )
    tool = mcp_tools(client, "srv")[0]
    assert tool.handler(x="ping").is_error is True


# --------------------------------------------------------------------------- #
# Failure isolation at startup (the loader)
# --------------------------------------------------------------------------- #


def _factory(clients_by_name):
    return lambda name, cfg: clients_by_name[name]


def test_cross_server_name_collision_resolved():
    # two server names that sanitize alike -> tools collide ACROSS servers; the
    # loader's cross-server pass must split them (per-server dedup can't see this).
    a = FakeMcpClient([_spec("x")])
    b = FakeMcpClient([_spec("x")])
    tools, _, _ = load_mcp_tools(
        {"my.server": {"command": "p"}, "my/server": {"command": "q"}},
        client_factory=_factory({"my.server": a, "my/server": b}),
    )
    names = [t.name for t in tools]
    assert len(set(names)) == 2  # both -> mcp__my_server__x base, resolved distinct
    assert all(_VALID_TOOL_NAME.match(n) for n in names)
    assert all(n.startswith("mcp__my_server__x") for n in names)


def test_server_down_at_startup_is_dropped_others_proceed():
    good = FakeMcpClient([_spec("ok")])
    bad = FakeMcpClient([_spec("never")], connect_error=ConnectionError("refused"))
    tools, clients, _ = load_mcp_tools(
        {"good": {"command": "x"}, "bad": {"command": "y"}},
        client_factory=_factory({"good": good, "bad": bad}),
    )
    names = [t.name for t in tools]
    assert names == ["mcp__good__ok"]  # bad server contributed nothing
    assert good in clients and bad not in clients
    assert bad.closed is True  # dropped server was closed


def test_slow_server_times_out_and_is_dropped():
    slow = FakeMcpClient([_spec("late")], connect_error=TimeoutError("too slow"))
    tools, clients, _ = load_mcp_tools(
        {"slow": {"command": "z"}},
        client_factory=_factory({"slow": slow}),
        connect_timeout=0.01,
    )
    assert tools == [] and clients == []


def test_empty_config_is_noop():
    tools, clients, runtime = load_mcp_tools({})
    assert tools == [] and clients == [] and runtime is None


# --------------------------------------------------------------------------- #
# Deterministic frozen set (cache-prefix stability)
# --------------------------------------------------------------------------- #


def test_frozen_set_is_deterministic_for_same_config():
    cfg = {"a": {"command": "x"}, "b": {"command": "y"}}

    def build():
        a = FakeMcpClient([_spec("one"), _spec("two")])
        b = FakeMcpClient([_spec("three")])
        tools, _, _ = load_mcp_tools(cfg, client_factory=_factory({"a": a, "b": b}))
        return [t.name for t in tools]

    assert build() == build()  # same config -> same declaration order -> same prefix
    assert build() == ["mcp__a__one", "mcp__a__two", "mcp__b__three"]


# --------------------------------------------------------------------------- #
# End-to-end: an MCP tool through the real loop, invariant holds
# --------------------------------------------------------------------------- #


def test_pathological_tool_name_flows_through_agent_run():
    # The conversation-wide-blast-radius regression: a tool named `search/files`
    # must produce a VALID frozen declaration and run end-to-end (no 400), with the
    # handler still dispatching the ORIGINAL name to the server.
    weird = "search/files"
    client = FakeMcpClient([_spec(weird)], results={weird: McpCallResult(text="from-mcp")})
    tools, _, _ = load_mcp_tools(
        {"srv": {"command": "x"}}, client_factory=_factory({"srv": client})
    )
    name = tools[0].name
    assert _VALID_TOOL_NAME.match(name) and name == "mcp__srv__search_files"

    prov = FakeProvider([assistant_tool_use(tool_use(name, x="hi")), assistant_text("done")])
    msgs = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    from harness.hooks import from_tools as approval_from_tools

    Agent(AgentConfig(
        provider=prov, tools=tools, hooks=[approval_from_tools(tools, approver=lambda n, a: True)]
    )).run(msgs)

    results = [
        b for m in msgs if m["role"] == "user"
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert len(results) == 1 and results[0]["is_error"] is False
    assert "from-mcp" in results[0]["content"][0]["text"]
    assert client.call_count == 1  # dispatched the original "search/files"


def test_runtime_call_enforces_timeout_on_hung_server():
    # A wedged MCP call must NOT hang the agent thread forever: the runtime times
    # out and raises, which _make_handler turns into an error_result (invariant).
    rt = McpRuntime()
    try:
        async def hang():
            await asyncio.sleep(5)

        with pytest.raises(TimeoutError):
            rt.call(hang(), timeout=0.05)
    finally:
        rt.shutdown()


def test_mcp_tool_flows_through_agent_run():
    client = FakeMcpClient([_spec("echo")], results={"echo": McpCallResult(text="from-mcp")})
    tools, _, _ = load_mcp_tools({"srv": {"command": "x"}}, client_factory=_factory({"srv": client}))
    tool_name = tools[0].name  # mcp__srv__echo

    prov = FakeProvider(
        [assistant_tool_use(tool_use(tool_name, x="hi")), assistant_text("done")]
    )
    msgs = [{"role": "user", "content": [{"type": "text", "text": "use the mcp tool"}]}]
    # read_only=False -> gated; auto-allow so the call goes through
    from harness.hooks import from_tools as approval_from_tools

    Agent(AgentConfig(
        provider=prov, tools=tools, hooks=[approval_from_tools(tools, approver=lambda n, a: True)]
    )).run(msgs)

    results = [
        b for m in msgs if m["role"] == "user"
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert len(results) == 1 and results[0]["is_error"] is False
    assert "from-mcp" in results[0]["content"][0]["text"]
    assert client.call_count == 1
