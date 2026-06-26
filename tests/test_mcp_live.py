"""One LIVE integration test: real StdioMcpClient + McpRuntime + a real MCP server
subprocess. Proves the async stack (shared loop, bridge, session lifecycle) works
end-to-end, not just against fakes. Skipped if the server can't be spawned (the
real path is environment-sensitive, like the SIGINT-subprocess test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from harness.mcp.client import McpRuntime, StdioMcpClient, StdioServerConfig
from harness.mcp.registry import mcp_tools

SERVER = str(Path(__file__).parent / "_mcp_echo_server.py")


@pytest.fixture
def live_client():
    runtime = McpRuntime()
    client = StdioMcpClient(runtime, StdioServerConfig(command=sys.executable, args=[SERVER]))
    try:
        client.connect(timeout=20)
    except Exception as e:  # noqa: BLE001
        client.close()
        runtime.shutdown()
        pytest.skip(f"could not spawn live MCP server: {e}")
    yield client
    client.close()
    runtime.shutdown()


def test_live_list_tools(live_client):
    specs = {s.name: s for s in live_client.list_tools()}
    assert "echo" in specs and "add" in specs
    assert specs["echo"].input_schema["properties"]["text"]["type"] == "string"


def test_live_call_tool(live_client):
    result = live_client.call_tool("echo", {"text": "harness"})
    assert result.is_error is False and "echo: harness" in result.text
    added = live_client.call_tool("add", {"a": 2, "b": 40})
    assert "42" in added.text


def test_live_tools_convert_to_frozen_harness_tools(live_client):
    tools = {t.name: t for t in mcp_tools(live_client, "echo")}
    assert "mcp__echo__echo" in tools and "mcp__echo__add" in tools
    # invoke through the harness Tool handler (the closure → real server round-trip)
    out = tools["mcp__echo__echo"].handler(text="live")
    assert out.is_error is False and "echo: live" in out.content[0].text
