"""MCP client integration — consume external MCP servers' tools.

A *client* of MCP servers, never the harness being one (see client.py docstring).
"""

from harness.mcp.client import (
    McpCallResult,
    McpClient,
    McpRuntime,
    McpToolSpec,
    StdioMcpClient,
    StdioServerConfig,
)
from harness.mcp.registry import load_mcp_config, load_mcp_tools, mcp_tools, namespaced

__all__ = [
    "McpCallResult",
    "McpClient",
    "McpRuntime",
    "McpToolSpec",
    "StdioMcpClient",
    "StdioServerConfig",
    "load_mcp_config",
    "load_mcp_tools",
    "mcp_tools",
    "namespaced",
]
