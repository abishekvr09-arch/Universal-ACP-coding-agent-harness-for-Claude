"""A minimal real MCP server over stdio, for the one live integration test.

Run as: python tests/_mcp_echo_server.py  (speaks MCP on stdin/stdout).
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-server")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the input back, prefixed."""
    return f"echo: {text}"


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    mcp.run()
