"""MCP → harness tool registry.

This is the architecture-validation surface: MCP server tools must drop into the
harness through the SAME three seams as a native tool, with NO core change.

Finding (worth stating): the design note "declarations frozen, handlers lazy" is
realized here as a **closure handler**, not a `handler=None` placeholder. A None
handler would require the loop to grow a lazy-resolution step — i.e. a CORE
change, which is exactly what the narrow-waist test forbids. So the closure is not
a shortcut; it's the proof. The frozen half (schemas fetched once at startup, hashed
into the cache prefix) and the lazy half (no MCP round-trip until the model calls the
tool) are both satisfied without touching `core/`.

Failure isolation mirrors the LSP "auto-blacklist broken servers" gotcha: a server
that won't connect / list in time is dropped with a WARN; the conversation proceeds
with the remaining tools. A `tools/call` error becomes an `error_result`, the same
as a native tool that raises — the tool_result invariant is untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from harness.core.types import TextContent, Tool, ToolResult
from harness.mcp.client import McpClient, McpRuntime, StdioMcpClient, StdioServerConfig

log = logging.getLogger("harness.mcp")

DEFAULT_CONNECT_TIMEOUT = 10.0
MAX_TOOL_NAME = 64  # Anthropic: tool name must match ^[a-zA-Z0-9_-]{1,64}$


def load_mcp_config(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read an `mcpServers`-shaped JSON file (Claude Desktop / OpenClaw format):
    `{"mcpServers": {"<name>": {"command": "...", "args": [...], "env": {...}}}}`.
    Returns the inner server map (accepts a bare map too)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    servers = data.get("mcpServers", data) if isinstance(data, dict) else {}
    return {k: v for k, v in servers.items() if isinstance(v, dict) and "command" in v}


def _sanitize(s: str) -> str:
    """Collapse anything outside Anthropic's tool-name charset to a single '_',
    trimming edges so we never emit a leading/trailing/doubled '_'. ASCII-only:
    `str.isalnum()` is True for non-ASCII letters (e.g. 'ï'), which are NOT in
    `[A-Za-z0-9_-]` and would 400 — so gate on `c.isascii()` too."""
    out: list[str] = []
    for c in s:
        if c.isascii() and (c.isalnum() or c == "-"):
            out.append(c)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_")


def namespaced(server: str, tool: str) -> str:
    """`mcp__<server>__<tool>`, GUARANTEED to satisfy `^[a-zA-Z0-9_-]{1,64}$`.

    BOTH halves are sanitized: an unsanitized tool name (`search/files`, a 70-char
    name, a unicode name) would 400 the ENTIRE conversation, because the tool name
    rides in the frozen `tools` prefix sent on every call — and unlike a bad
    `tools/list`, a bad *name* passes listing cleanly and only detonates at the API
    boundary, so the drop-the-server net never catches it. The FULL composed string
    is capped at 64; the tool half is truncated to fit (the server half is the
    namespace, worth preserving). Pure function of (server, tool); collisions from
    the many-to-one sanitize/truncate are resolved separately by `_resolve_collisions`."""
    s = _sanitize(server) or "server"
    t = _sanitize(tool) or "tool"
    prefix = f"mcp__{s}__"
    budget = MAX_TOOL_NAME - len(prefix)
    if budget < 1:
        # pathological: the server half alone blows the budget — cap it, but keep
        # room for at least one tool char so the namespace stays meaningful.
        s = s[: MAX_TOOL_NAME - len("mcp____") - 1].strip("_") or "server"
        prefix = f"mcp__{s}__"
        budget = MAX_TOOL_NAME - len(prefix)
    return f"{prefix}{t[:budget].rstrip('_') or 'tool'}"


def _suffix(name: str, seed: str) -> str:
    """Append a short, deterministic hash of `seed` (the ORIGINAL, pre-sanitization
    identity), re-capping to 64. Determined by `seed` ALONE — never iteration order
    or an enumerate index — so the frozen set is byte-stable across a resume."""
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:6]
    keep = MAX_TOOL_NAME - len(h) - 1
    return f"{name[:keep].rstrip('_')}_{h}"


def _resolve_collisions(names: list[str], seeds: list[str]) -> list[str]:
    """Make `names` unique WITHOUT depending on order: every member of a colliding
    group gets a per-seed suffix (singletons keep their clean name). Same inputs ->
    same outputs regardless of how a server enumerated its tools — a duplicate name
    in `tools` is itself a conversation-wide 400, so this must be deterministic."""
    counts = Counter(names)
    return [_suffix(n, s) if counts[n] > 1 else n for n, s in zip(names, seeds)]


def _make_handler(client: McpClient, tool_name: str) -> Callable[..., ToolResult]:
    def handler(cancel: Any = None, **kwargs: Any) -> ToolResult:
        # No MCP round-trip happens until this runs (the lazy half). A failure
        # never propagates — it becomes an error_result so the invariant holds.
        try:
            res = client.call_tool(tool_name, kwargs)
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                content=[TextContent(f"MCP call failed: {type(e).__name__}: {e}")], is_error=True
            )
        return ToolResult(content=[TextContent(res.text)], is_error=res.is_error)

    return handler


def mcp_tools(client: McpClient, server_name: str, *, namespace: bool = True) -> list[Tool]:
    """Convert one connected MCP server's tools into frozen `Tool`s. Raises if
    `list_tools()` fails — the loader (`load_mcp_tools`) isolates that."""
    specs = list(client.list_tools())
    if namespace:
        bases = [namespaced(server_name, s.name) for s in specs]
        # seed by the ORIGINAL tool name so `get.thing` vs `get/thing` (both ->
        # `mcp__srv__get_thing`) resolve to distinct, deterministic final names.
        names = _resolve_collisions(bases, [s.name for s in specs])
    else:
        names = [s.name for s in specs]
    tools: list[Tool] = []
    for spec, name in zip(specs, names):
        tools.append(
            Tool(
                name=name,
                description=spec.description,
                input_schema=spec.input_schema,
                handler=_make_handler(client, spec.name),  # dispatches the ORIGINAL name
                parallel_safe=True,
                # read-only MCP tools aren't gated; mutating/unknown ones are
                requires_approval=not spec.read_only,
                tags=("mcp",) + (("read",) if spec.read_only else ()),
            )
        )
    return tools


ClientFactory = Callable[[str, dict[str, Any]], McpClient]


def load_mcp_tools(
    servers: dict[str, dict[str, Any]],
    *,
    runtime: McpRuntime | None = None,
    client_factory: ClientFactory | None = None,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
) -> tuple[list[Tool], list[McpClient], McpRuntime | None]:
    """Build frozen MCP tools from an `mcpServers`-shaped config (the same JSON
    Claude Desktop / OpenClaw use, so users paste existing configs).

    Failure-isolated: a server that won't connect or list within `connect_timeout`
    is dropped with a WARN; remaining servers still contribute their tools. Returns
    (tools, live_clients, runtime). The frozen set is deterministic for a given
    config (server insertion order → tool order) so the cache prefix is stable.
    """
    if not servers:
        return [], [], runtime

    own_runtime = runtime is None and client_factory is None
    if client_factory is None:
        runtime = runtime or McpRuntime()

        def client_factory(name: str, cfg: dict[str, Any]) -> McpClient:  # type: ignore[misc]
            return StdioMcpClient(
                runtime,  # type: ignore[arg-type]
                StdioServerConfig(
                    command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env")
                ),
            )

    tools: list[Tool] = []
    origins: list[str] = []  # server name per tool, parallel to `tools`
    clients: list[McpClient] = []
    for name, cfg in servers.items():
        client = client_factory(name, cfg)
        try:
            client.connect(connect_timeout)
            server_tools = mcp_tools(client, name)
        except Exception as e:  # noqa: BLE001 — one bad server can't down the harness
            log.warning("MCP server %r dropped: %s: %s", name, type(e).__name__, e)
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
            continue
        tools.extend(server_tools)
        origins.extend([name] * len(server_tools))
        clients.append(client)

    # Cross-server collision pass. Per-server names are already unique, so any name
    # shared across the flat set is a CROSS-server clash (two server names that
    # sanitize alike). Resolve deterministically, seeded by server name — a
    # duplicate in the frozen `tools` prefix is a conversation-wide 400.
    if tools:
        resolved = _resolve_collisions([t.name for t in tools], origins)
        for t, final in zip(tools, resolved):
            if t.name != final:
                log.warning("MCP cross-server tool-name collision: %r -> %r", t.name, final)
                t.name = final

    # Only hand back a runtime the caller must shut down — i.e. one WE created.
    # If the caller passed their own in, its lifecycle is theirs; return None so
    # teardown ownership is unambiguous.
    return tools, clients, (runtime if own_runtime else None)
