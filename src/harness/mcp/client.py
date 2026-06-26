"""MCP client — speak to external MCP servers and expose their tools.

SCOPE: this module is strictly a *client* of external MCP servers (it consumes
their `tools/list` + `tools/call`). It is NOT the harness *being* an MCP server —
that's a different seam, and ACP already covers "something drives the harness".

Architecture: an MCP server is an async JSON-RPC peer (stdio here). Our loop is
sync + threads. We run ONE shared asyncio event loop on a dedicated thread
(`McpRuntime`), keep each server's session open on it for the whole conversation,
and marshal calls across with `AsyncBridge.call` (block the agent's worker thread
for the result) — the exact primitive the ACP work already built.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from harness.acp.bridge import AsyncBridge


@dataclass
class McpToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool = False  # from MCP annotations.readOnlyHint — gates approval


@dataclass
class McpCallResult:
    text: str
    is_error: bool = False


@runtime_checkable
class McpClient(Protocol):
    """The sync surface the registry depends on. The real client hides all async
    behind the shared runtime; tests provide a plain sync fake."""

    def connect(self, timeout: float) -> None: ...
    def list_tools(self) -> list[McpToolSpec]: ...
    def call_tool(self, name: str, arguments: dict[str, Any]) -> McpCallResult: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Shared runtime — one event loop + thread for ALL MCP servers
# --------------------------------------------------------------------------- #


class McpRuntime:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="harness-mcp"
        )
        self._thread.start()
        self._bridge = AsyncBridge(self._loop)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def call(self, coro: Any, timeout: float | None = None) -> Any:
        """Block the calling (worker) thread for an async result on the loop. On
        timeout, CANCEL the scheduled coroutine so a hung server doesn't keep
        running on the loop after we've given up. The TimeoutError then surfaces
        as an error_result at the tool boundary, so the invariant holds even when
        a server wedges mid-call."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout)
        except TimeoutError:
            fut.cancel()
            raise

    def spawn(self, coro: Any) -> None:
        """Schedule a long-lived coroutine on the loop, fire-and-forget."""
        self._bridge.emit(coro)

    def shutdown(self) -> None:
        """Cancel in-flight coroutines and AWAIT their unwind before stopping the
        loop — avoids 'Task was destroyed but it is pending' and orphaned child I/O
        when a session ends with a call still timing out."""

        async def _drain() -> None:
            pending = [
                t for t in asyncio.all_tasks(self._loop) if t is not asyncio.current_task()
            ]
            for t in pending:
                t.cancel()
            for t in pending:
                try:
                    await t
                except BaseException:  # noqa: BLE001 — tearing down; swallow CancelledError et al.
                    pass

        try:
            asyncio.run_coroutine_threadsafe(_drain(), self._loop).result(timeout=2)
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)


# --------------------------------------------------------------------------- #
# Real stdio client
# --------------------------------------------------------------------------- #


@dataclass
class StdioServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


class StdioMcpClient:
    """A persistent connection to one stdio MCP server, driven on the shared loop.

    The session's async context managers must stay open for the connection's
    life, so a single long-lived `_serve` coroutine holds them open on the loop
    and waits on a shutdown event; calls are dispatched against the live session.
    """

    def __init__(
        self, runtime: McpRuntime, config: StdioServerConfig, *, call_timeout: float = 30.0
    ) -> None:
        self._runtime = runtime
        self._config = config
        self._call_timeout = call_timeout
        self._session: Any = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._shutdown: asyncio.Event | None = None

    def connect(self, timeout: float) -> None:
        self._runtime.spawn(self._serve())
        if not self._ready.wait(timeout):
            raise TimeoutError(f"MCP server '{self._config.command}' did not start in {timeout}s")
        if self._session is None:
            raise ConnectionError(
                f"MCP server '{self._config.command}' failed to connect: {self._error}"
            )

    async def _serve(self) -> None:
        self._shutdown = asyncio.Event()
        try:
            # Imports live INSIDE the try: a missing `mcp` (the documented default —
            # it's a manual install) must surface to connect() as a fast ConnectionError,
            # not stall it for the full timeout. The except below sets _ready.
            from contextlib import AsyncExitStack

            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=self._config.command, args=self._config.args, env=self._config.env
            )
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._session = session
                self._ready.set()
                await self._shutdown.wait()
        except BaseException as e:  # noqa: BLE001 — surface to connect(); don't crash the loop
            self._error = e
            self._ready.set()

    def list_tools(self) -> list[McpToolSpec]:
        res = self._runtime.call(self._session.list_tools(), self._call_timeout)
        specs: list[McpToolSpec] = []
        for t in res.tools:
            ann = getattr(t, "annotations", None)
            read_only = bool(getattr(ann, "readOnlyHint", False)) if ann else False
            specs.append(
                McpToolSpec(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema or {"type": "object"},
                    read_only=read_only,
                )
            )
        return specs

    def call_tool(self, name: str, arguments: dict[str, Any]) -> McpCallResult:
        res = self._runtime.call(self._session.call_tool(name, arguments or {}), self._call_timeout)
        text = "\n".join(
            getattr(c, "text", "") for c in res.content if getattr(c, "type", None) == "text"
        )
        return McpCallResult(text=text, is_error=bool(getattr(res, "isError", False)))

    def close(self) -> None:
        if self._shutdown is not None:
            self._runtime.loop.call_soon_threadsafe(self._shutdown.set)
