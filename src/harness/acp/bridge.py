"""Sync-loop ↔ asyncio bridge.

The harness loop is synchronous and runs tools on threads (locked decision). The
ACP SDK is asyncio. This bridges the two without leaking either model:

- async side → sync work: `await bridge.run_blocking(agent.run, messages, cancel)`
  runs the blocking loop on a worker thread (the event loop stays responsive).
- worker thread → async notification (fire-and-forget): `bridge.emit(coro)` —
  used for `session_update` chunks; we don't wait for the editor to ack.
- worker thread → async request needing a reply (blocking): `bridge.call(coro)` —
  used for `request_permission`; the worker blocks until the editor answers.

Cancellation needs no bridge: a `CancelToken` is a `threading.Event`, already
thread-safe — the async `cancel` handler just calls `.set()`.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable


class AsyncBridge:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        # Own executor so we control the worker pool feeding the agent loops.
        self._executor = executor or ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="harness-acp"
        )

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    async def run_blocking(self, fn: Callable[..., Any], *args: Any) -> Any:
        """From the event loop: run a blocking function on a worker thread and
        await its result. Exceptions propagate to the awaiting coroutine."""
        return await self._loop.run_in_executor(self._executor, lambda: fn(*args))

    def emit(self, coro: Awaitable[Any]) -> None:
        """From a worker thread: schedule a coroutine on the loop, fire-and-forget
        (session_update chunks). Never blocks the worker."""
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def call(self, coro: Awaitable[Any], timeout: float | None = None) -> Any:
        """From a worker thread: schedule a coroutine on the loop and BLOCK for its
        result (request_permission). Raises on timeout or coroutine error."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
