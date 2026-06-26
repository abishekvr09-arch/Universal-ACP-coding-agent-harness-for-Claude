"""The sync<->asyncio bridge, exercised with a real event loop in a worker thread."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from harness.acp.bridge import AsyncBridge


class LoopThread:
    """Run an asyncio loop in a background thread for the duration of a test."""

    def __enter__(self) -> asyncio.AbstractEventLoop:
        self.loop = asyncio.new_event_loop()
        self.t = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.t.start()
        return self.loop

    def __exit__(self, *a):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.t.join(timeout=2)
        self.loop.close()


def test_run_blocking_returns_sync_result():
    with LoopThread() as loop:
        bridge = AsyncBridge(loop)

        def sync_work(a, b):
            time.sleep(0.01)
            return a + b

        fut = asyncio.run_coroutine_threadsafe(bridge.run_blocking(sync_work, 2, 3), loop)
        assert fut.result(timeout=2) == 5


def test_run_blocking_propagates_exception():
    with LoopThread() as loop:
        bridge = AsyncBridge(loop)

        def boom():
            raise ValueError("nope")

        fut = asyncio.run_coroutine_threadsafe(bridge.run_blocking(boom), loop)
        with pytest.raises(ValueError, match="nope"):
            fut.result(timeout=2)


def test_emit_is_fire_and_forget_from_worker():
    with LoopThread() as loop:
        bridge = AsyncBridge(loop)
        seen: list[int] = []

        async def record(x):
            seen.append(x)

        # simulate the worker thread emitting notifications
        for i in range(3):
            bridge.emit(record(i))
        # emit doesn't block; give the loop a moment to drain
        deadline = time.time() + 2
        while len(seen) < 3 and time.time() < deadline:
            time.sleep(0.01)
        assert seen == [0, 1, 2]


def test_call_blocks_worker_for_result():
    with LoopThread() as loop:
        bridge = AsyncBridge(loop)

        async def permission(answer):
            await asyncio.sleep(0.01)
            return answer

        # worker thread blocks until the loop produces the reply (request_permission)
        result = bridge.call(permission("allow_once"))
        assert result == "allow_once"


def test_cancel_token_needs_no_bridge():
    # A CancelToken is a threading.Event — the async cancel handler just sets it,
    # and the sync worker sees it. No bridge call required.
    from harness.core.types import CancelToken

    token = CancelToken()
    with LoopThread() as loop:
        bridge = AsyncBridge(loop)
        observed = {}

        def worker():
            for _ in range(100):
                if token.is_set():
                    observed["cancelled"] = True
                    return "stopped"
                time.sleep(0.005)
            return "ran-to-end"

        fut = asyncio.run_coroutine_threadsafe(bridge.run_blocking(worker), loop)
        time.sleep(0.02)
        token.set()  # "async cancel handler" trips it
        assert fut.result(timeout=2) == "stopped"
        assert observed.get("cancelled") is True
