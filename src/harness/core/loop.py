"""The agentic loop — the frozen core.

Everything here is load-bearing and small on purpose. Capability lives at the
edges (tools, providers, hooks). The single most important property is the
INVARIANT: every tool_use block the model emits gets exactly one tool_result —
denial, cancellation, and tool exceptions all emit a synthetic error_result, or
the next provider call 400s on a tool_use/tool_result mismatch.

Concurrency: synchronous loop; the tool batch runs on a bounded
ThreadPoolExecutor. The batch degrades to sequential if any call targets a tool
that is execution_mode="sequential" or not parallel_safe. Results are reassembled
in original call order so tool_result ordering is deterministic.
"""

from __future__ import annotations

import copy
import functools
import inspect
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable

from harness.core.budget import IterationBudget
from harness.core.repair import promote_plaintext_tool_calls, repair_call
from harness.core.types import (
    CancelToken,
    Deny,
    Hook,
    Provider,
    Response,
    TextContent,
    Tool,
    ToolCall,
    ToolResult,
    error_result,
)

log = logging.getLogger("harness.loop")


@functools.lru_cache(maxsize=256)
def _accepts_cancel(handler: Callable[..., Any]) -> bool:
    """True if the handler declares a `cancel` parameter or accepts **kwargs.
    Cached per handler object so we introspect once, not per call."""
    try:
        params = inspect.signature(handler).parameters
    except (TypeError, ValueError):
        return False
    if "cancel" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


@functools.lru_cache(maxsize=64)
def _accepts_on_chunk(stream_fn: Callable[..., Any]) -> bool:
    """True if a provider's stream() accepts an on_chunk kwarg (or **kwargs)."""
    try:
        params = inspect.signature(stream_fn).parameters
    except (TypeError, ValueError):
        return False
    if "on_chunk" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

# ThreadPoolExecutor bound: small constant by default (most tool batches are
# tiny; the work is I/O-bound — bash, file reads, ripgrep). Overridable via env.
DEFAULT_MAX_WORKERS = int(os.environ.get("HARNESS_MAX_TOOL_WORKERS", "4"))

# Persist callback: (message_dict) -> None. Defaults to no-op; the SQLite store
# injects a real one. Persistence happens BEFORE tool execution (crash-safety).
PersistFn = Callable[[dict[str, Any]], None]


_DEFAULT_TOOL_TIMEOUT = 900.0  # seconds; above bash's 600s max so bash self-limits first


def resolve_tool_timeout(default: float = _DEFAULT_TOOL_TIMEOUT) -> float | None:
    """Read HARNESS_TOOL_TIMEOUT (seconds) from the environment for the shipping drivers.
    Fail-CLOSED parsing — a typo must never silently remove the production backstop:
      unset      -> default (backstop on)
      n > 0      -> n
      n <= 0     -> None  (explicit, deliberate disable — the only way to turn it off)
      unparsable -> default + WARN  (e.g. '900s', 'ten' — keep the floor, don't drop it)
    """
    raw = os.environ.get("HARNESS_TOOL_TIMEOUT")
    if raw is None:
        return default
    try:
        val = float(raw)
    except (ValueError, TypeError):
        log.warning("invalid HARNESS_TOOL_TIMEOUT=%r; keeping the %ss backstop", raw, default)
        return default
    return val if val > 0 else None


@dataclass
class AgentConfig:
    provider: Provider
    tools: list[Tool]
    system: str = ""
    hooks: list[Hook] = field(default_factory=list)
    budget: IterationBudget | None = None
    persist: PersistFn | None = None
    # Mutates the per-turn message COPY to add ephemeral context (RAG/memory/plugin).
    # ISOLATION IS ENFORCED: run() hands this a DEEP COPY of the history (loop.run),
    # so any mutation — reassign OR in-place .append/+= — stays in the throwaway API
    # view and CANNOT leak into canonical/persisted history. (CLAUDE.md Gotcha 17 / L8.)
    inject_ephemeral: Callable[[list[dict[str, Any]]], None] | None = None
    max_workers: int = DEFAULT_MAX_WORKERS
    # streamed text deltas → live UIs (ACP AgentMessageChunk). None = no streaming.
    on_chunk: Callable[[str], None] | None = None
    # Backstop timeout (seconds) for a single tool call. None = no backstop (the
    # default; tools self-limit — bash subprocess timeout, MCP call_timeout — and
    # long-runners cooperate via `cancel`). Set it to bound a NON-cooperative tool that
    # hangs: on expiry the loop synthesizes a timeout error_result (invariant holds) and
    # stops waiting. Ceiling: the hung thread leaks until process exit (can't kill a
    # Python thread). See _dispatch_timed.
    tool_timeout: float | None = None


class Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.cfg = config
        self.registry: dict[str, Tool] = {t.name: t for t in config.tools}
        self.budget = config.budget or IterationBudget()
        self._allowed_names = set(self.registry)

    # ----------------------------------------------------------------- hooks --
    def _stream(self, api_msgs: list[dict[str, Any]], cancel: CancelToken | None) -> Response:
        """Call the provider, passing on_chunk only when set AND the provider
        accepts it — so minimal/fake providers with a narrower signature still work."""
        if self.cfg.on_chunk is not None and _accepts_on_chunk(self.cfg.provider.stream):
            return self.cfg.provider.stream(
                self.cfg.system, api_msgs, self.cfg.tools,
                cancel=cancel, on_chunk=self.cfg.on_chunk,
            )
        return self.cfg.provider.stream(self.cfg.system, api_msgs, self.cfg.tools, cancel=cancel)

    def _before_model(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for h in self.cfg.hooks:
            fn = getattr(h, "before_model", None)
            if callable(fn):
                messages = fn(messages)
        return messages

    def _before_tool(self, call: ToolCall) -> ToolCall | Deny:
        for h in self.cfg.hooks:
            fn = getattr(h, "before_tool", None)
            if callable(fn):
                outcome = fn(call)
                if isinstance(outcome, Deny):
                    return outcome
                if outcome is not None:
                    call = outcome
        return call

    def _after_tool(self, call: ToolCall, result: ToolResult) -> None:
        for h in self.cfg.hooks:
            fn = getattr(h, "after_tool", None)
            if callable(fn):
                fn(call, result)

    # ------------------------------------------------------------- execution --
    def _execute_one(self, call: ToolCall, cancel: CancelToken | None) -> ToolResult:
        tool = self.registry.get(call.name)
        if tool is None:
            return error_result(call.id, f"unknown tool: {call.name}")
        if tool.handler is None:
            return error_result(call.id, f"tool '{call.name}' has no handler bound")
        # Decide ONCE (cached) whether the handler takes `cancel`, by signature —
        # so a TypeError raised INSIDE the handler body isn't mistaken for an
        # arity mismatch and silently retried (which would mask the real bug).
        kwargs = dict(call.arguments)
        if _accepts_cancel(tool.handler):
            kwargs["cancel"] = cancel
        try:
            return tool.handler(**kwargs)
        except Exception as e:  # noqa: BLE001 — never propagate; invariant
            return error_result(call.id, f"{type(e).__name__}: {e}")

    def _dispatch(
        self, calls: list[ToolCall], cancel: CancelToken | None
    ) -> list[ToolResult]:
        """Run already-approved calls. Bounded parallelism, degrading to serial
        if any call is sequential / not parallel_safe. Order preserved."""
        if not calls:
            return []
        serial = any(
            (t := self.registry.get(c.name)) is None
            or t.execution_mode == "sequential"
            or not t.parallel_safe
            for c in calls
        )
        if self.cfg.tool_timeout is not None:
            return self._dispatch_timed(calls, cancel, serial)

        if serial or len(calls) == 1:
            return [self._execute_one(c, cancel) for c in calls]

        workers = min(self.cfg.max_workers, len(calls))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda c: self._execute_one(c, cancel), calls))

    def _dispatch_timed(
        self, calls: list[ToolCall], cancel: CancelToken | None, serial: bool
    ) -> list[ToolResult]:
        """Run the batch with a per-call backstop timeout. On expiry, synthesize a
        timeout error_result (preserving the invariant) and STOP WAITING — never block
        the loop on a hung thread. We can't kill a Python thread, so a non-cooperative
        hung tool leaks until process exit (cooperative tools see `cancel` and stop);
        the loop and the invariant are unaffected either way. Order is preserved."""
        timeout = self.cfg.tool_timeout
        results: list[ToolResult] = []
        if serial:
            # Sequential semantics: one at a time, NO overlap. A fresh single-worker pool
            # per call so a hung call can't occupy the worker the next call needs. But a
            # timeout does NOT kill the worker (can't kill a Python thread) — its thread
            # may still be mutating state. So once a serial call times out we FAIL-STOP:
            # skip the remaining serial calls with a synthetic result rather than start
            # another tool that could overlap the still-alive one. Preserves both the
            # invariant (one result per call) AND sequential isolation. The model sees the
            # timeout + skips and can retry. (Without this, the next serial tool runs while
            # the timed-out one is still live — verified, the bug this guards.)
            timed_out = False
            for c in calls:
                if timed_out:
                    results.append(error_result(
                        c.id,
                        f"tool '{c.name}' skipped: a prior sequential tool timed out and "
                        "could not be confirmed stopped (sequential isolation)",
                    ))
                    continue
                pool = ThreadPoolExecutor(max_workers=1)
                fut = pool.submit(self._execute_one, c, cancel)
                try:
                    results.append(fut.result(timeout=timeout))
                except FuturesTimeoutError:
                    results.append(error_result(c.id, f"tool '{c.name}' timed out after {timeout}s"))
                    timed_out = True
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
            return results
        # Parallel: submit all, collect against a single batch deadline (so N hung calls
        # cost ~timeout total, not N×timeout).
        workers = min(self.cfg.max_workers, len(calls))
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = [pool.submit(self._execute_one, c, cancel) for c in calls]
            deadline = time.monotonic() + timeout
            for c, fut in zip(calls, futs):
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    results.append(fut.result(timeout=remaining))
                except FuturesTimeoutError:
                    results.append(error_result(c.id, f"tool '{c.name}' timed out after {timeout}s"))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return results

    # ------------------------------------------------------------------ loop --
    def run(self, messages: list[dict[str, Any]], cancel: CancelToken | None = None) -> Response:
        """Drive the conversation to a stop. `messages` is mutated in place as the
        canonical history (Anthropic-shaped dicts). Returns the final Response."""
        from harness.providers.cache import mark_cache_breakpoints

        # Cross-restart invariant (Gotcha 18): a caller may seed a history whose final
        # assistant turn was interrupted mid-tool-execution — the loop persists the
        # tool_use turn BEFORE running tools, so a crash before the tool_result turn
        # leaves a dangling tool_use. Reconcile it (synthetic 'interrupted' results)
        # BEFORE the first provider call so the wire never carries an orphan. The
        # in-turn invariant only spans one turn; this extends it across the restart.
        reconcile_dangling_tool_calls(messages)

        last: Response | None = None
        while True:
            # Cancellation between turns: the prior turn's results are already
            # persisted, so unwind cleanly rather than raising. In-flight cancel
            # (mid-stream / mid-batch) is handled below by the provider + the
            # synthetic-result fill.
            if cancel is not None and cancel.is_set():
                break

            # Budget gate (with one grace turn after exhaustion).
            if self.budget.exhausted and not self.budget.take_grace():
                break
            self.budget.consume()

            # Per-turn view: a DEEP copy of the canonical history. This is the ONE
            # structural isolation boundary — inject_ephemeral / before_model operate
            # on it freely (reassign OR in-place) and physically cannot alias canonical
            # state. Exactly one deepcopy per turn: mark_cache_breakpoints then marks
            # the tail IN PLACE (no second copy — it replaces the old shallow-copy +
            # deepcopy-in-apply_cache_breakpoints pair, so cost is unchanged). (Gotcha 17.)
            api_msgs = copy.deepcopy(messages)
            if self.cfg.inject_ephemeral:
                self.cfg.inject_ephemeral(api_msgs)
            api_msgs = self._before_model(api_msgs)
            mark_cache_breakpoints(api_msgs)

            response = self._stream(api_msgs, cancel)
            response = promote_plaintext_tool_calls(response, self._allowed_names)
            last = response

            assistant_msg = _assistant_message(response)
            # Persist BEFORE appending to in-memory history AND before executing tools
            # (fail-closed): if the write fails (disk full / WAL exhaustion) it raises,
            # and neither the store nor `messages` advances — no divergence, no tool
            # side effects without a durable record. The turn is simply lost, cleanly.
            self._persist(assistant_msg)
            messages.append(assistant_msg)

            if response.stop_reason != "tool_use":
                break

            # ---- INVARIANT: one tool_result per tool_use, paired in order ----
            approved: list[ToolCall] = []
            preset: dict[int, tuple[ToolCall, ToolResult]] = {}  # idx -> short-circuit
            for i, raw in enumerate(response.tool_calls):
                tool = self.registry.get(raw.name)
                schema = tool.input_schema if tool else {}
                call = repair_call(raw, schema)  # rebinds (repaired object)

                if cancel is not None and cancel.is_set():
                    preset[i] = (call, error_result(call.id, "cancelled"))
                    continue
                outcome = self._before_tool(call)
                if isinstance(outcome, Deny):
                    preset[i] = (call, error_result(call.id, outcome.reason))
                    continue
                preset[i] = (outcome, None)  # placeholder; result filled below
                approved.append(outcome)

            results = self._dispatch(approved, cancel)
            ri = iter(results)
            processed: list[tuple[ToolCall, ToolResult]] = []
            for i in range(len(response.tool_calls)):
                call, fixed = preset[i]
                result = fixed if fixed is not None else next(ri)
                processed.append((call, result))

            tool_result_blocks: list[dict[str, Any]] = []
            all_terminate = bool(processed)
            for call, result in processed:
                self._after_tool(call, result)
                tool_result_blocks.append(_tool_result_block(call, result))
                if not result.terminate:
                    all_terminate = False

            user_msg = {"role": "user", "content": tool_result_blocks}
            # Persist before append (fail-closed). If this write fails the tools have
            # already run but their results aren't durable; the store keeps the
            # assistant tool_use turn → a dangling tool_use, which the next resume
            # RECONCILES. So a mid-turn persist failure is recoverable, never corrupt.
            self._persist(user_msg)
            messages.append(user_msg)

            if self._is_code_only_turn(response, processed):
                self.budget.refund()

            if all_terminate:
                break

        return last if last is not None else Response(content=[], tool_calls=[], stop_reason="end_turn")

    def _is_code_only_turn(
        self, response: Response, processed: list[tuple[ToolCall, ToolResult]]
    ) -> bool:
        """A turn the budget refunds: the model produced no text and called ONLY
        execute-tagged tools (pure execution churn, not reasoning)."""
        if not processed or _has_text(response):
            return False
        for call, _ in processed:
            tool = self.registry.get(call.name)
            if tool is None or "execute" not in tool.tags:
                return False
        return True

    def _persist(self, message: dict[str, Any]) -> None:
        if self.cfg.persist:
            self.cfg.persist(message)


# --------------------------------------------------------------------------- #
# Message construction (Anthropic-shaped)
# --------------------------------------------------------------------------- #


def _assistant_message(response: Response) -> dict[str, Any]:
    """Rebuild the assistant message from the response for history. Prefer the
    provider's raw content blocks (preserves thinking signatures, tool_use ids)."""
    if response.raw is not None and getattr(response.raw, "content", None) is not None:
        return {"role": "assistant", "content": response.raw.content}
    # Fallback: reconstruct from normalized fields.
    blocks: list[dict[str, Any]] = []
    for b in response.content:
        if isinstance(b, TextContent):
            blocks.append({"type": "text", "text": b.text})
    for c in response.tool_calls:
        blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
    return {"role": "assistant", "content": blocks}


def _tool_result_block(call: ToolCall, result: ToolResult) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for b in result.content:
        if isinstance(b, TextContent):
            content.append({"type": "text", "text": b.text})
        else:
            content.append(b)  # image / raw block passes through
    return {
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": content,
        "is_error": result.is_error,
    }


def _has_text(response: Response) -> bool:
    return any(isinstance(b, TextContent) and b.text.strip() for b in response.content)


# --------------------------------------------------------------------------- #
# Cross-restart reconciliation (Bucket B — the tool_result invariant across a crash)
# --------------------------------------------------------------------------- #

_INTERRUPTED = (
    "Tool call interrupted by a process restart; its result was not recorded. "
    "Re-verify any state it may have changed before relying on it."
)


def _content_list(message: dict[str, Any]) -> list[Any]:
    content = message.get("content")
    return content if isinstance(content, list) else []


def _interrupted_result(tool_use_id: str) -> dict[str, Any]:
    """A synthetic is_error tool_result for a tool_use that never ran — same shape as
    a real one (`_tool_result_block`). Re-executes nothing, so no duplicated side
    effects on resume."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": _INTERRUPTED}],
        "is_error": True,
    }


def reconcile_dangling_tool_calls(messages: list[dict[str, Any]]) -> int:
    """Ensure every assistant `tool_use` block has a matching `tool_result` in the
    IMMEDIATELY following user message — the property a crash can break (the loop
    persists the tool_use turn before running tools, the tool_result turn after, so a
    crash between the two writes orphans the tool_use). For each unmatched id, fold a
    synthetic 'interrupted' result into that following user message (prepended, so the
    structure stays valid + role-alternating), or insert a fresh user message if the
    next message is missing/not a user turn. Mutates `messages` in place; returns the
    count synthesized. IDEMPOTENT (a second pass finds nothing) and re-executes
    nothing. This is the same synthetic-result mechanism the in-turn invariant uses,
    applied across the persistence boundary."""
    synthesized = 0
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant":
            use_ids = [
                b["id"]
                for b in _content_list(m)
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
            ]
            if use_ids:
                nxt = messages[i + 1] if i + 1 < len(messages) else None
                if nxt is not None and nxt.get("role") == "user":
                    have = {
                        b.get("tool_use_id")
                        for b in _content_list(nxt)
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    }
                    missing = [u for u in use_ids if u not in have]
                    if missing:
                        nxt["content"] = [_interrupted_result(u) for u in missing] + _content_list(nxt)
                        synthesized += len(missing)
                else:
                    messages.insert(
                        i + 1,
                        {"role": "user", "content": [_interrupted_result(u) for u in use_ids]},
                    )
                    synthesized += len(use_ids)
        i += 1
    return synthesized
