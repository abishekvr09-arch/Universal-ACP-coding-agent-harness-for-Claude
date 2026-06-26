"""Context compression — Law 3: the ONLY legal context mutation.

Context grows; it never shrinks in place. The one sanctioned mutation is
compression at a SESSION BOUNDARY (after the model's final non-tool_use turn,
before the next user prompt). This module is caller-driven: `agent.run()` stays
small, and the CLI / ACP server / embedders each decide when to call
`Compressor.compress_if_needed(...)`.

Six rules, each independently testable:
1. Never split a tool_use assistant message from its paired tool_result user
   message — `is_safe_split_point` is the invariant; cuts land on turn boundaries.
2. Gate on REAL `prompt_tokens` (Response.usage.input_tokens), never an estimate.
3. Anti-thrash: skip if the last two compactions each saved < min_save_ratio.
4. Head-protection decays: early turns are protected at first, less each pass.
5. Keep a recent tail by (proxy) token budget; summarize the middle.
6. Summarize with a cheap aux model (Haiku) via a reused ClaudeProvider.

Char-based token estimates are used ONLY to size the tail / measure save ratios —
the compression *trigger* always uses the real token count from the API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from harness.core.types import Provider, TextContent

# --------------------------------------------------------------------------- #
# Message-shape predicates + the split invariant
# --------------------------------------------------------------------------- #


def _blocks(msg: dict[str, Any]) -> list[Any]:
    content = msg.get("content", [])
    return content if isinstance(content, list) else []


def _has_tool_use(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "assistant" and any(
        isinstance(b, dict) and b.get("type") == "tool_use" for b in _blocks(msg)
    )


def _has_tool_result(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "user" and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in _blocks(msg)
    )


def _is_plain_assistant(msg: dict[str, Any]) -> bool:
    """An assistant turn that ENDS an exchange (no pending tool calls)."""
    return msg.get("role") == "assistant" and not _has_tool_use(msg)


def is_safe_split_point(messages: list[dict[str, Any]], i: int) -> bool:
    """True if cutting the list as messages[:i] | messages[i:] does NOT orphan a
    tool_result from its tool_use. The only unsafe cut is between a tool_use
    assistant turn (i-1) and its tool_result user turn (i)."""
    if i <= 0 or i >= len(messages):
        return True
    return not (_has_tool_use(messages[i - 1]) and _has_tool_result(messages[i]))


def turn_boundaries(messages: list[dict[str, Any]]) -> list[int]:
    """Indices that begin a fresh exchange: 0, len, and every i where messages[i]
    is a user prompt (not tool_result) following a plain assistant turn. Cuts at
    these points keep complete exchanges intact AND alternate roles across the
    seam (…assistant | user…)."""
    out = [0]
    for i in range(1, len(messages)):
        prev, cur = messages[i - 1], messages[i]
        if _is_plain_assistant(prev) and cur.get("role") == "user" and not _has_tool_result(cur):
            out.append(i)
    out.append(len(messages))
    return out


def _est_tokens(msg: dict[str, Any]) -> int:
    """Cheap char-based proxy (~4 chars/token). For SIZING ONLY, never the trigger."""
    return max(len(json.dumps(msg.get("content", []), default=str)) // 4, 1)


# --------------------------------------------------------------------------- #
# Policy + Compressor
# --------------------------------------------------------------------------- #


@dataclass
class CompressionPolicy:
    context_window: int = 200_000
    trigger_ratio: float = 0.6  # compress when real input_tokens exceed this fraction
    tail_ratio: float = 0.2  # keep ~this fraction of the window as recent tail
    head_protect_turns: int = 2  # exchanges protected at the head (decays each pass)
    min_save_ratio: float = 0.10  # anti-thrash: skip if last 2 saves were < this
    summary_model: str = "claude-haiku-4-5"
    summary_max_tokens: int = 1024


_SUMMARY_SYSTEM = (
    "You compress conversation history for an autonomous coding agent. Produce a "
    "dense, faithful summary that preserves: the user's goals and constraints, key "
    "decisions, file paths and identifiers touched, and any unresolved threads. "
    "Omit pleasantries. Write in compact prose or bullets. Do not invent facts."
)


class Compressor:
    def __init__(
        self,
        summary_provider: Provider,
        policy: CompressionPolicy | None = None,
    ) -> None:
        # summary_provider is a ClaudeProvider already bound to the cheap model —
        # we reuse the provider class rather than building a second one.
        self.provider = summary_provider
        self.policy = policy or CompressionPolicy()
        # in-memory fallback state when no SessionStore is supplied
        self._mem: dict[str, list[float]] = {}

    # ----------------------------------------------------------------- gate --
    def should_compress(self, input_tokens: int) -> bool:
        return input_tokens > self.policy.trigger_ratio * self.policy.context_window

    # ----------------------------------------------------------------- main --
    def compress_if_needed(
        self,
        messages: list[dict[str, Any]],
        last_response: Any,
        store: Any = None,
        session_id: str | None = None,
    ) -> bool:
        """Compress `messages` IN PLACE at a session boundary. Returns True if the
        list was rewritten. No-op (False) if under threshold, anti-thrashed, or no
        safe boundary exists."""
        input_tokens = getattr(getattr(last_response, "usage", None), "input_tokens", 0)
        if not self.should_compress(input_tokens):
            return False

        prior_count, recent_saves = self._stats(store, session_id)
        if self._anti_thrash(recent_saves):
            return False

        plan = self._plan_cut(messages, prior_count)
        if plan is None:
            return False
        head_end, tail_start = plan

        before = sum(_est_tokens(m) for m in messages)
        summary_text = self._summarize(messages[head_end:tail_start])
        if not summary_text:
            return False

        # Fold the summary into the first tail message (a fresh user prompt), so we
        # add no message and the …assistant | user… alternation across the seam holds.
        tail0 = dict(messages[tail_start])
        tail0["content"] = [
            {"type": "text", "text": f"[Earlier conversation summarized]\n{summary_text}"}
        ] + _blocks(tail0)
        new_messages = messages[:head_end] + [tail0] + messages[tail_start + 1 :]

        after = sum(_est_tokens(m) for m in new_messages)
        if after >= before:  # never grow
            return False

        messages[:] = new_messages
        self._record(store, session_id, before, after)
        return True

    # ------------------------------------------------------------- planning --
    def _plan_cut(
        self, messages: list[dict[str, Any]], prior_compactions: int
    ) -> tuple[int, int] | None:
        bounds = turn_boundaries(messages)
        if len(bounds) < 3:
            return None  # need at least head | middle | tail boundaries

        # Head protection decays toward 0 with each prior compaction.
        effective_head = max(self.policy.head_protect_turns - prior_compactions, 0)
        head_end = bounds[min(effective_head, len(bounds) - 1)]

        # Tail: walk from the end accumulating proxy tokens up to the tail budget,
        # then snap to the latest turn boundary at/under that index.
        budget = self.policy.tail_ratio * self.policy.context_window
        acc = 0
        tail_target = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            acc += _est_tokens(messages[i])
            if acc > budget:
                tail_target = i + 1
                break
            tail_target = i
        tail_start = max(b for b in bounds if b <= tail_target)

        if head_end >= tail_start or tail_start - head_end < 1:
            return None  # nothing safely summarizable between head and tail
        # tail_start must be a real user-prompt boundary (it is, by construction)
        return head_end, tail_start

    # ------------------------------------------------------------ summarize --
    def _summarize(self, middle: list[dict[str, Any]]) -> str:
        if not middle:
            return ""
        transcript = _render_transcript(middle)
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": transcript + "\n\nSummarize the conversation above."}
                ],
            }
        ]
        try:
            resp = self.provider.stream(_SUMMARY_SYSTEM, msgs, [], cancel=None)
        except Exception:  # noqa: BLE001 — a failed summary must never corrupt history
            return ""
        return "".join(b.text for b in resp.content if isinstance(b, TextContent)).strip()

    # --------------------------------------------------------- state helpers --
    def _stats(self, store: Any, session_id: str | None) -> tuple[int, list[float]]:
        if store is not None and session_id is not None:
            return store.compaction_stats(session_id)
        saves = self._mem.get(session_id or "_default", [])
        return len(saves), saves[-2:]

    def _record(self, store: Any, session_id: str | None, before: int, after: int) -> None:
        if store is not None and session_id is not None:
            store.record_compaction(session_id, before, after)
            return
        self._mem.setdefault(session_id or "_default", []).append(
            1.0 - (after / before) if before else 0.0
        )

    def _anti_thrash(self, recent_saves: list[float]) -> bool:
        if len(recent_saves) < 2:
            return False
        return all(s < self.policy.min_save_ratio for s in recent_saves[:2])


def _render_transcript(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        for b in _blocks(m):
            if not isinstance(b, dict):
                lines.append(f"{role}: {b}")
                continue
            t = b.get("type")
            if t == "text":
                lines.append(f"{role}: {b.get('text', '')}")
            elif t == "tool_use":
                lines.append(f"{role} -> tool {b.get('name')}({json.dumps(b.get('input', {}), default=str)})")
            elif t == "tool_result":
                inner = b.get("content", [])
                txt = " ".join(
                    x.get("text", "") for x in inner if isinstance(x, dict) and x.get("type") == "text"
                )
                lines.append(f"tool_result: {txt[:500]}")
    return "\n".join(lines)
