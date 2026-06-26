"""CLI approval hook — TTY prompt for gated tools.

Mirrors `AcpApprovalHook`'s allow-always shape, but for an in-process terminal:
prompt `y / n / a` per gated call; `a` (always) grants the tool for the rest of
the session. Safe-by-default — when there's no TTY (piped/non-interactive) a gated
tool is auto-DENIED unless `--yes` was passed.
"""

from __future__ import annotations

from typing import Callable

from harness.core.types import Deny, ToolCall

AskFn = Callable[[str], str]


class CliApprovalHook:
    def __init__(
        self,
        gated: set[str],
        *,
        assume_yes: bool = False,
        interactive: bool = True,
        ask: AskFn = input,
    ) -> None:
        self._gated = set(gated)
        self._assume_yes = assume_yes
        self._interactive = interactive
        self._ask = ask
        self._session_allowed: set[str] = set()

    def before_tool(self, call: ToolCall) -> ToolCall | Deny:
        if call.name not in self._gated or call.name in self._session_allowed:
            return call
        if self._assume_yes:
            return call
        if not self._interactive:
            return Deny(f"'{call.name}' denied (non-interactive; pass --yes to allow)")

        summary = _summarize(call)
        try:
            answer = self._ask(f"Run {summary}? [y/N/a] ").strip().lower()
        except EOFError:
            return Deny(f"'{call.name}' denied (no input)")

        if answer in ("a", "always"):
            self._session_allowed.add(call.name)
            return call
        if answer in ("y", "yes"):
            return call
        return Deny(f"'{call.name}' denied by user")


def _summarize(call: ToolCall) -> str:
    # one-line, bounded — show the tool and its most salient arg
    key = next(
        (k for k in ("command", "path", "pattern", "old_string") if k in call.arguments), None
    )
    if key is not None:
        val = str(call.arguments[key])
        if len(val) > 80:
            val = val[:77] + "..."
        return f"{call.name}({key}={val!r})"
    return f"{call.name}({', '.join(call.arguments)})"


def from_tools(tools, *, assume_yes: bool = False, interactive: bool = True, ask: AskFn = input):
    gated = {t.name for t in tools if t.requires_approval}
    return CliApprovalHook(gated, assume_yes=assume_yes, interactive=interactive, ask=ask)
