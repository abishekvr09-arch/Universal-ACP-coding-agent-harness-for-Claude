"""Approval hook — gates tools flagged requires_approval.

`before_tool` sees the REPAIRED call (so what's approved is what runs). The
decision callback decides allow/deny; the default policy auto-allows everything
EXCEPT tools whose `requires_approval` is set, which it denies unless a per-call
approver says yes. Wire a real prompt (CLI y/n, ACP request_permission) via the
`approver` callback.
"""

from __future__ import annotations

from typing import Callable

from harness.core.types import Deny, ToolCall

# approver(tool_name, arguments) -> True to allow, False to deny.
Approver = Callable[[str, dict], bool]


def _deny_all(_name: str, _args: dict) -> bool:
    return False


class ApprovalHook:
    def __init__(
        self,
        requires_approval: set[str],
        approver: Approver | None = None,
        *,
        auto_allow: bool = False,
    ) -> None:
        """`requires_approval`: tool names that need approval. `approver`: called
        for those tools. `auto_allow`: if True and no approver, allow them
        (useful for non-interactive/test runs)."""
        self._gated = set(requires_approval)
        self._approver = approver or ((lambda n, a: True) if auto_allow else _deny_all)

    def before_tool(self, call: ToolCall) -> ToolCall | Deny:
        if call.name not in self._gated:
            return call
        if self._approver(call.name, call.arguments):
            return call
        return Deny(f"'{call.name}' was not approved")


def from_tools(tools, approver: Approver | None = None, *, auto_allow: bool = False) -> ApprovalHook:
    """Build an ApprovalHook from a tool list (gates those with requires_approval)."""
    gated = {t.name for t in tools if t.requires_approval}
    return ApprovalHook(gated, approver, auto_allow=auto_allow)
