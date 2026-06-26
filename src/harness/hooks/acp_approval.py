"""ACP approval relay — replaces local `ApprovalHook` when running under ACP.

For a tool flagged `requires_approval`, this hook's `before_tool` (running on the
agent's worker thread) blocks on `bridge.call(client.request_permission(...))` and
returns the gated call or a `Deny` based on the editor's answer. Allow-for-session
is remembered so the user isn't re-prompted for the same tool.

Also emits ToolCallStart / ToolCallProgress `session_update`s around every tool so
the editor shows tool activity live.
"""

from __future__ import annotations

from typing import Any

from harness.acp import events
from harness.core.types import Deny, ToolCall, ToolResult


class AcpApprovalHook:
    def __init__(
        self,
        bridge: Any,
        client: Any,
        session_id: str,
        gated: set[str],
    ) -> None:
        self._bridge = bridge
        self._client = client
        self._session_id = session_id
        self._gated = set(gated)
        self._session_allowed: set[str] = set()  # tools granted allow_always

    # ---- before_tool: emit start + (maybe) relay a permission request --------
    def before_tool(self, call: ToolCall) -> ToolCall | Deny:
        kind = events.tool_kind(self._tool_tags(call.name))
        # surface the tool starting, regardless of gating
        self._bridge.emit(
            self._client.session_update(self._session_id, events.tool_call_start(call, kind))
        )
        if call.name not in self._gated or call.name in self._session_allowed:
            return call

        outcome = self._bridge.call(
            self._client.request_permission(
                options=events.permission_options(),
                session_id=self._session_id,
                tool_call=events.permission_tool_call(call, kind),
            )
        ).outcome

        if events.is_allow_outcome(outcome):
            if getattr(outcome, "option_id", None) == "allow_always":
                self._session_allowed.add(call.name)
            return call
        return Deny(f"'{call.name}' denied by user")

    # ---- after_tool: emit completion/failure ---------------------------------
    def after_tool(self, call: ToolCall, result: ToolResult) -> None:
        self._bridge.emit(
            self._client.session_update(self._session_id, events.tool_call_done(call, result))
        )

    # tool tags are injected by the server (it knows the registry); default ().
    def _tool_tags(self, name: str) -> tuple[str, ...]:
        return self._tags_by_name.get(name, ()) if hasattr(self, "_tags_by_name") else ()

    def set_tag_map(self, tags_by_name: dict[str, tuple[str, ...]]) -> None:
        self._tags_by_name = tags_by_name


def from_tools(bridge: Any, client: Any, session_id: str, tools) -> AcpApprovalHook:
    """Build an AcpApprovalHook gating tools with requires_approval, wired with the
    tool→tags map so tool-kind reporting is accurate."""
    gated = {t.name for t in tools if t.requires_approval}
    hook = AcpApprovalHook(bridge, client, session_id, gated)
    hook.set_tag_map({t.name: t.tags for t in tools})
    return hook
