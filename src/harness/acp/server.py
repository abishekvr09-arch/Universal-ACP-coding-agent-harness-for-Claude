"""ACP server — exposes the harness over the Agent Client Protocol.

We subclass `acp.Agent` and run it with `acp.run_agent(self)` over stdio. The ACP
runtime is asyncio; our agent loop is sync + threads. The `AsyncBridge` is the
seam: each `prompt` runs `Agent.run()` on a worker thread, streaming
`session_update` chunks back onto the event loop and blocking only for
`request_permission`.

Implemented methods (wire ← pythonic):
  initialize ← initialize, session/new ← new_session, session/prompt ← prompt,
  session/cancel ← cancel, session/close ← close_session.

Verified against agent-client-protocol 0.10.1 (PROTOCOL_VERSION=1).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

import acp
from acp import schema

from harness.acp import events
from harness.acp.bridge import AsyncBridge
from harness.core.context import Compressor
from harness.core.loop import Agent, AgentConfig, reconcile_dangling_tool_calls, resolve_tool_timeout
from harness.core.types import CancelToken, TextContent, Tool
from harness.hooks.acp_approval import from_tools as acp_approval_from_tools
from harness.session.store import SessionStore


@dataclass
class _Session:
    session_id: str
    cwd: str
    model: str | None = None  # per-session model from session/new `_meta` (stable ACP channel)
    system: str | None = None  # per-session system prompt (restored byte-for-byte on resume)
    messages: list[dict[str, Any]] = field(default_factory=list)
    cancel: CancelToken = field(default_factory=CancelToken)


class HarnessAgent(acp.Agent):
    """The harness as an ACP agent. One process serves many sessions."""

    def __init__(
        self,
        provider: Any,
        tools: list[Tool],
        system: str = "",
        compressor: Compressor | None = None,
        bridge: AsyncBridge | None = None,
        make_provider: Any = None,
        available_models: tuple[str, ...] | None = None,
        default_model: str | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._system = system
        self._compressor = compressor
        self._bridge = bridge  # set lazily on first use if None (needs the loop)
        # Optional durable session store. When set, sessions persist (system byte-for-byte
        # + message log) and survive a process restart via session/load. When None the
        # server is in-memory only (tests/probes) — back-compat.
        self._store = store
        # Per-session model selection: make_provider maps a validated model_id ->
        # Provider; when absent the agent is single-model.
        self._make_provider = make_provider
        self._available_models = tuple(available_models or ())
        self._default_model = (
            default_model
            or getattr(provider, "model", None)
            or (self._available_models[0] if self._available_models else None)
        )
        self._client: acp.Client | None = None
        self._sessions: dict[str, _Session] = {}
        self._counter = 0

    def _provider_for(self, sess: _Session) -> Any:
        """The provider to drive this session's turn — per-session model if set."""
        if self._make_provider is not None and sess.model is not None:
            return self._make_provider(sess.model)
        return self._provider

    # ----------------------------------------------------------- lifecycle --
    def on_connect(self, conn: acp.Client) -> None:
        self._client = conn

    def _get_bridge(self) -> AsyncBridge:
        if self._bridge is None:
            self._bridge = AsyncBridge(asyncio.get_event_loop())
        return self._bridge

    async def initialize(
        self, protocol_version: int, client_capabilities=None, client_info=None, **kwargs
    ) -> acp.InitializeResponse:
        return acp.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=schema.AgentCapabilities(),
            agent_info=schema.Implementation(name="harness", version="0.0.1"),
            auth_methods=[],
        )

    def _new_session_id(self) -> str:
        """Durable, collision-resistant id when persisting (must survive a restart so
        the client can session/load it); a readable counter id when in-memory only."""
        if self._store is not None:
            return f"sess-{uuid.uuid4().hex[:12]}"
        self._counter += 1
        return f"sess-{self._counter}"

    def _validate_model(self, model: str | None) -> None:
        # Trust floor: validate against advertised models — never forward an arbitrary
        # client string to the provider/API. Unknown id => loud error.
        if model is not None and self._available_models and model not in self._available_models:
            raise acp.RequestError.invalid_params(f"unsupported model: {model}")

    def _model_state(self, current: str | None):
        """Advertise selectable models so an editor can discover valid ids."""
        if not self._available_models:
            return None
        return schema.SessionModelState(
            available_models=[schema.ModelInfo(model_id=m, name=m) for m in self._available_models],
            current_model_id=current or self._available_models[0],
        )

    async def new_session(self, cwd: str, **kwargs) -> acp.NewSessionResponse:
        # Per-session model arrives via _meta on session/new (the stable ACP
        # extension channel — the router spreads _meta keys into kwargs). We do NOT
        # use set_session_model: it's part of ACP's *unstable* protocol (gated behind
        # use_unstable_protocol, raises method_not_found otherwise).
        model = kwargs.get("modelId") or self._default_model
        self._validate_model(model)
        sid = self._new_session_id()
        self._sessions[sid] = _Session(session_id=sid, cwd=cwd, model=model, system=self._system)
        # Persist the session header (system byte-for-byte) so a restart can resume it.
        if self._store is not None:
            self._store.create_session(sid, self._system, model=model)
        return acp.NewSessionResponse(session_id=sid, models=self._model_state(model))

    async def load_session(self, cwd: str, session_id: str, **kwargs) -> acp.LoadSessionResponse:
        """Store-backed recovery (stable ACP `session/load`): a fresh process restores
        a prior session by id — system prompt BYTE-FOR-BYTE (Law 1) + the full message
        log. Any tool_use orphaned by the crash that ended the prior process is
        reconciled on the next prompt (and defensively in `run()`)."""
        if self._store is None:
            raise acp.RequestError.invalid_params("session resume requires a session store")
        system = self._store.get_system(session_id)
        if system is None:
            raise acp.RequestError.invalid_params(f"unknown session: {session_id}")
        model = kwargs.get("modelId") or self._default_model
        self._validate_model(model)
        self._sessions[session_id] = _Session(
            session_id=session_id,
            cwd=cwd,
            model=model,
            system=system,
            messages=self._store.load_messages(session_id),
        )
        return acp.LoadSessionResponse(models=self._model_state(model))

    async def cancel(self, session_id: str, **kwargs) -> None:
        # Pure threading.Event — no bridge needed; the worker sees it cooperatively.
        sess = self._sessions.get(session_id)
        if sess is not None:
            sess.cancel.set()

    async def close_session(self, session_id: str, **kwargs):
        self._sessions.pop(session_id, None)
        return None

    # --------------------------------------------------------------- prompt --
    async def prompt(self, prompt, session_id: str, message_id=None, **kwargs) -> acp.PromptResponse:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise acp.RequestError.invalid_params(f"unknown session: {session_id}")

        sess.cancel.clear()
        sess.messages.append({"role": "user", "content": _to_user_content(prompt)})
        # Cross-restart: if a prior turn was interrupted, fold synthetic 'interrupted'
        # results into THIS user turn so the request is valid (mirrors the CLI resume
        # path). Idempotent; run() also reconciles defensively.
        reconcile_dangling_tool_calls(sess.messages)
        if self._store is not None:
            # Persist the (reconciled) user turn in final form; the loop persists the
            # assistant + tool_result turns via persist_fn.
            self._store.append_message(session_id, sess.messages[-1])

        bridge = self._get_bridge()
        client = self._client
        assert client is not None  # on_connect ran before any prompt

        # Stream assistant text to the editor as it arrives.
        def on_chunk(text: str) -> None:
            bridge.emit(client.session_update(session_id, events.message_chunk(text)))

        hooks = [acp_approval_from_tools(bridge, client, session_id, self._tools)]
        persist = self._store.persist_fn(session_id) if self._store is not None else None
        agent = Agent(AgentConfig(
            provider=self._provider_for(sess),
            tools=self._tools,
            system=sess.system or self._system,
            hooks=hooks,
            on_chunk=on_chunk,
            persist=persist,
            tool_timeout=resolve_tool_timeout(),
        ))

        # Run the blocking loop off the event loop; bridge marshals updates back.
        final = await bridge.run_blocking(agent.run, sess.messages, sess.cancel)

        # Compression is caller-driven; the server is a caller. Session boundary.
        if self._compressor is not None:
            self._compressor.compress_if_needed(sess.messages, final)

        stop = "cancelled" if sess.cancel.is_set() else events.map_stop_reason(final.stop_reason)
        return acp.PromptResponse(stop_reason=stop)


def _to_user_content(prompt: list[Any]) -> list[dict[str, Any]]:
    """ACP prompt content blocks -> Anthropic-shaped user content."""
    out: list[dict[str, Any]] = []
    for block in prompt:
        if getattr(block, "type", None) == "text" or hasattr(block, "text"):
            out.append({"type": "text", "text": getattr(block, "text", "")})
    return out or [{"type": "text", "text": ""}]


def main() -> None:
    """Console entry (`harness-acp`): build the agent and serve over stdio."""
    import os

    # Test affordance: a scripted offline provider for ACP wire/translation probes.
    # Lazy + env-gated so the shipped entry NEVER loads a fake in normal operation.
    if os.environ.get("HARNESS_PROVIDER", "").startswith("fake"):
        from harness.testing import build_blocking_provider, build_fake_provider, fake_tools

        # `fake-cancel` blocks until a session/cancel sets the token (cancellation probe);
        # plain `fake` runs the scripted text→tool→pong scenario.
        fake = (
            build_blocking_provider()
            if os.environ["HARNESS_PROVIDER"] == "fake-cancel"
            else build_fake_provider()
        )
        agent = HarnessAgent(
            provider=fake,
            tools=fake_tools(),
            system="fake",
            compressor=None,
            make_provider=lambda _m: fake,  # single fake model; lets us assert propagation
            available_models=("fake",),
            default_model="fake",
        )
        asyncio.run(acp.run_agent(agent))
        return

    from harness.providers import ClaudeProvider, default_profile
    from harness.providers.claude import DEFAULT_MODEL
    from harness.tools import default_tools

    profile = default_profile()
    default_model = os.environ.get("HARNESS_MODEL", DEFAULT_MODEL)
    provider = ClaudeProvider(model=default_model, profile=profile)

    def make_provider(model_id: str) -> ClaudeProvider:
        # Per-session model: reuse the base for the default model, else a sibling
        # sharing the base's (lazily-built) HTTP client — one connection, many models.
        if model_id == provider.model:
            return provider
        return ClaudeProvider(model=model_id, profile=profile, client=provider._get_client())

    compressor = Compressor(ClaudeProvider(model="claude-haiku-4-5"))
    # Durable session store: persists each session (system byte-for-byte + message log)
    # so a restarted process can resume via session/load. Same DB the CLI uses.
    store = SessionStore(os.environ.get("HARNESS_DB", "harness.db"))
    agent = HarnessAgent(
        provider=provider,
        tools=default_tools(),
        system="You are a coding agent operating over ACP.",
        compressor=compressor,
        make_provider=make_provider,
        available_models=tuple(profile.supported_models),
        default_model=default_model,
        store=store,
    )
    asyncio.run(acp.run_agent(agent))


if __name__ == "__main__":
    main()
