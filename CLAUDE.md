# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The project: `harness/` — Claude-first coding agent harness

A focused dev/coding agent harness — a composable tool-calling loop with a
clean plugin API for tools, providers, and hooks. NOT a multi-channel assistant.
Ships as a **standalone core** + an **OpenClaw plugin adapter** (a TS process
that spawns the Python harness as an ACP subprocess) for ClawHub distribution,
speaks **ACP** for language-independent ecosystem interop, and is an **MCP
client** (inherits the MCP tool catalogue through the same three seams — no core
change).

Claude-first, but the provider contract supports any model backend.

## Build & run

```bash
# install (editable) with dev + acp extras
python -m pip install -e ".[dev,acp]"

# run the whole test suite (offline — uses a scripted fake provider)
python -m pytest -q

# run a single test file / single test
python -m pytest tests/test_invariant.py -q
python -m pytest tests/test_invariant.py::test_denial_still_emits_result -q

# byte-compile sanity check
python -m compileall -q src

# attach external MCP servers (mcpServers-shaped JSON, like Claude Desktop)
harness --mcp mcp.json "your prompt"
python -m pip install mcp        # live stdio MCP (test_mcp_live); no `mcp` extra yet

# tests don't need a key; a live agent run needs ANTHROPIC_API_KEY in env
```

The suite is **155 tests** — 152 offline (no key, no network: a scripted
FakeProvider + an in-process FakeMcpClient) plus `test_mcp_live.py` (3 tests) that
spawn a real stdio MCP server and **skip** unless the `mcp` package + a server are
present. Source layout is `src/harness/...`; tests add `src` to `pythonpath` via
`pyproject.toml`, and use `from conftest import ...` (so **no `tests/__init__.py`** —
it breaks pytest's default import mode). `HARNESS_MAX_TOOL_WORKERS` (default 4)
bounds the tool ThreadPoolExecutor. `HARNESS_TOOL_TIMEOUT` (default 900s) sets the
non-cooperative-hang backstop for both CLI and ACP drivers (0 disables).

## Three design laws

Load-bearing constraints extracted from OpenClaw (380k★ TS) and Hermes Agent
(200k★ Python). Violating any creates subtle, expensive bugs.

### 1. Prompt caching is sacred
- Tool **declarations** (schemas sent to the model) are **FROZEN at conversation
  start** — adding/removing/reordering invalidates the cached prefix. Tool
  **handlers** (the code that runs) may be lazy-loaded on first call; the model
  never sees the difference.
- System prompt built ONCE, stored in session store, replayed byte-for-byte.
  WARN on restore miss (silent cost multiplier).
- Never mutate the live message list. The loop hands hooks/injectors a **deep copy**
  of the history each turn (`copy.deepcopy(messages)` at the top of `loop.run`), so
  ephemeral context (RAG, memory, plugins) can be added by reassign OR in-place
  mutation and **cannot** leak into canonical/persisted history — isolation is
  ENFORCED, not a convention. Exactly one deepcopy per turn: `mark_cache_breakpoints`
  marks the tail in place afterward (no second copy), so `cache_control` never touches
  canonical state either. (Audited 7/7 + structurally enforced — see Gotcha 17.)
- Cache breakpoints: `system_and_3` — system prompt + last 3 messages (sliding
  window, applied to a deep copy). These markers are **Anthropic-shaped**
  (`cache_control` blocks); other providers ignore or translate them.

### 2. Narrow waist — capability at the edges
Three extension seams, nothing else in the core:
- **Tools** — `name`, `description`, `input_schema`, `handler` + metadata
  (`parallel_safe`, `requires_approval`, `execution_mode`, `tags`).
- **Providers** — declarative profile + override hooks. Client construction,
  credential rotation, streaming stay OUT. `(extra_body, top_level_kwargs)`
  split for reasoning config.
- **Hooks** — `before_model`, `before_tool`, `after_tool`. Approval gates,
  cost tracking, logging.

### 3. Compression is the only legal context mutation
Context grows, never shrinks in-place. Compression fires at a **session
boundary** (= between user messages, after the model's turn completes and
before the next user prompt is processed — never mid-turn). Summarize the
middle, protect head + tail, never split tool_call/tool_result pairs. Gated
by real `prompt_tokens`, anti-thrash (skip if last 2 saves <10%).

---

## Core loop contract

```
reconcile_dangling_tool_calls(messages)          # cross-restart: fill interrupted turn (Gotcha 18)
while True:
    check(cancel_token)                          # cancellation check

    api_msgs = deepcopy(messages)                # ONE deepcopy/turn = enforced isolation (Gotcha 17)
    inject_ephemeral(api_msgs)                    # RAG/memory: mutate freely, can't leak
    api_msgs = hooks.before_model(api_msgs)      # hook can inspect/transform
    mark_cache_breakpoints(api_msgs)             # system + last 3, in place (after hooks)

    response = provider.stream(                  # cancel_token plumbed through
        system, api_msgs, tools, cancel=cancel_token)

    messages.append(response)
    persist(response)                            # BEFORE executing tools

    if response.stop_reason != "tool_use": break

    # INVARIANT: every tool_use block gets EXACTLY ONE tool_result —
    # denial, cancellation, and error all emit one, or the next call 400s.
    # Track (call, result) PAIRS so after_tool sees the SAME (repaired) call
    # that before_tool saw — never the original pre-repair object.
    processed = []                               # (call, result) in call order
    for call in response.tool_calls:
        call = repair_if_malformed(call)         # repair FIRST (rebinds call)

        if cancel_token.is_set():                # cancelled mid-batch:
            processed.append((call, error_result(  # fill remaining w/ synthetic
                call.id, "cancelled")))          # results so the turn is valid
            continue

        call = hooks.before_tool(call)           # approval sees repaired call
        if denied(call):
            processed.append((call, error_result(   # DENIAL still needs a result
                call.id, "denied by approval hook")))
            continue

        processed.append((call, dispatch(call, cancel_token)))  # real exec (below)

    for call, result in processed:               # same call object both ends
        hooks.after_tool(call, result)
        messages.append(tool_result(
            call.id, result.content, is_error=result.is_error))

    if processed and all(r.terminate for _, r in processed): break   # batch stopped
    if budget.exhausted: grace_call(); break
```

**Concurrency model** — `dispatch()` runs the batch. The default is **bounded
parallelism** via a `ThreadPoolExecutor`; the batch degrades to sequential if
ANY call targets a tool with `execution_mode="sequential"` or
`parallel_safe=False`. Results are reassembled in original call order (the
`zip` above) so `tool_result` ordering is deterministic regardless of finish
order. A tool that raises returns an `error_result` (never propagates) so the
invariant holds. The loop is **synchronous + thread-based** (see Phase 1
decision); `CancelToken = threading.Event`.

Key behaviors:
- **The tool_result invariant** — every `tool_use` block emitted by the model
  gets EXACTLY ONE `tool_result` in the following turn. Denial, cancellation,
  and tool exceptions all emit a synthetic `error_result` rather than skipping.
  Skipping (the obvious `if denied: continue`) is a guaranteed 400 on the next
  `provider.stream()` due to `tool_use`/`tool_result` mismatch. This is the
  single most common harness bug — the loop is built around it.
- **Persist before execute** — crash-resilience for destructive tools.
- **Tool-call repair** in the loop — models regularly emit plain-text tool
  calls (bracket `[name]{json}[/name]`, XML-ish `<function=name>`, Harmony
  syntax). Balanced-brace JSON finder + argument coercion (string→number,
  string→boolean, null→default) before schema validation. **Repair fires
  before approval hooks** — plugin authors see (and approve) the repaired
  call, so what they approve is what runs.
- **Cancellation** — a `CancelToken` (`threading.Event`) is plumbed through the
  loop, `provider.stream()`, and `tool.execute()`. ACP `cancel` sets it.
  Provider raises on cancel; tools check cooperatively. A mid-batch cancel
  still fills remaining `tool_use` blocks with synthetic `error_result`s so the
  invariant holds.
- **`before_model` hook** fires after ephemeral injection, before cache
  breakpoints. Hooks can inspect or transform the message list (e.g. inject
  guardrails, log prompts). Runs on the copy, not the live list.
- **`ToolResult.terminate`** — if every tool in a batch sets `terminate: True`,
  the loop stops. A single non-terminating tool keeps the loop alive.
- **`ToolResult.is_error`** — maps to Anthropic's `is_error: true` on
  `tool_result` blocks. Models behave noticeably better when failures are
  explicit vs. stuffed into content text.
- **Concurrency** — synchronous loop with a `ThreadPoolExecutor` for the tool
  batch. Bounded parallelism by default; degrades to sequential if any call in
  the batch is `execution_mode="sequential"` or `parallel_safe=False`. Results
  reassembled in call order for deterministic `tool_result` ordering.
- **Deferred tool resolution** — tool declarations (schemas) are frozen and
  always-sent; tool handlers are lazy-loaded on first call.
- **Iteration budget** — consume/refund per turn; "code-only turn" = a turn
  where the model called only `bash`/`execute`-tagged tools with no text
  output (pure execution, not reasoning). These get refunded. One grace call
  when exhausted.
- **Streaming by default** — use `stream().get_final_message()` even for
  non-streaming; SSE-only gateways crash `.create()` callers.

## Type contracts

```python
CancelToken = threading.Event  # set() to cancel; checked cooperatively

@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict                    # JSON Schema
    handler: Callable[..., ToolResult]    # may be lazy-loaded (None until first call)
    parallel_safe: bool = False
    requires_approval: bool = False
    execution_mode: str = "parallel"      # "parallel" | "sequential"
    tags: tuple[str, ...] = ()            # e.g. ("execute",) for budget refund

@dataclass
class ToolResult:
    content: list[TextContent | ImageContent]
    is_error: bool = False                # maps to Anthropic is_error on tool_result
    terminate: bool = False               # hint: if ALL tools in batch terminate, loop stops

def error_result(call_id: str, msg: str) -> ToolResult:
    # synthetic result for denial / cancellation / tool exception — preserves
    # the one-tool_result-per-tool_use invariant
    return ToolResult(content=[TextContent(msg)], is_error=True)

# ProviderProfile is PURE DATA (a frozen dataclass). Behavior lives in a
# separate ProviderHooks Protocol so the data/behavior split is unambiguous.
@dataclass(frozen=True)
class ProviderProfile:
    id: str
    supported_models: tuple[str, ...]
    max_tokens_by_model: dict[str, int]
    # capability flags / quirks as plain fields, e.g.:
    supports_thinking: bool = True
    reasoning_in_extra_body: bool = True  # where reasoning config goes (the split)

class ProviderHooks(Protocol):
    """Optional behavior overrides — one implementation per provider."""
    def prepare_messages(self, msgs: list) -> list: ...
    def build_api_kwargs(self) -> tuple[dict, dict]: ...      # → (extra_body, top_level_kwargs)
    def fetch_models(self) -> list[str]: ...

class Provider(Protocol):
    profile: ProviderProfile      # data
    hooks: ProviderHooks          # behavior
    def stream(self, system: str, messages: list, tools: list[Tool],
               cancel: CancelToken | None = None) -> Response: ...

class Hook(Protocol):
    def before_model(self, messages: list) -> list: ...       # after ephemeral inject, before cache bp
    def before_tool(self, call: ToolCall) -> ToolCall | Deny: ...   # sees REPAIRED call
    def after_tool(self, call: ToolCall, result: ToolResult) -> None: ...
```

## ACP — the interop layer

JSON-RPC 2.0 over stdio. Both OpenClaw (TS) and Hermes (Python) speak it.
Our harness speaks it too → compatible with both ecosystems + ACP-capable
editors (Zed, etc.) regardless of implementation language.

**SDK PINNED (verified against `agent-client-protocol` 0.10.1, PROTOCOL_VERSION=1):**
- We subclass `acp.Agent` and run it with `acp.run_agent(agent)` (defaults to
  stdin/stdout). The ABC methods we implement (pythonic names → wire methods):
  `initialize` → `initialize`; `new_session(cwd, ...)` → `session/new`;
  `prompt(prompt, session_id, message_id=None)` → `session/prompt` (**NOT** bare
  `prompt`); `cancel(session_id)` → `session/cancel`; `close_session(session_id)`
  → `session/close`; `load_session(cwd, session_id)` → `session/load` (**stable** route,
  not unstable — verified in the router; powers store-backed crash recovery, Gotcha 19).
  Optional/not-implemented: `list_sessions`, `authenticate`.
- `on_connect(self, conn: acp.Client)` hands us the **client** to push to the editor:
  - `client.session_update(session_id, update)` — `update` built by helpers
    `update_agent_message_text(text)→AgentMessageChunk`,
    `update_agent_thought_text(text)→AgentThoughtChunk`,
    `start_tool_call(tool_call_id, title, kind=, status=)→ToolCallStart`,
    `update_tool_call(tool_call_id, status=, content=)→ToolCallProgress`.
  - `client.request_permission(options, session_id, tool_call)` → response with
    `.outcome` = `AllowedOutcome(option_id=...)` | `DeniedOutcome()`.
    `PermissionOption(kind, name, option_id)`; kinds: `allow_once`, `allow_always`,
    `reject_once`, `reject_always`.
- `PromptResponse.stop_reason` ∈ `{end_turn, max_tokens, max_turn_requests,
  refusal, cancelled}` (no `tool_use` — that's internal; map ours: end_turn→
  end_turn, max_tokens→max_tokens, cancelled→cancelled).
- `ToolKind` ∈ `{read, edit, delete, move, search, execute, think, fetch,
  switch_mode, other}` — map our tool `tags` onto these.
- All SDK models are pydantic (snake_case fields, serialize to camelCase wire).

---

## MCP — inheriting the tool catalogue

We are a **client** of external MCP servers (consume their `tools/list` +
`tools/call`); we are NOT *being* an MCP server — ACP already covers "something
drives the harness". MCP was the test of whether the narrow waist holds: external
tools drop in through Tools/Providers/Hooks with **zero `core/` change**. It does.

- **The closure handler is the proof, not a shortcut.** Each MCP tool becomes a
  `Tool` whose `handler` is a closure over the live client. A `handler=None`
  placeholder would have forced the *loop* to grow a lazy-resolution step — a core
  change, exactly what the waist forbids. The closure satisfies both halves:
  frozen (schemas fetched once at startup, hashed into the cache prefix) + lazy
  (no `tools/call` round-trip until the model invokes it). `_make_handler` keeps
  the **original** tool name for dispatch; the model sees the namespaced name.
- **One shared runtime.** `McpRuntime` runs ONE asyncio loop on a dedicated
  thread for ALL servers; each session stays open for the conversation. Calls
  marshal across via the ACP `AsyncBridge.call` primitive (block the worker for a
  result). Connection setup is the expensive part — never a loop per call.
- **Frozen-set timing (Law 1).** `tools/list` for every server MUST complete
  *before the first API call* — those declarations are in the cache-hashed prefix
  and the set is frozen for the conversation. The loader blocks at startup with a
  bounded `connect_timeout`; a server that misses the window is dropped, not
  awaited mid-conversation (that would invalidate the cache). Wired before
  `Agent.run()` in both the CLI (`--mcp`) and the ACP server, torn down in a
  `finally` (`client.close()` + `runtime.shutdown()`).
- **Config** is the standard `mcpServers` JSON shape (Claude Desktop / OpenClaw),
  so users paste existing configs. `load_mcp_tools(servers) -> (tools, clients,
  runtime)`; the caller owns teardown of what it gets back.

Package is `src/harness/...` (importable as `harness`). `[x]` = built & tested,
`[ ]` = not yet.

```
harness/
├── CLAUDE.md  README.md  pyproject.toml
├── src/harness/
│   ├── core/
│   │   ├── [x] types.py        # Tool, ToolResult, Deny, error_result, Provider/Hooks, Response
│   │   ├── [x] loop.py         # the agentic loop + the tool_result invariant
│   │   ├── [x] budget.py       # iteration budget (consume/refund/grace)
│   │   ├── [x] repair.py       # tool-call repair (3 syntaxes + coercion + promotion)
│   │   └── [x] context.py      # compression engine (Law 3 — the one legal mutation)
│   ├── providers/
│   │   ├── [x] cache.py        # cache-breakpoint placement (system_and_3)
│   │   └── [x] claude.py       # Anthropic Messages API provider
│   ├── tools/                  # [x] read, edit, bash, glob, grep  (+ default_tools())
│   ├── hooks/                  # [x] approval.py, cost.py
│   ├── session/
│   │   └── [x] store.py        # SQLite + WAL: system restore, msg log, compaction log
│   ├── acp/
│   │   ├── [x] bridge.py       # sync-loop <-> asyncio bridge (run_blocking/emit/call)
│   │   ├── [x] events.py       # session_update builders + tool-kind/stop-reason maps
│   │   └── [x] server.py       # HarnessAgent(acp.Agent): initialize/new/prompt/cancel/close
│   ├── mcp/
│   │   ├── [x] client.py       # McpRuntime (one shared loop/thread) + StdioMcpClient (call timeout)
│   │   └── [x] registry.py     # tools/list → frozen Tools (closure handler); name sanitize + dedup
│   ├── [x] cli.py              # harness CLI entry (in-process Agent.run, TTY approval, session resume)
│   └── [x] testing.py          # scripted FakeProvider + scenario provider (single source; conftest re-exports; HARNESS_PROVIDER=fake)
├── adapters/openclaw/          # [x] OpenClaw adapter (Phase 3): harness.ts / acp-client.ts / index.ts (tsc-green vs real SDKs) + DESIGN.md + probes
└── tests/                      # [x] 155 (152 offline + 3 live-MCP; incl. serve/EOF, per-session-model, cancel, cache-stability, crash-resume, fail-closed persist, ACP recovery, tool-lifecycle+timeout-wiring (CLI+ACP)+serial-isolation, provider/IO fault-injection); test_invariant.py is never-regress
```

## Phase plan

| Phase | What                                                        | Status         |
|-------|-------------------------------------------------------------|----------------|
| 0     | Architecture strip-down (OpenClaw + Hermes + ACP)           | **DONE**       |
| 1     | Design: lock language/async, write real contracts, scaffold | **DONE**       |
| 2     | MVP: loop + ClaudeProvider + 5 tools + hooks + repair + compression + ACP + CLI + MCP | **DONE** (110 tests green) |
| 3     | OpenClaw harness-plugin adapter (TS ↔ Python over ACP)      | **BUILT + wire-verified** |
| 4     | Publish (ClawHub / npm or PyPI) + docs                      | —              |

**Phase 2 COMPLETE (110 offline tests green).** Everything built and tested offline
(scripted FakeProvider + in-process FakeMcpClient): core loop + invariant, Claude
provider + cache breakpoints + streaming (`on_chunk`), 5 tools (read/edit/bash/glob/grep),
approval/cost hooks, SQLite+WAL store, tool-call repair, compression engine (Law 3),
ACP server (`acp/server.py` — `HarnessAgent(acp.Agent)` via `acp.run_agent`,
`AsyncBridge` marshals sync loop ↔ asyncio), the **CLI** (`cli.py` — in-process
`Agent.run()`, TTY approval gate via `cli_approval` hook, session resume, Ctrl-C
cancellation), and the **MCP client** (`mcp/` — closure handlers, one shared
runtime, name sanitize + deterministic dedup, call timeout + drain). SDK pinned to
agent-client-protocol 0.10.1. Compression is **caller-driven**:
`Compressor.compress_if_needed(messages, last_response, store=, session_id=)`. A
live API run needs `ANTHROPIC_API_KEY`.

The high-leverage MCP bridge was folded into Phase 2 (before the Phase 3 adapter,
where "what tools the harness ships with" becomes externally visible). Remaining
MCP polish for the safety pass — both cosmetic, neither conversation-fatal:
`McpRuntime`'s `AsyncBridge` spins an unused 8-thread pool (`run_blocking` never
called there); and mutating MCP tools are `parallel_safe=True` (consider
`parallel_safe=spec.read_only`). A `[mcp]` extra in `pyproject.toml` is still TODO
(the `mcp` package is currently a manual install for live stdio).

**Phase 3 — OpenClaw adapter BUILT + wire-verified (`adapters/openclaw/`).**
A TypeScript OpenClaw plugin that `registerAgentHarness` + spawns the Python `harness-acp`
as an ACP subprocess: `harness.ts` (AgentHarness factory, ~20-field result builder, `AcpClient`
seam), `acp-client.ts` (spawn a **resolved** Python — locate-not-bundle (config/env/managed-venv/PATH) via the `windows-spawn` helper, fail-closed — → `@agentclientprotocol/sdk@1.0.0`
client → `initialize` with `PROTOCOL_VERSION==1` asserted → `ActiveSession`; `request_permission`
deny-by-default), `index.ts` (`definePluginEntry`). **`tsc --noEmit` green against the real
`openclaw@2026.6.10` + ACP SDKs** — every contract fact pinned from real source via clean-channel
`curl`, never WebFetch paraphrase (full provenance in `adapters/openclaw/DESIGN.md`). Verified by
RUNNING, not trusting: a handshake probe (`probe.mjs`) and a deterministic fake round-trip
(`probe-roundtrip-fake.mjs`, $0) asserting #1 streaming → `onPartialReply`, #2 `runAgentEndSideEffects`
on finalize, #3 result builder populated, #4 the **tool_result invariant surviving the ACP round-trip**,
#5 per-session **model propagation** (model rides in `session/new`'s `_meta`, accepted across the wire),
#6 **trust-floor rejection** of an unadvertised model id at session creation. The fake provider lives in
`src/harness/testing.py` (single source; `acp/server.py:main()` lazy-loads it gated by
`HARNESS_PROVIDER=fake` — never hot in production). Bugs/decisions fixed en route: (a) `server.py:main()`
called the async `acp.run_agent` without `asyncio.run` (the entry never served — a subprocess probe
caught what method unit tests can't); (b) **model propagation uses the STABLE channel — `_meta` on
session/new (the ACP router spreads it into the `new_session` kwargs), NOT ACP `set_session_model`: a
wire probe showed that method is registered `unstable=True` (returns `method_not_found` unless
`use_unstable_protocol`), and a core path must not rest on an unstable protocol method.** The agent
validates the id against `supported_models` and builds the provider per-session; per-attempt because the
adapter opens a fresh session each attempt (NOT spawn-time `HARNESS_MODEL`, which a reused process would
pin). Cancellation across the wire is also done — `params.abortSignal` → ACP `session/cancel` →
`CancelToken` → `aborted`, verified by fake round-trip #7 ($0). `contextEngineHostCapabilities` is
resolved too — declare **none** (we run our own context/compaction and don't host OpenClaw's context
engine; declaring any would falsely claim its hooks → OpenClaw fails closed instead of corrupting state).
**Remaining (DESIGN.md §8):** real-API smoke gate (request-acceptance, pre-publish, needs
`ANTHROPIC_API_KEY` — held, no spend yet) + ClawHub publish (Phase 4 packaging BUILT + verified
offline — locate-not-bundle manifest/dist/README; remaining: `clawhub --dry-run` + wheel + scanner re-pin).

**Hardening (adversarial — Buckets A & B, post-Phase-3).** Robustness beyond the happy
path, each item proven by a runnable probe/test (not asserted), with details in the
Gotchas:
- **Per-turn isolation (L8 → Gotcha 17):** the loop deep-copies history before
  hooks/injectors run, so they cannot alias canonical state. `tests/test_cache_stability.py`.
- **Crash/restart reconciliation (Bucket B → Gotcha 18):** the tool_result invariant
  survives a process death mid-tool-execution — a dangling tool_use is reconciled on
  resume with a synthetic result. `tests/test_hardening_resume.py`.
- **Fail-closed persistence (Bucket B → Gotcha 19):** persist-before-append/execute; a
  write failure (disk full / WAL exhaustion) raises without advancing memory or running a
  tool un-recorded; single-row writes are atomic.
- **Store-backed recovery (Bucket B → Gotcha 19):** the ACP server persists + resumes via
  the stable `session/load`; the CLI via `--session-id`. `tests/test_hardening_acp_recovery.py`.
- **Tool lifecycle under fault (Bucket A → Gotcha 20):** one tool_result per tool_use
  under timeout, mid-batch & mid-execution cancel, denial, exception, SIGINT, and provider
  disconnect; the `tool_timeout` backstop (default 900s via `HARNESS_TOOL_TIMEOUT`) is
  wired into both shipping drivers (CLI + ACP). `tests/test_hardening_lifecycle.py` +
  `test_invariant.py`.
- **Provider/IO fault injection (Bucket C):** fail-closed at the provider/stream boundary —
  500 at request, fault mid-stream (after partial deltas), malformed payload (normalizer),
  stream-cancel sever, and a broken pipe on the write side (fire-and-forget `emit` can't
  wedge/corrupt the loop). Each raises and leaves history reconcilable; no zombie threads.
  `tests/test_hardening_faults.py`. ("duplicate responses" and "kill -9" are non-faults at
  our layer — no double-ingest path; SIGKILL is uncatchable, recovered via Bucket B.)

Open (next): **descoped-D** (SQLite WAL contention + ACP serialization losslessness only —
subagents/parallel-adapters are out of scope). Specifically deferred to D: `store.append_message`
does `SELECT MAX(seq)+1` then `INSERT` as two statements with no per-session transaction/lock
or `UNIQUE(session_id, seq)`, so two writers on the SAME session could race to one seq. Not
reachable in single-agent use (the loop persists serially; the ACP bridge serializes prompts
per session), but it IS the D fix when concurrent same-session writes become real (BEGIN
IMMEDIATE + a unique index). The OpenClaw adapter's single-session-per-client ceiling is the
adapter's, not the agent's (the Python server is multi-session) — see its DESIGN.md §6.

Phase 0 findings passed an Opus design review (10 contract gaps reconciled:
`before_model`/`terminate`/`is_error` wired into the loop, cancellation seam
added, frozen-declarations-vs-lazy-handlers split clarified, provider
`(extra_body, top_level_kwargs)` surfaced, session-boundary + code-only-turn
defined). Optional Gemini cross-reference is a sanity check, not a blocker.

## Phase 1 deliverables (concrete next steps)

1. **Language** → **LOCKED: Python.**
2. **Concurrency** → **LOCKED: sync loop + `ThreadPoolExecutor`**, `threading.Event`
   cancel. ACP server runs the loop in a worker thread.
3. **Write real type contracts** → `src/core/types.py`: `Tool`, `ToolResult`
   (with `is_error`/`terminate`), `error_result`, `ToolCall`, `Response`,
   `ProviderProfile` (frozen data) + `ProviderHooks` (Protocol), `Hook`,
   `CancelToken`. Promote the pseudocode above to actual dataclasses.
4. **Define the 5 tool input schemas** → read/edit/bash/glob/grep JSON Schemas.
5. **Design the SQLite session schema** → sessions / messages / tool-calls /
   system-prompt tables; WAL mode; connection discipline.
6. **Scaffold `pyproject.toml`** → deps (anthropic, agent-client-protocol),
   entry points (`harness`, `harness-acp`), metadata.

## Phase 1 decisions (open)

| Decision          | Option A                          | Option B                          | Leaning        |
|-------------------|-----------------------------------|-----------------------------------|----------------|
| **Language**       | TypeScript (native OpenClaw plugin, TypeBox schemas) | Python (ergonomics, Hermes coding DNA, pip plugins) | **LOCKED: Python.** ACP dissolves interop; fastest for us. OpenClaw adapter = TS process spawning Python harness as ACP subprocess (not a trivial shim — budget for it) |
| **MVP tools**      | read, edit, bash, glob, grep      | + tool-search for lazy loading    | Start with 5, add search if needed |
| **Persistence**    | SQLite (Hermes)                   | JSONL (OpenClaw)                  | SQLite — crash-safe, query-friendly. **Note:** CLI + ACP server + future UI = multiple writers → must use WAL mode + connection discipline from day one |
| **Edit format**    | string-replace only               | + V4A patch for GPT/Codex         | string-replace first (Claude-native), V4A later |
| **Compression**    | aux Haiku call                    | local summarizer                  | Haiku ($1/5M) — cheap, good enough |
| **Concurrency**    | sync loop + `ThreadPoolExecutor`  | `asyncio` event loop              | **LOCKED: sync + threads.** Simpler to reason about, blocking tool I/O (bash/file) is natural, `threading.Event` cancel. ACP server runs the sync loop in a worker thread |
| **Cancellation**   | `threading.Event` (CancelToken)   | `asyncio.CancelledError`          | **LOCKED: `threading.Event`** (follows from sync concurrency) |

## Claude API reference

| Model | Cost (in/out per 1M) | Use |
|-------|----------------------|-----|
| `claude-opus-4-8` | $5 / $25 | Default agent model, `effort: "xhigh"` |
| `claude-sonnet-4-6` | $3 / $15 | Fast/cheap agent |
| `claude-haiku-4-5` | $1 / $5 | Compression, classification |

- Agentic loop: `stop_reason ∈ {end_turn, tool_use, max_tokens, stop_sequence,
  pause_turn, refusal}`. Only `tool_use` continues the loop; everything else stops.
  **`refusal`** is a 200 (not an error) — a safety decline; preserve it, don't
  collapse to `end_turn` (we don't). `pause_turn` only occurs with server-side
  tools (none in MVP).
- Thinking: `thinking: {type: "adaptive"}` + `output_config: {effort}`. On Opus
  4.8/4.7 `budget_tokens` is **removed** (400s); adaptive is the only on-mode.
- Context: 1M tokens, 128K max output (we cap `max_tokens` lower per profile)
- Cache: `tools` + `system` = cached prefix; adding tools invalidates it.
  **Minimum cacheable prefix on Opus 4.8 = 4096 tokens** (Sonnet 4.6 = 2048) —
  a system prompt shorter than that silently won't cache (`cache_creation_input_tokens: 0`).
- MCP: we do **not** use `anthropic.lib.tools.mcp` — those helpers target Anthropic's
  SDK tool runner, and we run our own loop. Our `mcp/` client bridges MCP tools into
  our registry directly (that's the whole narrow-waist point).

## Gotchas (don't relearn)

0. **One tool_result per tool_use — ALWAYS.** Denial, cancellation, and tool
   exceptions must each emit a synthetic `error_result`, never skip. A skipped
   result 400s the next `provider.stream()` (`tool_use`/`tool_result` mismatch).
   The most common harness bug; the loop is built around the invariant.
1. **Tool-call repair is first-class.** Three syntax families + arg coercion at
   the framework level. Not an afterthought.
2. **Persist BEFORE append, BEFORE execute (fail-closed).** Each turn is written to
   the store before it's appended to in-memory history and before any tool runs. If the
   write fails (disk full / WAL exhaustion) it raises and neither store nor memory
   advances — no divergence, no tool side effects without a durable record. A single
   message write is one atomic SQLite transaction (no partial rows); a mid-turn failure
   leaves at worst a dangling tool_use, which resume RECONCILES (Gotcha 18). See Gotcha 19.
3. **`(extra_body, top_level_kwargs)` split.** Providers disagree where reasoning
   config goes. The `ProviderProfile` must surface both via `build_api_kwargs()`.
4. **Anti-thrash compression.** Skip if last 2 attempts saved <10% tokens.
5. **Head protection decay.** First compress protects early task turns; subsequent
   ones decay to 0 so they don't fossilize.
6. **System prompt in session store, not memory.** Fresh-process-per-turn gateways
   need it restored from SQLite. Restore fail = WARN.
7. **Stream by default.** `stream().get_final_message()` even for "non-streaming"
   calls. SSE-only gateways lie.
8. **Declarations frozen, handlers lazy.** Tool schemas (what the model sees) are
   frozen at conversation start. Tool handlers (the code) can be lazy-loaded on
   first call. Don't confuse these — only schema changes invalidate the cache.
9. **LSP delta diagnostics.** Snapshot before write, report only NEW diagnostics.
   Auto-blacklist broken servers.
10. **Model-aware edit format.** Claude → string-replace. GPT/Codex → V4A patch.
    Steer via coding context, not user config.
11. **Repair before approval.** The loop repairs malformed tool calls before
    running `before_tool` hooks. Plugin authors see and approve the repaired
    call — what they approve is what runs.
12. **SQLite WAL from day one.** CLI, ACP server, and future web UI all touch the
    same session DB. Use WAL mode + connection discipline, not default journal.
13. **Cache breakpoints are Anthropic-shaped.** `cache_control` markers are
    Anthropic-specific. Other providers must ignore or translate them — the
    provider contract is "Anthropic shape, others adapt," not truly agnostic.
14. **MCP tool names: sanitize BOTH halves, cap the FULL string at 64, dedup
    deterministically.** Anthropic requires `^[a-zA-Z0-9_-]{1,64}$`; the name
    rides in the frozen `tools` prefix, so ONE bad name 400s the WHOLE
    conversation — and this *evades* the drop-the-bad-server net, because a weird
    name passes `tools/list` cleanly and only detonates at the API boundary.
    Sanitize is ASCII-only (`str.isalnum()` is `True` for `'ï'` — must also gate
    on `c.isascii()`). Sanitize + truncate are both many-to-one, so collisions are
    inevitable → resolve by suffixing EVERY member of a colliding group with a
    `sha1(original)[:6]`, seeded by the pre-sanitization identity so it's a pure
    function of inputs (NOT iteration order) — else the frozen set isn't
    byte-stable across a `--session-id` resume → silent cache miss. Per-server +
    cross-server passes (`namespaced` / `_resolve_collisions` in `mcp/registry.py`).
15. **MCP calls need a timeout (and cancel-on-timeout).** Failure isolation must
    cover a server that *hangs*, not just one that errors or won't start. A
    `tools/call` with no timeout blocks the agent thread forever (and `cancel`
    won't help — the handler doesn't see it). Thread a `call_timeout` through;
    `McpRuntime.call` cancels the scheduled coroutine on timeout so it doesn't keep
    running on the shared loop. `TimeoutError` then becomes an `error_result` —
    the invariant holds for hangs too.
16. **MCP runtime teardown drains before stopping the loop.** `McpRuntime.shutdown`
    cancels in-flight tasks and *awaits* their unwind before `loop.stop()` —
    otherwise a call still timing out at session end leaks ("Task was destroyed but
    it is pending") and can orphan child I/O. `close()` (graceful) then
    `runtime.shutdown()` (hard) is the order.
17. **Per-turn isolation is ENFORCED by a deep copy (not a convention).** The loop
    hands hooks/injectors `copy.deepcopy(messages)` at the top of `loop.run`, so an
    `inject_ephemeral` / `before_model` may mutate the view **any** way — reassign OR
    in-place `.append`/`+=` — and it physically cannot reach canonical (persisted)
    history. Cost stays at exactly one deepcopy per turn: `mark_cache_breakpoints`
    (`providers/cache.py`) marks the tail **in place** afterward, replacing the old
    shallow-copy + `apply_cache_breakpoints`-deepcopy pair (same cost, stronger
    guarantee). `apply_cache_breakpoints` remains the pure (deepcopying) variant for
    callers that don't own an isolated list. Empirically audited (7/7 cache-stability
    probes; the prior in-place-leak finding is now closed) and guarded by
    `tests/test_cache_stability.py`.
18. **The tool_result invariant survives a crash/restart (cross-turn reconciliation).**
    The in-turn invariant (one `tool_result` per `tool_use`) only spans ONE turn — and
    a turn is two separate writes (the assistant `tool_use` turn is persisted BEFORE
    tools run, the `tool_result` turn AFTER). A crash between them leaves a dangling
    `tool_use` that 400s the resumed request. `reconcile_dangling_tool_calls`
    (`core/loop.py`), called at the top of `run()`, in the CLI resume path, AND in the
    ACP `prompt` path (so `session/load` recovery is covered too), folds a synthetic
    is_error `tool_result` into the following user turn for any unmatched id. It is
    **idempotent**, **re-executes nothing** (no duplicate side effects — history
    `tool_use` blocks are never re-run), preserves role alternation (fold-into-next-user,
    not a second user turn), and the reconciled turn is persisted in final form
    (append-only). Guarded by `tests/test_hardening_resume.py` +
    `tests/test_hardening_acp_recovery.py`.
19. **Persistence fails closed; recovery is store-backed.** A write failure surfaces
    (the store never silently swallows — `tests/test_hardening_resume.py`), single-row
    writes are atomic (no partial state), and persist-before-append/execute (Gotcha 2)
    means a failure can't advance memory past durable state or run a tool without a
    record. Both drivers persist to the SAME SQLite store: the CLI resumes via
    `--session-id`; the ACP server resumes via the stable `session/load` method
    (`load_session` restores the system prompt byte-for-byte + the message log into a
    fresh process). Session ids are durable (uuid) when a store is attached, a readable
    counter when in-memory (tests/probes). Guarded by `tests/test_hardening_acp_recovery.py`.
20. **Tool-timeout backstop (liveness under a non-cooperative hang).** Tools self-limit
    (bash subprocess timeout, MCP `call_timeout`) and long-runners cooperate via `cancel`.
    For a tool that does NEITHER — hangs *and* ignores `cancel` — `AgentConfig.tool_timeout`
    bounds each call: on expiry the loop emits a timeout `error_result` (invariant holds)
    and STOPS WAITING (`_dispatch_timed`). Both shipping drivers (CLI + ACP server) wire
    `resolve_tool_timeout()` → default 900s via `HARNESS_TOOL_TIMEOUT` env. Parsing is
    FAIL-CLOSED: unset → default; `n>0` → n; `n<=0` → disabled (the only way to turn it
    off); an unparsable typo (`900s`, `ten`) → keeps the default + WARN (never silently
    drops the floor). 900s is above bash's 600s ceiling so bash's own subprocess timeout
    always fires first with a cleaner error. Consequence of a non-None default: production
    ALWAYS takes the `_dispatch_timed` path, so every tool runs in a worker thread (never
    inline) — the inherent cost of an always-on backstop, since you cannot abandon an
    inline call. The library default stays `None` (inline), so embedders opt in.
    **Serial isolation under timeout:** because a timeout does NOT kill the worker (no way
    to kill a Python thread), once a *sequential* call times out the loop FAIL-STOPS the
    rest of that serial batch — remaining calls get a synthetic 'skipped' result rather
    than starting while the timed-out tool's thread may still be mutating state. This keeps
    `execution_mode="sequential"` / `parallel_safe=False` honest (no overlap) AND the
    invariant (one result per call). Parallel calls share one batch deadline (~timeout
    total, not N×). Ceiling: the hung worker still leaks until process exit — the LOOP
    stays live and the invariant holds regardless. Guarded by
    `tests/test_hardening_lifecycle.py` (7 lifecycle faults + 5 resolve/CLI-wiring +
    serial-isolation fail-stop) + the ACP-wiring test in `test_hardening_acp_recovery.py`
    + `test_invariant.py`.
