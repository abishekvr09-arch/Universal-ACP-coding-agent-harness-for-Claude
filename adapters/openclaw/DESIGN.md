# Phase 3 — OpenClaw Adapter: Design

Status: **BUILT + wire-verified.** All TypeScript compiled (`tsc --noEmit` green vs
`openclaw@2026.6.10` + `@agentclientprotocol/sdk@1.0.0`); fake round-trip #1–#7 green
($0); 155 pytest green. Phase 4 packaging **BUILT + verified offline** (manifest, Python resolution,
compiled `dist/`, README; `npm pack`=7 files). Remaining (publish-gated): real-API smoke gate (held) +
`clawhub --dry-run` + wheel/PyPI-name + scanner re-pin.

> **Revision 2 — pinned from `docs/plugins/sdk-agent-harness.md` + `extensions/codex/harness.ts` (curl raw bytes).**
> Resolved the gate-zero delivery question and four open items; corrected one factual
> error (see below). Key result: **delivery + streaming are host-owned callbacks the
> harness invokes** (`params.onPartialReply`, `params.onAgentEvent`) — the returned
> `assistantTexts` is a *record/classification*, not the delivery channel. The
> "output never reaches the user" risk is refuted. Reference impl for our exact
> architecture (plugin harness wrapping an external stdio agent process):
> `extensions/codex/harness.ts`.
> Corrections: `runtimePlan` is **host policy to USE, not DROP** (§3); native harnesses
> **MUST** call `runAgentEndSideEffects(...)` and should **pair a provider** with the
> harness (§2, §5).

The adapter is a TypeScript OpenClaw plugin that registers an `AgentHarness` and
spawns the Python harness (`harness-acp`) as a long-lived ACP subprocess over
stdio, translating OpenClaw's harness contract ↔ ACP in both directions.

---

## 0. Verification status — how each fact below was established

Everything load-bearing was pulled this session from raw bytes
(`raw.githubusercontent.com/openclaw/openclaw/main/...`) via `curl`, not WebFetch
paraphrase. `git ls-remote` + REST API + a `torvalds/linux` control established the
repo is real (HEAD advanced `d83cd282`→`b58e6e07` between two runs — a live repo).

| Fact | Source (raw) | Status |
|------|--------------|--------|
| `AgentHarness` = 5-facet intersection | `src/agents/harness/types.ts:134-138` | curl-verified |
| `RunCapability` surface (`id,label,supports,runAttempt`) | same file | curl-verified |
| `supports(ctx)` context + return union | same file | curl-verified |
| `AttemptParams = EmbeddedRunAttemptParams` | same file (alias) | curl-verified |
| `EmbeddedRunAttemptParams` fields | `src/agents/embedded-agent-runner/run/types.ts:54-112` | curl-verified |
| `EmbeddedRunAttemptResult` fields | same file `:115-246` | curl-verified |
| `registerAgentHarness(harness)` | `src/plugins/types.ts:2758` | curl-verified |
| `beforeToolCall` is internal (not on the seam) | `packages/agent-core/src/harness/agent-harness.ts:500` | curl-verified |
| Agent event types (`message_start/end`, `turn_end`, `agent_end`) | same file `:581-625` | curl-verified |
| `dispose?()` real, not crash-safe | types.ts:131 + issue #18420 (REST) | curl-verified |
| Prompt input = `params.prompt`; streaming = `params.onPartialReply` / `params.onAgentEvent` | `docs/plugins/sdk-agent-harness.md` | curl-verified |
| **Delivery is host-owned** (harness feeds callbacks; result is a record) | same doc ("channel reply callbacks and streaming callbacks" are core-owned) | curl-verified |
| `runtimePlan` is host policy to USE, not drop | same doc | curl-verified |
| Required: `runAgentEndSideEffects()` after finalize; pair a provider | same doc | curl-verified |
| Reference impl (external-process harness) | `extensions/codex/harness.ts` | curl-verified |
| Exact callback signatures + `EmbeddedRunAttemptBase` member types | not pulled | OPEN — implementation-time pin |

---

## 1. Headline: the seam is thick, not thin

The project's "thin TS shim" framing is **wrong** and must be retired.
`runAttempt(params)` does not receive "a prompt." It receives OpenClaw's
fully-wired internal runner context — its own `model: Model`, `authStorage`,
`authProfileStore`, `modelRegistry`, `contextEngine`, plus lifecycle callbacks
(`onToolOutcome`, `onAttemptTimeout`, `onAttemptAbort`) — and must **return** a
~50-field `EmbeddedRunAttemptResult` including OpenClaw messaging-delivery fields.

Our Python harness has its *own* model client, auth, context/compression engine,
and tools. So the adapter:

- **ignores** OpenClaw's infra objects (model/authStorage/modelRegistry/contextEngine),
- **translates** a small config subset (provider/modelId/key/thinkLevel) into the
  subprocess so it talks to the same model,
- **maps** OpenClaw's callbacks ↔ ACP events,
- **constructs** the big result from ACP output, leaving OpenClaw-specific fields
  at defaults.

This is a substantial bidirectional translation layer with genuine impedance
mismatches (OpenClaw is a multi-platform *messaging* assistant; our harness is an
editor/ACP *coding* agent). Budget Phase 3 accordingly.

---

## 2. AgentHarness implementation shape

| Facet | MVP | Why |
|-------|-----|-----|
| `RunCapability` (`id,label,supports,runAttempt`) | **implement** | required |
| `SessionLifecycleCapability` (`dispose`) | **implement** | subprocess teardown |
| `SideQuestionCapability` | omit | params are messaging-specific (senderE164, groupChannel…) — N/A to a coding harness |
| `ClassificationCapability` | omit | optional; default `"ok"` classification is fine for MVP |
| `CompactionCapability` | omit | our harness compresses internally; OpenClaw's `compact()` targets OpenClaw's own session store |

`supports(ctx)`: return `{supported:true, priority:N}` when `ctx.provider` is one
our harness backs (Anthropic-messages) **and** `ctx.requestedRuntime` matches our
registered runtime id; else `{supported:false, reason}`. Pin `requestedRuntime`
matching against `agent-runtime-id.ts`.

**Provider pairing — defer for MVP.** The SDK guide recommends pairing a provider so
model refs + `/model` are visible. But if `supports()` can claim a stock provider
(Anthropic/OpenAI) on the resolved route, v1 ships without registering our own — add
it in v1.1 when `/model` visibility actually matters. **Verify first:** if `supports()`
can't match without our provider, keep it. Either way: once our harness claims a run,
OpenClaw will **not** replay it through another runtime, so failures surface as run
failures — clean teardown/error mapping is load-bearing.

---

## 3. `runAttempt` translation — the core, the riskiest

### Input map (`EmbeddedRunAttemptParams` → subprocess config + ACP `session/prompt`)

**HONOR** (translate into the subprocess):
- `params.prompt`, tools, images → the ACP `session/prompt` (field names curl-verified from the SDK guide)
- `provider`, `modelId`, `resolvedApiKey` / `authProfileId` → subprocess env/config so it hits the same provider/model/key
- `thinkLevel` → our effort/thinking config
- `sessionId`, `sessionFile` → ACP session identity + our SQLite store key
- `toolsAllow?` → tool gating
- **delivery/streaming callbacks** `params.onPartialReply` (assistant text) + `params.onAgentEvent` (plan/reasoning/tool events) → fed from ACP `session_update` chunks. **This is how output reaches the user.**
- lifecycle callbacks `onToolOutcome` / `onAttemptTimeoutArmed` / `onAttemptTimeout` / `onAttemptAbort` → driven from ACP tool/timeout/cancel events
- **`runtimePlan` (host policy — USE, do not mutate). MVP needs two:** `runtimePlan.delivery.isSilentPayload(...)` (honor `NO_REPLY`) and `runtimePlan.outcome.classifyRunResult(...)` (model-fallback classification). Defer `tools.normalize` / `transcript.resolvePolicy` — our harness already owns tool schemas + repair; adopt them only if a real divergence shows up.

**DROP** (we have our own — and name what breaks):
- `model`, `authStorage`, `authProfileStore`, `modelRegistry` → no OpenClaw-side auth rotation; we use the resolved key only
- `contextEngine`, `contextTokenBudget`, `contextWindowInfo` → our internal compression is authoritative. This is **expected**: the SDK guide names "a native coding-agent server that owns threads and compaction" as a valid harness. Still pin `contextEngineHostCapabilities` — omitting it makes us "unsupported for engines that declare host requirements."
- `fastMode`, `beforeAgentStartResult` → OpenClaw orchestration we don't model

### Output construction (ACP result → `EmbeddedRunAttemptResult`)

**POPULATE:** `assistantTexts`, `messagesSnapshot` (our history → `AgentMessage[]`),
`toolMetas`, `attemptUsage` (our `Usage` → `NormalizedUsage`), `aborted` /
`timedOut` / `idleTimedOut`, `agentHarnessId`, `lastAssistant`, minimal `replayMetadata`.

**DEFAULT/EMPTY:** `didSendViaMessagingTool=false`, `messagingToolSent*=[]`,
`heartbeatToolResponse`, cron fields — our harness doesn't call OpenClaw messaging tools.

> **RESOLVED (was flagged gate-zero).** User-visible output does **not** depend on
> the returned `assistantTexts`. Delivery is host-owned: the harness streams via
> `params.onPartialReply` / `params.onAgentEvent` during the turn (the SDK guide:
> core owns "channel reply callbacks and streaming callbacks"; a harness "does not
> replace channel delivery"). The result fields are a record/classification. So the
> messaging-send fields stay at defaults and that is correct — output flows through
> the callbacks, not the result. For a turn that produced no visible assistant text,
> use `classifyAgentHarnessTerminalOutcome(...)` (`empty` / `reasoning-only` /
> `planning-only`) so OpenClaw's fallback policy can decide on a retry.

---

## 4. Approval — MVP handled in-subprocess (verified)

`beforeToolCall` lives on `CoreAgentHarness`'s internal `AgentLoopConfig` via
`emitHook` ([agent-harness.ts:500]) — **not** on the plugin-facing `AgentHarness`.
A proxying harness owns its own tool dispatch and therefore its own approval.

**MVP policy (safe-by-default, headless):** the embedded context has no
interactive approver. Auto-allow read-only/`parallel_safe` tools (read/glob/grep);
**block** `requires_approval` tools (write/bash) unless an explicit config flag
opts in (the embedded analog of `--yes`). **No silent auto-approve of destructive
tools.** Handled by the existing `acp_approval` hook in the subprocess.

**v2 (deferred):** surface approval to the host via a separate
`registerHook('tool_call', …)`. Rejected for MVP — it gates *all* host tool calls
through `emitHook`, not just our subprocess's, and couples us to the host hook
system.

---

## 5. ACP client wiring + lifecycle

- Spawn `harness-acp` via `child_process.spawn` using a **resolved Python path** —
  config `pythonPath` / `OPENCLAW_HARNESS_PYTHON` / managed venv / ambient `harness`
  (§7 locate-not-bundle), via the `windows-spawn` SDK helper to design out the #71139
  `.cmd`/`.bat` shim class — speaking ACP JSON-RPC over its stdio as the **client**.
- **MCP — deferred for MVP (the code refutes the earlier claim).** This bullet once
  asserted the subprocess does an MCP `tools/list` frozen-set startup before its first
  model call. **It doesn't:** the ACP entry (`server.py:main`) ships `default_tools()`
  only — no MCP. MVP runs Claude + the 5 built-in tools end-to-end without MCP; fine
  for v1. *Follow-up:* wire MCP into `server.py` like the CLI's `--mcp` (Law 1 then
  holds — frozen set fixed at startup, before the first `provider.stream`). **When MCP
  lands, `main()` must wrap `run_agent` in `try/finally` to tear the runtime down**
  (`client.close()` → `runtime.shutdown()`): `acp.run_agent` exits cleanly on stdin
  EOF (§6) but has no reference to our runtime, so it won't shut it down. No `finally`
  needed today — there is no runtime yet.
- **Lifecycle:** spawn on first `supports()→true` `runAttempt`, keep the subprocess
  alive for the plugin's lifetime, reuse it across attempts (one subprocess, many
  ACP sessions). `register()`-time spawn trades memory for first-call latency — not
  worth it for MVP.
- **Session mapping:** OpenClaw `sessionId`/`sessionFile` ↔ ACP session.

**STREAMING — RESOLVED.** Plugin harnesses stream incrementally via
`params.onPartialReply` (assistant text) and `params.onAgentEvent` (native
plan/reasoning/tool events) — the SDK guide names "a local CLI or daemon that must
stream native plan/reasoning/tool events" as a first-class harness use case, i.e.
exactly ours. Map ACP `session_update` chunks → those callbacks. No buffering-only
fallback needed.

**Required integration points (curl-verified, MUST do):**
- **`runAgentEndSideEffects({event, ctx})`** from `openclaw/plugin-sdk/agent-harness-runtime`
  after finalizing each attempt — dispatches the portable `agent_end` hook + research
  capture. (Use `awaitAgentEndSideEffects(...)` for non-interactive runs.) Skipping
  this silently drops host-side end-of-turn behavior.
- **Version handshake.** Codex blocks app-servers below a tested floor at the
  initialize handshake; our adapter should likewise check the ACP `PROTOCOL_VERSION`
  of the spawned `harness-acp` and refuse a mismatch, rather than fail mid-turn.
- **Lazy imports + shared client (Codex pattern).** Codex keeps app-server runtime
  code behind lazy `import()` so plugin *discovery* stays cheap, and holds one
  shared client across attempts. Mirror both: lazy-load the spawn/ACP code, one
  subprocess reused across attempts.

---

## 6. Teardown — no watchdog needed; the ACP SDK already exits on stdin EOF

`dispose?()` is real (types.ts:131) but not crash-safe (#18420: the gateway orphans
children on signal). The earlier plan was a custom stdin-EOF watchdog. **Probed it —
it's YAGNI (rung 4: the SDK does it).** `acp.run_agent` → `Connection._receive_loop`
does `line = await reader.readline(); if not line: break` → on stdin EOF the loop
ends, `_disconnect()` runs, `run_agent` returns, the process exits. Verified by
spawning the real entry: serves while stdin is open, exits **code 0 in 0.11s** when
stdin closes (parent death by any cause closes the pipe). Regression-guarded by
`test_main_entry_serves_then_exits_on_stdin_eof` (the control assertion — "alive with
stdin open" — is what catches regressions; clean exit alone does not).

So teardown needs **nothing new on the Python side**. `dispose()` on the TS side just
closes the subprocess's stdin (or lets it close on plugin unload); the OS closes it on
crash. Same trigger either way.

> **Bug found while probing (fixed):** `server.py:main()` called `acp.run_agent(agent)`
> without `await`/`asyncio.run` — the coroutine was created and discarded, so the entry
> the whole adapter spawns **never served** (fell through, exit 0). Phase-2 tests drive
> `HarnessAgent` methods directly and never exercised `main()`. Fixed to
> `asyncio.run(acp.run_agent(agent))`; the new subprocess test guards it. (Lesson: a
> console entry needs one runnable check — unit-testing the methods doesn't cover it.)

**Known ceiling (`ponytail:`) — now recoverable (Bucket B).** The probe covers the idle
case (parent dies between turns). If the parent dies *mid-turn*, a worker thread may still
be in `provider.stream` and the OS reaps it at exit; **the persisted state is made
consistent on the next resume** — a tool_use orphaned by the crash is reconciled with a
synthetic result by `core/loop.py::reconcile_dangling_tool_calls` (called in `run()`, the
CLI resume path, and the ACP `prompt`/`session/load` path). So mid-turn death is no longer
just "acceptable" — it's recovered. Persistence is also fail-closed (persist-before-append/
execute). See the core CLAUDE.md Gotchas 18–19.

**Known ceiling — single session per client (descoped-D: parallel adapters).** `acp-client.ts`
holds ONE `session` slot; `createHarnessAcp` reuses one client across `runAttempt` calls.
Sequential reuse (finish S1, open S2) is fine. CONCURRENT attempts on the same harness
instance are NOT supported and we do not claim they are — a second `newSession()` would
overwrite the slot. To keep that from *silently* crossing sessions, `prompt(sessionId,…)`
now asserts `sessionId === session.sessionId` and **fails loud** on a mismatch (it no longer
ignores the argument). The real fix when concurrency is in scope is a `Map<sessionId,
ActiveSession>` + a two-attempt fake probe; until then the guard converts a silent
correctness bug into an explicit error. (The Python `harness-acp` server itself is already
multi-session — `_sessions` keyed by id — so this ceiling is the TS adapter's, not the agent's.)

---

## 7. Packaging (ClawHub) — Phase 4

All facts below are **curl-free, pinned from the installed `openclaw@2026.6.10`
docs/SDK** (clean local channel: `node_modules/openclaw/docs/**`, `dist/plugin-sdk/**`).

### Verified contract
- **`openclaw.plugin.json` is REQUIRED at the plugin root** (`docs/plugins/manifest.md:28`):
  OpenClaw reads it to validate config **without executing plugin code**. We added it —
  `id: "harness-acp"`, `configSchema` (pythonPath, allowGatedTools), and
  `activation: { onStartup: false, onAgentHarnesses: ["harness-acp"] }` (we are inert at
  startup and load only when our harness runtime is requested). We deliberately do **not**
  declare `providers: ["anthropic"]` — that field claims *ownership* of the provider id,
  which is OpenClaw's, not ours; `supports()` does the runtime backing check instead.
- **Publish** (`docs/clawhub/publishing.md`): `clawhub package publish @owner/pkg`
  (`--dry-run` first). npm-style **scoped name; the scope must match the publish owner**.
  The server validates owner perms, name, version, **file limits**, and source metadata,
  then runs **automated security checks**; releases stay **hidden until review/verification**.
  Requires a public GitHub repo + setup docs (`docs/plugins/community.md`).
- **Install**: `openclaw plugins install clawhub:<pkg>` (or `npm:<pkg>` during cutover).
- **Codex is our exact precedent** (external agent runtime, spawned `transport:"stdio"`):
  OpenClaw ships a **"managed binary" as the default** and exposes `appServer.command` +
  `OPENCLAW_CODEX_APP_SERVER_BIN` to **override to a local executable**
  (`docs/plugins/codex-harness-reference.md:52-62,519`). Dual model: a default + an override.
- **SDK helper for our exact Windows spawn problem**: `openclaw/plugin-sdk/windows-spawn`
  (`resolveWindowsSpawnProgram` / `materializeWindowsSpawnProgram`). We hand-rolled
  `.venv/Scripts/python.exe`; rung 4 — use the helper (deliverable #3).

### Decision — locate, don't bundle (fail-closed); Codex-style override
A Python **venv** is an interpreter + site-packages: large and **platform×arch specific**
(win-x64, mac-arm64/x64, linux-x64…), which collides with ClawHub **file limits** and
balloons the package. So we ship the plugin **lean** (JS + manifest only) and the adapter
**resolves** a Python in priority order — (1) `config.pythonPath`, (2) `OPENCLAW_HARNESS_PYTHON`
env, (3) a managed venv if one was created, (4) ambient `harness` on PATH — **failing closed
with a clear setup message** if none works. `pip install <harness-dist>` is a documented
prerequisite, exactly like Codex needing its CLI. Rationale: smallest, most **auditable**
attack surface (no shipped prebuilt binaries, no silent first-run `pip` network install) —
which is the project's whole pitch. Trade-off: one explicit user setup step. A managed-venv
auto-bootstrap is a later opt-in, **not** MVP.

### Correction (stale claim removed)
The prior §7 said community installs "trip the dangerous-code scanner on `child_process` →
require `--dangerously-force-unsafe-install`." **That text does NOT appear in `openclaw@2026.6.10`
docs.** The verified gate is "automated security checks + hidden until review" (publishing.md).
The exact scanner behavior toward `child_process`/spawn is **unverified for this version** and
must be re-pinned against the real scanner before publish (open item).

### Deliverables (ordered)
1. **`openclaw.plugin.json`** — **DONE** (id + configSchema + activation; JSON-valid).
2. **Python distribution** — **DONE (offline)**: dist renamed `claude-harness-acp` (import stays
   `harness`); `harness-acp`/`harness` entry points + `src/harness` wheel package confirmed via
   `tomllib`. The actual `python -m build` wheel needs the `build` module installed → deferred to
   publish (like the key); PyPI-name availability confirmed then too.
3. **Robust Python resolution** — **DONE**: `python-resolve.ts` (pure, unit-probed `#1–#4b`),
   `index.ts` now consumes `api.pluginConfig` (`pythonPath`/`allowGatedTools`) + `api.rootDir`,
   `acp-client.ts` spawns via the `windows-spawn` SDK helper and **fails closed** with a remediation
   message. Closes the declared-but-env-only gap *and* the stale "vendored venv" comment.
4. **Package the TS** — **DONE**: ship **compiled `dist/`** (building-plugins.md:100 — external
   plugins point at built JS), `openclaw.extensions: ["./dist/index.js"]`, `files` allowlist +
   `build`/`prepack` scripts. `npm pack --dry-run` = 7 files / 7.5kB (dist + manifest + README; no
   probes/sources/node_modules).
5. **Docs** — **DONE**: `README.md` (prereqs, install, config table, security posture, dev/probes).
6. **`clawhub package publish --dry-run`** — **DEFERRED** (needs the ClawHub CLI + owner auth; the
   real validation gate: file limits + scan). Last step before publish, with the real-API smoke (#1).

### $0 verification (no key, no publish, no spend) — all GREEN
- `python-resolve` probe `#1–#4b`; `npm pack --dry-run` shape; `tsc --noEmit` clean.
- Handshake probe + fake round-trip `#1–#7` still prove the wire end-to-end; 155 pytest green.
- Remaining external gates (deferred, by design): `clawhub --dry-run`, `python -m build` wheel,
   PyPI-name availability, the real-API smoke, and re-pinning the scanner's `child_process` stance.

---

## 8. Open items

**RESOLVED this revision** (curl-verified against `docs/plugins/sdk-agent-harness.md`
+ `extensions/codex/harness.ts`):
- ~~streaming to host UI~~ → `params.onPartialReply` / `params.onAgentEvent` (§5)
- ~~output delivery / `didSendViaMessagingTool`~~ → host-owned callbacks; result is a record (§3)
- ~~prompt input~~ → `params.prompt` (§3)
- ~~bundled registration pattern~~ → reference impl is `extensions/codex/harness.ts` (whole file)
- ~~`runtimePlan` drop~~ → corrected to USE (§3)

**PINNED via `tsc` against the installed `openclaw@2026.6.10` SDK** (harness.ts compiles clean):
- `params.prompt` is a **`string`**, not ACP content blocks (no block-parsing needed).
- `params.onPartialReply` takes **`{ delta?: string; replace?: true }`**, not a bare string.
- `EmbeddedRunReplayMetadata` requires `{ hadPotentialSideEffects, replaySafe }`.
- The full required set of `AgentHarnessAttemptResult` is satisfied by `buildResult()` in harness.ts.

**Still to pin — implementation-time:**
1. `params.onAgentEvent` payload shape — pin when wiring native plan/reasoning/tool events.
2. ACP **client** wire layer: use `@agentclientprotocol/sdk` (on npm, HTTP 200) — rung 5, don't
   hand-roll JSON-RPC. Needs a 2nd devDep install. Then a runtime probe (TS adapter ↔ real
   Python `harness-acp`) verifies wire correctness — tsc proves types, not param-name match.
3. `runAgentEndSideEffects(...)` call site (§5) — wire when the client carries the runtime ctx.
4. `contextEngineHostCapabilities` — RESOLVED: declare none (we don't host OpenClaw's context engine;
   declaring any would falsely claim its hooks). See the §8 remaining list for the verified rationale.
5. ClawHub packaging — **BUILT + verified offline** (§7 deliverables #1–#5; locate-not-bundle,
   fail-closed). Remaining: re-pin the scanner's `child_process` stance + `clawhub --dry-run` (publish-time).

**Build status — full adapter `tsc --noEmit` green against the real `openclaw@2026.6.10` +
`@agentclientprotocol/sdk@1.0.0` SDKs** (node usage genuinely checked — confirmed by a project-mode
probe that caught a deliberate `ChildProcess`→`number` error):
- `harness.ts` — AgentHarness factory (`supports`/`runAttempt`/`dispose`), result builder, `AcpClient` seam.
- `acp-client.ts` — spawn venv `python.exe` → `ndJsonStream` over stdio → `client()` connect →
  `initialize` with **`protocolVersion===PROTOCOL_VERSION(1)` asserted** (§5 handshake) →
  `buildSession().start()` → `ActiveSession.prompt()` + `nextUpdate()` loop mapping
  `agent_message_chunk` text → `onPartialReply`. `request_permission` → **deny-by-default** (§4;
  `allowGatedTools` opt-in). The ACP wire-shape usage was compiler-verified, not guessed.
- `index.ts` — `definePluginEntry` → `registerAgentHarness`.

**Fragility found:** node types currently resolve via `openclaw`'s *transitive* `@types/node`, not a direct
dep — fragile across hoisting/dep-tree changes. Add `@types/node` as a direct devDep (next install).

**Runtime interop — VERIFIED (no-key handshake probe, `probe.mjs`).** Ran the *compiled* `acp-client`
against the real Python `harness-acp`: `initialize` handshake completed, **`PROTOCOL_VERSION` negotiated
to 1** (TS `@agentclientprotocol/sdk@1.0.0` ↔ Python `agent-client-protocol==0.10.1` are the same
protocol generation — the version concern is empirically retired), **ndjson framing compatible** (no
hang under a 10s timeout), and `session/new` → `sess-1` (the subprocess serves; the `asyncio.run` fix
holds). Repro: `npx tsc -p tsconfig.build.json && node probe.mjs`. `@types/node` is now a direct devDep.

**Round-trip — VERIFIED deterministically, $0 (`probe-roundtrip-fake.mjs`).** The fake provider is the
*better* instrument for wire+translation than a live call: it forces the tool path and removes model
variance from a test that isn't about the model. Promote-the-fake: the scripted provider lives in
`src/harness/testing.py` (one source — the test suite re-exports it via `conftest`), and `server.py:main()`
**lazy-imports it gated by `HARNESS_PROVIDER=fake`** so the shipped entry never loads a fake in normal
operation. Ran the real adapter ↔ Python `harness-acp` (fake mode): **#1** streaming deltas → `onPartialReply`,
**#2** `runAgentEndSideEffects` fires on finalize (no throw — confirmed callable standalone), **#3** result
builder populated from real output, **#4** `echo` tool runs across two processes and the turn completes →
**tool_result invariant survived the ACP round-trip**, **#5** the per-session model rides in `session/new`'s
`_meta` and is accepted across the wire, **#6** an unadvertised model id is **rejected at session creation in
the subprocess** (trust floor — never forwarded to a provider), **#7** an `abortSignal` aborted mid-turn →
ACP `session/cancel` → `CancelToken` → **aborted** result (cancellation across the wire). 155 pytest green
(145 + 3 live-MCP, skipped where no server).

**Remaining (not architecture-blocking):**
1. **Real-API smoke test** (`probe-roundtrip.mjs`, deferred) — a *request-acceptance* gate, not "does Claude
   reply": the first time the real API validates our actual request shape (adaptive-thinking, `cache_control`,
   model id, the `(extra_body, top_level_kwargs)` split). One Haiku prompt. **Pre-Phase-4-publish gate — don't
   ship without it.** Needs `ANTHROPIC_API_KEY` in `../.env`.
2. **Cancellation across the wire — RESOLVED (verified $0, probe #7).** `runAttempt` listens to
   `params.abortSignal`; on abort it relays ACP `session/cancel` via `AcpClient.cancel()`, setting the
   subprocess `CancelToken`; the loop breaks into a `"cancelled"` stop → `aborted` result. (`tsc` caught the
   real field name — `abortSignal`, not `signal`.) Verified with a dedicated blocking provider
   (`testing.build_blocking_provider` + `HARNESS_PROVIDER=fake-cancel`): the abort fires mid-turn across two
   processes. The loop's `CancelToken` itself remains covered in-process (`test_cancel_sets_token_...`).
3. **Model propagation — RESOLVED (verified $0).** The adapter forwards OpenClaw's per-attempt
   `params.modelId` into the subprocess via **`session/new`'s `_meta`** — `newSession(model)` sends
   `{cwd, mcpServers, _meta:{modelId}}`; the ACP router spreads `_meta` keys into the `new_session` handler
   kwargs; the agent validates the id against its advertised `supported_models` (rejecting unknowns,
   `invalid_params`) and builds the provider per-session via a `make_provider` factory. Per-attempt because the
   adapter opens a fresh session per `runAttempt`. **NOT** spawn-time `HARNESS_MODEL` (the process is reused →
   every attempt would pin to the first model). **NOT** ACP `set_session_model`: a wire probe returned
   `method_not_found` — that method is registered `unstable=True` (gated behind `use_unstable_protocol`), and
   building a core mechanism on an explicitly-unstable protocol method is the durability risk we exist to avoid.
   `HARNESS_MODEL` remains only as the *default* when no `_meta` model is sent. Covered by
   `test_per_session_model_via_meta_...` (offline) + fake round-trip #5/#6.
4. **`contextEngineHostCapabilities` — RESOLVED: declare none (verified trust-correct).** We run our own
   context/compaction and do **not** host OpenClaw's context engine; declaring any capability
   (`compact`/`assemble-before-prompt`/…) would falsely claim to invoke that engine's hooks. Omitting makes
   OpenClaw **fail-closed** (with the engine's `unsupportedMessage`) when a context engine is paired with us,
   rather than silently corrupt its state (`docs/concepts/context-engine.md` §Host requirements). The bundled
   Codex harness declares capabilities because it projects context into Codex; we don't.
5. **ClawHub packaging** (§7, Phase 4) — **BUILT + verified offline**: manifest, Python resolution
   (`windows-spawn` + fail-closed), compiled `dist/` + `files` allowlist (`npm pack` = 7 files / 7.5kB),
   README. Locate-not-bundle (no shipped venv). Last items before publish: `clawhub --dry-run`, the
   `python -m build` wheel + PyPI-name check, and scanner re-pin (with the real-API smoke gate, #1).

The architecture is now **buildable**: the seam, delivery, streaming, approval,
teardown, and the required side-effect/provider-pairing integration points are all
pinned from raw source. The remaining items are signatures and packaging, safely
pinned at coding time.
