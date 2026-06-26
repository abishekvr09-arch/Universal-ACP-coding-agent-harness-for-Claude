# openclaw-harness-acp

An OpenClaw **AgentHarness** that runs the standalone [Python harness](../../) as an
**ACP** subprocess. OpenClaw drives the harness over the Agent Client Protocol; the
harness runs its own tool-calling loop, tools, approval gates, and compaction.

- **Plugin id:** `harness-acp`
- **Backs provider:** `anthropic` (Claude-first; the harness's provider contract is generic)
- **Transport:** the plugin spawns `python -m harness.acp.server` and speaks ACP over its stdio.

## Prerequisites

This plugin is **lean by design** — it does *not* bundle a Python runtime (that would be
large, platform-specific, and harder to audit). You provide Python:

1. **Node 22.19+** (OpenClaw requirement).
2. **Python 3.11+** on the machine running the OpenClaw gateway.
3. **The Python harness installed:**
   ```bash
   pip install claude-harness-acp
   ```
   (Distribution name `claude-harness-acp`; it installs the `harness` import package and
   the `harness-acp` entry point. A virtualenv is recommended.)

## Install

Once published:

```bash
openclaw plugins install clawhub:<owner>/openclaw-harness-acp
```

During the launch cutover, npm install also works: `openclaw plugins install npm:openclaw-harness-acp`.

## Configure

Config lives under `plugins.entries.harness-acp.config`:

| Key               | Type      | Default | What it does                                                                 |
| ----------------- | --------- | ------- | ---------------------------------------------------------------------------- |
| `pythonPath`      | `string`  | —       | Explicit Python to spawn. Overrides all other resolution.                    |
| `allowGatedTools` | `boolean` | `false` | Allow approval-gated tools without an interactive prompt (see Security).     |

**Python resolution order** (first match wins; "locate, don't bundle"):
1. `config.pythonPath`
2. `OPENCLAW_HARNESS_PYTHON` env var
3. a managed venv under the plugin root (`.venv/…`), if present
4. `python` / `python3` on `PATH`

(1) and (2) are explicit, so a broken value **fails closed loudly at spawn** with a
remediation message — it never silently falls back to a different Python.

## Security posture

- **Approval-gated tools are denied by default** in this headless embedded context.
  Set `allowGatedTools: true` only if you accept the harness running `requires_approval`
  tools without an interactive prompt.
- **Fail-closed startup:** if the Python harness can't start (missing/misconfigured),
  the attempt fails with a clear message rather than degrading silently.
- The harness preserves the **tool_result invariant**, persist-before-execute, and its
  own approval/cancellation seams inside the subprocess.

## How it works

`runAttempt` opens an ACP session (the per-attempt model rides in `session/new`'s `_meta`,
the stable channel — not the unstable `set_session_model`), streams assistant text to
OpenClaw via `params.onPartialReply`, relays `params.abortSignal` → ACP `session/cancel`,
and calls `awaitAgentEndSideEffects` on finalize. The subprocess exits on stdin EOF (no
watchdog needed). Full design + provenance: [`DESIGN.md`](./DESIGN.md).

## Develop

```bash
npm install
npm run typecheck          # tsc --noEmit vs the real openclaw + ACP SDKs
npm run build              # tsc -p tsconfig.build.json → dist/
node probe.mjs             # no-key ACP handshake (spawns the real Python harness-acp)
node probe-python-resolve.mjs   # unit-checks Python resolution ($0)
node probe-roundtrip-fake.mjs   # full wire round-trip #1–#7, $0 (HARNESS_PROVIDER=fake)
node probe-roundtrip.mjs        # real-API request-acceptance smoke (needs ANTHROPIC_API_KEY; skips otherwise)
```

The published package ships compiled `dist/` (OpenClaw loads built JS for external plugins)
plus `openclaw.plugin.json`; probes and sources stay out of the tarball.
