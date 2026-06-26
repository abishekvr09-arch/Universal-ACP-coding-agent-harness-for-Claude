/**
 * ACP client: spawns the Python `harness-acp` subprocess and drives it over ACP.
 *
 * Uses the official `@agentclientprotocol/sdk` (rung 5 — not hand-rolled JSON-RPC).
 * The wire layer the SDK encodes is what makes this trustworthy; we add only the
 * spawn, the safe-by-default permission policy (§4), and the streaming→callback map.
 */
import { spawn, type ChildProcess } from "node:child_process";
import { Readable, Writable } from "node:stream";

import {
  client,
  ndJsonStream,
  PROTOCOL_VERSION,
  type ActiveSession,
  type ClientContext,
} from "@agentclientprotocol/sdk";
import {
  materializeWindowsSpawnProgram,
  resolveWindowsSpawnProgram,
} from "openclaw/plugin-sdk/windows-spawn";

import type { AcpClient, AcpPromptHandlers, AcpPromptOutcome } from "./harness.js";

export interface AcpClientConfig {
  /** the resolved Python command/path to spawn (see python-resolve.ts, §7 locate-not-bundle).
   *  Spawned via the windows-spawn SDK helper so .cmd/.bat/exe shims resolve safely (#71139). */
  pythonPath: string;
  /** absolute cwd for the session + the spawned process */
  cwd: string;
  serverArgs?: readonly string[];
  env?: Record<string, string>;
  /** §4: false (default) denies gated (requires_approval) tools — safe by default */
  allowGatedTools?: boolean;
}

const DEFAULT_SERVER_ARGS = ["-m", "harness.acp.server"] as const;

export function createAcpClient(cfg: AcpClientConfig): AcpClient {
  let proc: ChildProcess | undefined;
  let ctx: ClientContext | undefined;
  let session: ActiveSession | undefined;
  let connClose: ((error?: unknown) => void) | undefined;

  async function ensureStarted(): Promise<void> {
    if (ctx) return;
    const childEnv = { ...process.env, ...cfg.env };
    const argv = [...(cfg.serverArgs ?? DEFAULT_SERVER_ARGS)];
    // Resolve the spawn program through the SDK helper so a .cmd/.bat/exe Python shim
    // (pyenv-win, Store python, etc.) launches without the #71139 shell-wrapper class.
    const program = resolveWindowsSpawnProgram({ command: cfg.pythonPath, env: childEnv });
    const inv = materializeWindowsSpawnProgram(program, argv);
    const child = spawn(inv.command, inv.argv, {
      cwd: cfg.cwd,
      env: childEnv,
      stdio: ["pipe", "pipe", "pipe"],
      shell: inv.shell,
      windowsHide: inv.windowsHide,
    });
    proc = child;
    // Capture a spawn failure (ENOENT for a missing/misconfigured Python) so we can
    // fail closed with a remediation message instead of an opaque stream error.
    let spawnError: Error | undefined;
    child.on("error", (e: Error) => {
      spawnError = e;
    });
    if (!child.stdin || !child.stdout) throw new Error("harness-acp: missing stdio pipes");

    const stream = ndJsonStream(
      Writable.toWeb(child.stdin) as WritableStream<Uint8Array>,
      Readable.toWeb(child.stdout) as ReadableStream<Uint8Array>,
    );

    const app = client({ name: "harness-acp-adapter" });
    // §4 safe-by-default: the agent only asks for gated tools; deny unless opted in.
    app.onRequest("session/request_permission", ({ params }) => ({
      outcome: cfg.allowGatedTools
        ? { outcome: "selected", optionId: params.options[0]?.optionId ?? "allow" }
        : { outcome: "cancelled" },
    }));

    const connection = app.connect(stream);
    connClose = (e) => connection.close(e);
    ctx = connection.agent;

    // §5 version handshake — assert the negotiated PROTOCOL_VERSION, don't just connect.
    // Fail closed: if the Python harness can't start (bad path, not installed), surface
    // a clear remediation rather than an opaque ACP/stream error.
    let init: { protocolVersion: number };
    try {
      init = (await ctx.request("initialize", {
        protocolVersion: PROTOCOL_VERSION,
        clientCapabilities: {},
      })) as { protocolVersion: number };
    } catch (e) {
      const cause = spawnError?.message ?? (e as Error)?.message ?? String(e);
      throw new Error(
        `harness-acp: could not start the Python harness via "${cfg.pythonPath}". ` +
          `Install it (pip install claude-harness-acp) and set the plugin's pythonPath ` +
          `or the OPENCLAW_HARNESS_PYTHON env var. Cause: ${cause}`,
      );
    }
    if (init.protocolVersion !== PROTOCOL_VERSION) {
      throw new Error(
        `ACP protocol mismatch: agent=${init.protocolVersion} client=${PROTOCOL_VERSION}`,
      );
    }
  }

  async function newSession(model?: string): Promise<string> {
    if (!ctx) throw new Error("acp-client: not started");
    // Model rides in session/new's _meta (stable extension channel). We avoid
    // set_session_model: it's ACP-unstable (method_not_found unless the agent opts
    // into use_unstable_protocol). The agent validates the id and rejects unknowns,
    // which surfaces here as a rejected start() — a loud failure, never a silent swap.
    session = await ctx
      .buildSession({ cwd: cfg.cwd, mcpServers: [], ...(model ? { _meta: { modelId: model } } : {}) })
      .start();
    return session.sessionId;
  }

  async function prompt(
    sessionId: string,
    text: string,
    handlers: AcpPromptHandlers,
  ): Promise<AcpPromptOutcome> {
    if (!session) throw new Error("acp-client: no active session");
    // ONE session per client by construction (a single `session` slot). Concurrent
    // attempts on one client are NOT supported (descoped-D: parallel adapters). If a
    // second newSession() overwrote the slot, driving `session` for a DIFFERENT requested
    // id would silently cross sessions — so FAIL LOUD on a mismatch instead. Sequential
    // reuse (finish S1, open S2) is fine; only overlap trips this.
    if (session.sessionId !== sessionId) {
      throw new Error(
        `acp-client: session mismatch (requested ${sessionId}, active ${session.sessionId}); ` +
          "this client drives one session at a time — concurrent attempts are not supported",
      );
    }
    const done = session.prompt(text); // completion is also queued as a `stop` update
    let assistantText = "";
    let aborted = false;
    for (;;) {
      const msg = await session.nextUpdate();
      if (msg.kind === "stop") {
        aborted = msg.stopReason === "cancelled";
        break;
      }
      const update = msg.update;
      if (update.sessionUpdate === "agent_message_chunk" && update.content.type === "text") {
        assistantText += update.content.text;
        handlers.onText(update.content.text);
      } else if (update.sessionUpdate === "tool_call") {
        handlers.onToolCall?.(update.title);
      }
    }
    await done;
    return { assistantText, aborted };
  }

  async function cancel(sessionId: string): Promise<void> {
    // Relay ACP session/cancel (a notification) → sets the subprocess CancelToken,
    // which the loop sees cooperatively and breaks into a "cancelled" stop.
    if (ctx) await ctx.notify("session/cancel", { sessionId });
  }

  async function close(): Promise<void> {
    session?.dispose();
    session = undefined;
    connClose?.();
    proc?.stdin?.end(); // stdin EOF → harness-acp exits cleanly (§6)
    ctx = undefined;
  }

  return { ensureStarted, newSession, prompt, cancel, close };
}
