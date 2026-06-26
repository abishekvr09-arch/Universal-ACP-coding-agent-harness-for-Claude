/**
 * The OpenClaw AgentHarness — the thin registration + translation surface.
 *
 * Modeled on the verified `extensions/codex/harness.ts` shape: a factory returning
 * an `AgentHarness` object literal whose `runAttempt` delegates to a client that
 * owns an external agent process. Here the client speaks ACP to our Python
 * `harness-acp` subprocess (impl: ./acp-client, next increment).
 *
 * Type-checked against the installed `openclaw` SDK (`tsc --noEmit`). The `AcpClient`
 * interface is the seam — the same pattern as the Python side's `McpClient` Protocol.
 */
import { awaitAgentEndSideEffects } from "openclaw/plugin-sdk/agent-harness-runtime";
import type {
  AgentHarness,
  AgentHarnessAttemptParams,
  AgentHarnessAttemptResult,
  AgentHarnessSupport,
  AgentHarnessSupportContext,
} from "openclaw/plugin-sdk/agent-harness-runtime";

// --- the seam: what runAttempt needs from the ACP subprocess ----------------

export interface AcpPromptHandlers {
  /** assistant text deltas → params.onPartialReply */
  onText(text: string): void;
  /** observed tool calls — for verification now, params.onAgentEvent mapping later */
  onToolCall?(title: string): void;
}

export interface AcpPromptOutcome {
  assistantText: string;
  aborted: boolean;
}

/** Minimal ACP surface the harness depends on. Real impl spawns the venv python
 *  and speaks ACP over stdio (./acp-client); a fake can satisfy this for tests. */
export interface AcpClient {
  ensureStarted(): Promise<void>;
  /** opens an ACP session; the client owns its own cwd (it spawns the subprocess).
   *  `model`, when given, rides in session/new's _meta (the stable ACP extension
   *  channel — NOT the unstable set_session_model method); the agent validates it. */
  newSession(model?: string): Promise<string>;
  prompt(sessionId: string, text: string, handlers: AcpPromptHandlers): Promise<AcpPromptOutcome>;
  /** relay an ACP session/cancel for an in-flight turn (OpenClaw aborted the attempt) */
  cancel(sessionId: string): Promise<void>;
  close(): Promise<void>;
}

export interface HarnessAcpOptions {
  /** our registered runtime id; supports() matches ctx.requestedRuntime against it */
  id?: string;
  /** providers our Python harness backs (it runs Claude) */
  providers?: readonly string[];
}

const DEFAULT_ID = "harness-acp";
const DEFAULT_PROVIDERS = ["anthropic"] as const;

/** Build a valid AgentHarnessAttemptResult from one completed ACP turn.
 *  Delivery is host-owned (§3) — text already streamed via onPartialReply — so the
 *  messaging-send fields stay empty; this result is a record/classification. */
function buildResult(sessionId: string, out: AcpPromptOutcome): AgentHarnessAttemptResult {
  const assistantTexts = out.assistantText ? [out.assistantText] : [];
  return {
    aborted: out.aborted,
    externalAbort: out.aborted,
    timedOut: false,
    idleTimedOut: false,
    timedOutDuringCompaction: false,
    promptError: null,
    promptErrorSource: null,
    sessionIdUsed: sessionId,
    assistantTexts,
    messagesSnapshot: [],
    toolMetas: [],
    lastAssistant: undefined,
    didSendViaMessagingTool: false,
    messagingToolSentTexts: [],
    messagingToolSentMediaUrls: [],
    messagingToolSentTargets: [],
    cloudCodeAssistFormatError: false,
    replayMetadata: { hadPotentialSideEffects: false, replaySafe: true },
    itemLifecycle: { startedCount: 0, completedCount: 0, activeCount: 0 },
  };
}

export function createHarnessAcp(client: AcpClient, opts: HarnessAcpOptions = {}): AgentHarness {
  const id = opts.id ?? DEFAULT_ID;
  const providers = new Set(opts.providers ?? DEFAULT_PROVIDERS);
  // contextEngineHostCapabilities intentionally omitted (verified trust-correct, not a gap):
  // we run our OWN context/compaction in the subprocess and do NOT host OpenClaw's context
  // engine. Declaring any capability ("compact"/"assemble-before-prompt"/…) would falsely
  // claim to invoke that engine's hooks; omitting makes OpenClaw fail-closed (with the
  // engine's unsupportedMessage) if one is paired with us, rather than corrupt its state.
  // (Codex declares capabilities because it projects context into Codex; we don't.)

  return {
    id,
    label: "Harness (Python, ACP)",

    supports(ctx: AgentHarnessSupportContext): AgentHarnessSupport {
      const ok = providers.has(ctx.provider) && ctx.requestedRuntime === id;
      return ok
        ? { supported: true, priority: 100 }
        : { supported: false, reason: `harness-acp does not back ${ctx.provider}/${ctx.requestedRuntime}` };
    },

    async runAttempt(params: AgentHarnessAttemptParams): Promise<AgentHarnessAttemptResult> {
      await client.ensureStarted();
      // Forward OpenClaw's per-attempt model into the subprocess via session/new `_meta`
      // (the stable ACP channel — NOT spawn-time HARNESS_MODEL, which would pin every
      // reused-process attempt to the first model; NOT the unstable set_session_model).
      const sessionId = await client.newSession(params.modelId);
      // Cancellation across the wire: OpenClaw aborts the attempt via params.abortSignal →
      // relay an ACP session/cancel, which sets the subprocess's CancelToken.
      const onAbort = () => {
        void client.cancel(sessionId);
      };
      params.abortSignal?.addEventListener("abort", onAbort);
      try {
        const out = await client.prompt(sessionId, params.prompt, {
          onText: (text) => params.onPartialReply?.({ delta: text }),
        });
        const result = buildResult(sessionId, out);
        // §5 MUST-do: agent-end side effects on finalize (portable agent_end hook +
        // research capture). awaitAgentEndSideEffects = the non-interactive variant.
        await awaitAgentEndSideEffects({
          event: { messages: result.messagesSnapshot, success: !out.aborted },
          ctx: { sessionId, modelId: params.modelId, modelProviderId: params.provider },
        });
        return result;
      } finally {
        params.abortSignal?.removeEventListener("abort", onAbort);
      }
    },

    async dispose(): Promise<void> {
      await client.close();
    },
  };
}
