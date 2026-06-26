/**
 * Deterministic round-trip probe — $0, no key, no model variance.
 *
 * Drives the REAL adapter ↔ Python `harness-acp` (HARNESS_PROVIDER=fake), where a
 * scripted provider forces: stream text → call `echo` tool → stream `pong`. This is
 * the BETTER instrument for verifying wire+translation than a live call: it forces
 * the tool path and removes the model from a test that isn't about the model.
 *
 *  #1 streaming: scripted text deltas reach onPartialReply
 *  #2 finalize: runAttempt completes (runAgentEndSideEffects fires, no throw)
 *  #3 result builder populated from real output ("pong")
 *  #4 tool_result invariant survives the ACP round-trip (echo runs across two
 *     processes and the turn still completes — a broken invariant 400s the loop)
 *  #5 model propagation: the _meta model on session/new reaches + is accepted by the subprocess
 *  #6 trust floor: an unsupported model id is rejected at session creation, not forwarded
 *
 * Build:  npx tsc -p tsconfig.build.json
 * Run:    node probe-roundtrip-fake.mjs   (no key needed)
 */
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createAcpClient } from "./dist/acp-client.js";
import { createHarnessAcp } from "./dist/harness.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const harnessDir = path.resolve(here, "..", "..");
const srcDir = path.join(harnessDir, "src");
const repoRoot = path.resolve(harnessDir, "..");
const venvPy = path.join(repoRoot, ".venv", "Scripts", "python.exe");
import fs from "node:fs";
const pythonPath = fs.existsSync(venvPy) ? venvPy : "python";

const withTimeout = (p, ms, label) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error(`TIMEOUT ${ms}ms at ${label}`)), ms)),
  ]);

const env = { PYTHONPATH: srcDir, HARNESS_PROVIDER: "fake" };
const client = createAcpClient({ pythonPath, cwd: harnessDir, env });
const harness = createHarnessAcp(client);

let failures = 0;
const check = (ok, label, detail = "") => {
  console.log(`${ok ? "OK " : "XX "} ${label}${detail ? " — " + detail : ""}`);
  if (!ok) failures++;
};

try {
  // --- Call A: via harness.runAttempt → #1 streaming, #2 finalize, #3 result ---
  let deltas = 0;
  const result = await withTimeout(
    harness.runAttempt({
      prompt: "go",
      modelId: "fake",
      provider: "anthropic",
      onPartialReply: (p) => {
        if (p && typeof p.delta === "string") deltas++;
      },
    }),
    15000,
    "runAttempt",
  );
  const text = (result.assistantTexts || []).join("").trim();
  check(deltas > 0, "#1 streaming: deltas → onPartialReply", `${deltas} deltas`);
  // accumulates ALL assistant text across the multi-turn turn ("working " + "pong")
  check(text.includes("pong"), "#3 result builder populated from real output", `answer="${text}"`);
  check(
    result.aborted === false && result.promptError == null,
    "#2 finalize ran (runAgentEndSideEffects, no throw)",
  );

  // --- Call B: tool-using turn via client.prompt → #4 invariant across the wire ---
  // newSession("fake") forwards the model in session/new's _meta; it only resolves
  // because the subprocess accepted the advertised id → #5.
  let toolCalls = 0;
  let sid;
  let modelAccepted = true;
  try {
    sid = await withTimeout(client.newSession("fake"), 10000, "session/new");
  } catch {
    modelAccepted = false;
    sid = await withTimeout(client.newSession(), 10000, "session/new(fallback)");
  }
  check(modelAccepted, "#5 model propagation: _meta model accepted across the wire");
  const out = await withTimeout(
    client.prompt(sid, "go", { onText: () => {}, onToolCall: () => toolCalls++ }),
    15000,
    "tool round-trip",
  );
  check(
    toolCalls > 0 && out.aborted === false,
    "#4 tool_result invariant survived ACP round-trip",
    `${toolCalls} tool call(s), turn completed`,
  );

  // #6 trust floor: an unadvertised model is REJECTED at session creation in the
  // subprocess, not silently forwarded to a provider. (#5 is implicit above: Call B's
  // newSession("fake") only resolved because the _meta model was accepted.)
  let rejected = false;
  try {
    await withTimeout(client.newSession("gpt-4"), 10000, "newSession(gpt-4)");
  } catch {
    rejected = true;
  }
  check(rejected, "#6 trust floor: unsupported model rejected at session creation");
  await client.close();

  // --- Call C: cancellation across the wire (#7) — a separate fake-cancel subprocess
  // whose provider blocks until a session/cancel arrives. We abort the attempt's
  // abortSignal mid-turn; runAttempt relays ACP session/cancel; the loop breaks into a
  // "cancelled" stop → result.aborted. Verifies the adapter's abort→cancel hop across
  // two processes (the loop's CancelToken itself is covered in-process by pytest).
  const cancelClient = createAcpClient({
    pythonPath,
    cwd: harnessDir,
    env: { PYTHONPATH: srcDir, HARNESS_PROVIDER: "fake-cancel" },
  });
  const cancelHarness = createHarnessAcp(cancelClient);
  const ac = new AbortController();
  const runP = cancelHarness.runAttempt({
    prompt: "go",
    modelId: "fake",
    provider: "anthropic",
    onPartialReply: () => {},
    abortSignal: ac.signal,
  });
  await new Promise((r) => setTimeout(r, 3000)); // spawn + handshake + session + reach the blocking prompt
  ac.abort();
  const cancelled = await withTimeout(runP, 12000, "cancel runAttempt");
  check(
    cancelled.aborted === true,
    "#7 cancellation across the wire: abortSignal → session/cancel → aborted",
  );
  await cancelClient.close();

  console.log(failures === 0 ? "\nFAKE ROUND-TRIP PASS — #1–#7 green, $0." : `\n${failures} assertion(s) failed.`);
  process.exit(failures === 0 ? 0 : 1);
} catch (e) {
  console.error("ROUND-TRIP FINDING:", e?.message ?? e);
  try {
    await client.close();
  } catch {}
  process.exit(1);
}
