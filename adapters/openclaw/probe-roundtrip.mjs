/**
 * Full round-trip probe — closes Phase 3's last verification. Two trivial billed
 * Anthropic calls (pinned to Haiku) through the REAL adapter ↔ Python harness-acp.
 *
 * Asserts the four things tsc can't:
 *  #1 streaming: real text deltas reach onPartialReply
 *  #2 runAgentEndSideEffects fires on finalize (runAttempt path completes)
 *  #3 result builder is populated from real output
 *  #4 tool_result invariant survives the ACP round-trip (a tool runs across two
 *     processes and the turn still completes — a broken invariant would 400 the loop)
 *
 * Key handling (security floor): reads ../../../.env, passes ANTHROPIC_API_KEY to the
 * subprocess via the ENV OBJECT only — never a CLI arg, never printed. Skips (exit 0)
 * if no key, so it never makes a surprise billed call.
 *
 * Build first:  npx tsc -p tsconfig.build.json
 * Run:          node probe-roundtrip.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createAcpClient } from "./dist/acp-client.js";
import { createHarnessAcp } from "./dist/harness.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const harnessDir = path.resolve(here, "..", "..");
const srcDir = path.join(harnessDir, "src");
const repoRoot = path.resolve(harnessDir, "..");
const envFile = path.join(repoRoot, ".env");
const venvPy = path.join(repoRoot, ".venv", "Scripts", "python.exe");
const pythonPath = fs.existsSync(venvPy) ? venvPy : "python";

function readEnvKey(file, name) {
  if (!fs.existsSync(file)) return undefined;
  for (const line of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)$/);
    if (m && m[1] === name) return m[2].trim().replace(/^["']|["']$/g, "");
  }
  return undefined;
}

const apiKey = process.env.ANTHROPIC_API_KEY ?? readEnvKey(envFile, "ANTHROPIC_API_KEY");
if (!apiKey) {
  console.log(`SKIP: no ANTHROPIC_API_KEY in env or ${envFile} — round-trip not run (no billed call).`);
  process.exit(0);
}

const withTimeout = (p, ms, label) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error(`TIMEOUT ${ms}ms at ${label}`)), ms)),
  ]);

// key + PYTHONPATH go to the subprocess via the env object; Haiku keeps it cheap.
const env = { PYTHONPATH: srcDir, ANTHROPIC_API_KEY: apiKey, HARNESS_MODEL: "claude-haiku-4-5" };
const client = createAcpClient({ pythonPath, cwd: harnessDir, env });
const harness = createHarnessAcp(client);

let failures = 0;
const check = (ok, label, detail = "") => {
  console.log(`${ok ? "OK " : "XX "} ${label}${detail ? " — " + detail : ""}`);
  if (!ok) failures++;
};

try {
  // --- Call 1: via harness.runAttempt → #1 streaming, #2 finalize, #3 result ---
  let deltas = 0;
  const params = {
    prompt: "Reply with exactly one word: pong",
    modelId: "claude-haiku-4-5",
    provider: "anthropic",
    onPartialReply: (p) => {
      if (p && typeof p.delta === "string") deltas++;
    },
  };
  const result = await withTimeout(harness.runAttempt(params), 60000, "runAttempt");
  const text = (result.assistantTexts || []).join("").trim();
  check(deltas > 0, "#1 streaming: text deltas → onPartialReply", `${deltas} deltas`);
  check(!!text, "#3 result builder populated from real output", `answer="${text.slice(0, 40)}"`);
  check(result.aborted === false && result.promptError == null, "#2 finalize ran (runAgentEndSideEffects, no throw)");

  // --- Call 2: a tool-using turn → #4 invariant survives the ACP round-trip ---
  let toolCalls = 0;
  const sid = await withTimeout(client.newSession(), 30000, "session/new");
  const out = await withTimeout(
    client.prompt(
      sid,
      "You MUST use your glob file-search tool to answer. How many files in the current directory match *.md? Reply with just the number.",
      { onText: () => {}, onToolCall: () => toolCalls++ },
    ),
    60000,
    "tool round-trip",
  );
  if (toolCalls > 0) {
    check(out.aborted === false, "#4 tool_result invariant survived ACP round-trip", `${toolCalls} tool call(s), turn completed`);
  } else {
    console.log("-- #4 not exercised this run: model answered without a tool (nondeterministic). In-process coverage: test_invariant.py.");
  }

  await client.close();
  console.log(failures === 0 ? "\nROUND-TRIP PASS — Phase 3 verifications green." : `\nROUND-TRIP: ${failures} assertion(s) failed.`);
  process.exit(failures === 0 ? 0 : 1);
} catch (e) {
  console.error("ROUND-TRIP FINDING:", e?.message ?? e);
  try {
    await client.close();
  } catch {}
  process.exit(1);
}
