/**
 * No-key handshake probe: runs the REAL compiled acp-client against the real Python
 * `harness-acp`, exercising the actual wire (ndjson framing + JSON-RPC + ACP). Verifies
 * what tsc can't: that two independently-versioned SDKs speak the same protocol, that
 * the subprocess serves, and that PROTOCOL_VERSION negotiates to 1. No prompt → no API key.
 *
 * Build first:  npx tsc -p tsconfig.build.json
 * Run:          node probe.mjs
 * An assertion failure here is the probe SUCCEEDING — it caught a real wire mismatch.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createAcpClient } from "./dist/acp-client.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const harnessDir = path.resolve(here, "..", ".."); // harness/
const srcDir = path.join(harnessDir, "src");
const repoRoot = path.resolve(harnessDir, ".."); // Assignment/
const venvPy = path.join(repoRoot, ".venv", "Scripts", "python.exe");
const pythonPath = fs.existsSync(venvPy) ? venvPy : "python";

const withTimeout = (p, ms, label) =>
  Promise.race([
    p,
    new Promise((_, rej) =>
      setTimeout(() => rej(new Error(`TIMEOUT ${ms}ms at ${label} — framing/hang?`)), ms),
    ),
  ]);

const client = createAcpClient({ pythonPath, cwd: harnessDir, env: { PYTHONPATH: srcDir } });

try {
  console.log("python:", pythonPath);
  await withTimeout(client.ensureStarted(), 10000, "initialize");
  console.log("OK  initialize: handshake complete, PROTOCOL_VERSION assertion held (==1)");
  const sid = await withTimeout(client.newSession(), 10000, "session/new");
  console.log("OK  session/new:", sid);
  await client.close();
  console.log("PROBE PASS — TS adapter ↔ Python harness-acp interop verified (no key, no prompt).");
  process.exit(0);
} catch (e) {
  console.error("PROBE FINDING:", e?.message ?? e);
  try {
    await client.close();
  } catch {}
  process.exit(1);
}
