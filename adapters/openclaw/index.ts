/**
 * Plugin entry: register the ACP-backed AgentHarness with OpenClaw.
 *
 * `definePluginEntry` → `api.registerAgentHarness(...)` is the whole surface
 * (verified thin, like extensions/codex/harness.ts). The harness spawns the Python
 * `harness-acp` over ACP; this file wires it in and reads plugin config.
 */
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

import { createAcpClient } from "./acp-client.js";
import { createHarnessAcp } from "./harness.js";
import { resolvePythonPath } from "./python-resolve.js";

interface HarnessPluginConfig {
  pythonPath?: string;
  allowGatedTools?: boolean;
}

export default definePluginEntry({
  id: "harness-acp",
  name: "Harness (ACP)",
  description: "Runs the standalone Python harness as an ACP subprocess via an OpenClaw AgentHarness.",
  register(api) {
    // Consume this plugin's config (manifest configSchema validated it). Locate-not-bundle:
    // resolve the Python from config → OPENCLAW_HARNESS_PYTHON → managed venv → PATH (§7).
    const cfg = (api.pluginConfig ?? {}) as HarnessPluginConfig;
    const pythonPath = resolvePythonPath({
      configPath: cfg.pythonPath,
      env: process.env,
      rootDir: api.rootDir,
    });
    const client = createAcpClient({
      pythonPath,
      cwd: process.cwd(),
      allowGatedTools: cfg.allowGatedTools ?? false, // §4 safe-by-default
    });
    api.registerAgentHarness(createHarnessAcp(client));
  },
});
