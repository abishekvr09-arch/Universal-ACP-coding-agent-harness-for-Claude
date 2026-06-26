/**
 * Resolve which Python runs `harness-acp` — the "locate, don't bundle" policy (§7).
 *
 * We do NOT ship a venv (large + platform×arch specific + ClawHub file limits). We
 * locate a Python in priority order and fail closed at spawn if none works. This is a
 * pure function (existence via an injectable seam) so it is unit-tested without spawning.
 */
import fs from "node:fs";
import path from "node:path";

export interface ResolvePythonParams {
  /** explicit override from OpenClaw plugin config: plugins.entries.harness-acp.config.pythonPath */
  configPath?: string;
  /** process env — read OPENCLAW_HARNESS_PYTHON */
  env?: NodeJS.ProcessEnv;
  /** plugin root (api.rootDir) — locate a managed venv if one was ever created there */
  rootDir?: string;
  /** platform override (tests) */
  platform?: NodeJS.Platform;
  /** existence-check seam (tests) */
  exists?: (p: string) => boolean;
}

/** The command/path to spawn for `harness-acp`, in priority order:
 *  1) config.pythonPath, 2) OPENCLAW_HARNESS_PYTHON — both EXPLICIT, so honored even if
 *     missing (a broken explicit setting must fail loudly at spawn, never silently fall back),
 *  3) a managed venv under the plugin root, used only if actually present,
 *  4) the platform default on PATH. Existence is probed for the bare PATH default at spawn
 *     time (fail-closed there), not here. */
export function resolvePythonPath(params: ResolvePythonParams = {}): string {
  const env = params.env ?? process.env;
  const platform = params.platform ?? process.platform;
  const exists = params.exists ?? ((p: string) => fs.existsSync(p));

  if (params.configPath) return params.configPath;
  if (env.OPENCLAW_HARNESS_PYTHON) return env.OPENCLAW_HARNESS_PYTHON;
  if (params.rootDir) {
    // Use the TARGET platform's path flavor (not the host's) so the produced venv path is
    // correct under a platform override; in production host === target, so this is a no-op.
    const join = platform === "win32" ? path.win32.join : path.posix.join;
    const venv =
      platform === "win32"
        ? join(params.rootDir, ".venv", "Scripts", "python.exe")
        : join(params.rootDir, ".venv", "bin", "python3");
    if (exists(venv)) return venv;
  }
  return platform === "win32" ? "python" : "python3";
}
