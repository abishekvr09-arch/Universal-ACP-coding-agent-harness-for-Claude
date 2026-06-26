/**
 * Unit probe for resolvePythonPath (§7 locate-not-bundle). Pure function, no spawn, $0.
 * Build:  npx tsc -p tsconfig.build.json
 * Run:    node probe-python-resolve.mjs
 */
import { resolvePythonPath } from "./dist/python-resolve.js";

let failures = 0;
const check = (ok, label, detail = "") => {
  console.log(`${ok ? "OK " : "XX "} ${label}${detail ? " — " + detail : ""}`);
  if (!ok) failures++;
};

const noExist = () => false;
const allExist = () => true;

// 1) explicit config wins, honored even if missing (fail loud at spawn, never silent fallback)
check(
  resolvePythonPath({ configPath: "/x/py", env: {}, platform: "linux", exists: noExist }) === "/x/py",
  "#1 config.pythonPath wins (honored even if absent)",
);

// 2) env override when no config
check(
  resolvePythonPath({ env: { OPENCLAW_HARNESS_PYTHON: "/e/py" }, platform: "linux", exists: noExist }) === "/e/py",
  "#2 OPENCLAW_HARNESS_PYTHON used when no config",
);

// 3) managed venv used only if present (win + posix layouts)
{
  const winVenv = resolvePythonPath({ rootDir: "/root", env: {}, platform: "win32", exists: allExist });
  check(
    winVenv.includes(".venv") && winVenv.endsWith("python.exe"),
    "#3a managed venv (win) used when present",
    winVenv,
  );
}
check(
  resolvePythonPath({ rootDir: "/root", env: {}, platform: "linux", exists: allExist }).endsWith("/.venv/bin/python3"),
  "#3b managed venv (posix) used when present",
);
check(
  resolvePythonPath({ rootDir: "/root", env: {}, platform: "linux", exists: noExist }) === "python3",
  "#3c managed venv skipped when absent → PATH fallback",
);

// 4) platform default on PATH (no config/env/venv)
check(resolvePythonPath({ env: {}, platform: "win32", exists: noExist }) === "python", "#4a win PATH fallback = python");
check(resolvePythonPath({ env: {}, platform: "linux", exists: noExist }) === "python3", "#4b posix PATH fallback = python3");

console.log(failures === 0 ? "\nPYTHON-RESOLVE PASS — all green, $0." : `\n${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
