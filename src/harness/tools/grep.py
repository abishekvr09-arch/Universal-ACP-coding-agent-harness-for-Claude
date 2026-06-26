"""grep — regex content search.

Uses ripgrep (`rg`) when available (fast, respects .gitignore); falls back to a
pure-Python walk so it works on a bare Windows box with no rg installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from harness.core.types import Tool
from harness.tools._util import err, text_result

MAX_MATCHES = 250
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def _rg(pattern: str, path: str, glob: str | None, ignore_case: bool) -> "object":
    cmd = ["rg", "--line-number", "--no-heading", "--color", "never"]
    if ignore_case:
        cmd.append("--ignore-case")
    if glob:
        cmd += ["--glob", glob]
    cmd += ["--", pattern, path or "."]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return err(f"grep(rg): {type(e).__name__}: {e}")
    if proc.returncode not in (0, 1):  # 1 = no matches, not an error
        return err(f"grep(rg): {proc.stderr.strip()}")
    lines = proc.stdout.splitlines()
    return _format(lines)


def _python_grep(
    pattern: str, path: str, glob: str | None, ignore_case: bool
) -> "object":
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return err(f"grep: bad regex: {e}")
    base = Path(path or ".")
    if base.is_file():
        files = [base]
    else:
        files = [
            p
            for p in base.rglob(glob or "*")
            if p.is_file() and not any(part in _SKIP_DIRS for part in p.parts)
        ]
    out: list[str] = []
    for f in files:
        try:
            for i, line in enumerate(
                f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if rx.search(line):
                    out.append(f"{f}:{i}:{line}")
                    if len(out) >= MAX_MATCHES:
                        return _format(out)
        except OSError:
            continue
    return _format(out)


def _format(lines: list[str]) -> "object":
    if not lines:
        return text_result("(no matches)")
    shown = lines[:MAX_MATCHES]
    body = "\n".join(shown)
    if len(lines) > MAX_MATCHES:
        body += f"\n... [{len(lines) - MAX_MATCHES} more matches]"
    return text_result(body)


def grep_handler(
    cancel=None,
    pattern: str = "",
    path: str = ".",
    glob: str | None = None,
    ignore_case: bool = False,
) -> "object":
    if not pattern:
        return err("grep: 'pattern' is required")
    if shutil.which("rg"):
        return _rg(pattern, path, glob, ignore_case)
    return _python_grep(pattern, path, glob, ignore_case)


grep = Tool(
    name="grep",
    description="Search file contents by regex. Returns file:line:text matches. "
    "Uses ripgrep if installed, else a Python fallback.",
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression"},
            "path": {"type": "string", "description": "File or directory (default cwd)"},
            "glob": {"type": "string", "description": "Filter files, e.g. '*.py'"},
            "ignore_case": {"type": "boolean"},
        },
        "required": ["pattern"],
    },
    handler=grep_handler,
    parallel_safe=True,
    tags=("search",),
)
