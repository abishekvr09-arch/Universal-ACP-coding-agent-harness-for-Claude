"""bash — run a shell command.

execute-tagged (budget refund), sequential + not parallel_safe (commands can
have side effects), and requires_approval by default (it's the dangerous one).
Subprocess decoded as utf-8 with errors="replace" — the cp1252 crash on non-ASCII
output is a known footgun.
"""

from __future__ import annotations

import subprocess

from harness.core.types import Tool
from harness.tools._util import err, text_result

DEFAULT_TIMEOUT_MS = 120_000
MAX_OUTPUT = 30_000  # characters, then truncate


def bash_handler(cancel=None, command: str = "", timeout: int = DEFAULT_TIMEOUT_MS) -> "object":
    if not command.strip():
        return err("bash: 'command' is required")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",  # never crash on non-ASCII stderr/stdout
            timeout=max(timeout, 1) / 1000.0,
        )
    except subprocess.TimeoutExpired:
        return err(f"bash: command timed out after {timeout}ms")
    except OSError as e:
        return err(f"bash: {type(e).__name__}: {e}")

    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n... [truncated at {MAX_OUTPUT} chars]"
    body = out if out.strip() else "(no output)"
    text = f"exit code: {proc.returncode}\n{body}"
    return text_result(text, is_error=proc.returncode != 0)


bash = Tool(
    name="bash",
    description="Execute a shell command and return stdout+stderr and the exit code.",
    input_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "Timeout in ms (max 600000)"},
        },
        "required": ["command"],
    },
    handler=bash_handler,
    parallel_safe=False,
    execution_mode="sequential",
    requires_approval=True,
    tags=("execute",),
)
