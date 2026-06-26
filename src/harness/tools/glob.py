"""glob — file pattern matching, results sorted by modification time (newest first)."""

from __future__ import annotations

from pathlib import Path

from harness.core.types import Tool
from harness.tools._util import err, text_result

MAX_RESULTS = 1000


def glob_handler(cancel=None, pattern: str = "", path: str = ".") -> "object":
    if not pattern:
        return err("glob: 'pattern' is required")
    base = Path(path or ".")
    if not base.exists():
        return err(f"glob: no such directory: {path}")
    try:
        matches = [p for p in base.glob(pattern) if p.is_file()]
    except (ValueError, OSError) as e:
        return err(f"glob: {type(e).__name__}: {e}")

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return text_result(f"(no files match {pattern!r} under {base})")
    shown = matches[:MAX_RESULTS]
    listing = "\n".join(str(p) for p in shown)
    if len(matches) > MAX_RESULTS:
        listing += f"\n... [{len(matches) - MAX_RESULTS} more]"
    return text_result(listing)


glob = Tool(
    name="glob",
    description="Find files matching a glob pattern (e.g. '**/*.py'), newest first.",
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, supports **"},
            "path": {"type": "string", "description": "Directory to search (default cwd)"},
        },
        "required": ["pattern"],
    },
    handler=glob_handler,
    parallel_safe=True,
    tags=("search",),
)
