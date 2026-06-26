"""read — read a file (optionally a line range), returned with line numbers."""

from __future__ import annotations

from pathlib import Path

from harness.core.types import Tool
from harness.tools._util import err, text_result

DEFAULT_LIMIT = 2000


def read_handler(
    cancel=None, path: str = "", offset: int = 0, limit: int = DEFAULT_LIMIT
) -> "object":
    if not path:
        return err("read: 'path' is required")
    p = Path(path)
    if not p.exists():
        return err(f"read: no such file: {path}")
    if p.is_dir():
        return err(f"read: '{path}' is a directory")
    try:
        # utf-8 with replacement — never crash on non-ASCII / binary-ish bytes
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return err(f"read: {type(e).__name__}: {e}")

    start = max(offset, 0)
    window = lines[start : start + limit]
    if not window:
        return text_result(f"(file has {len(lines)} lines; offset {offset} is past the end)")
    numbered = "\n".join(f"{start + i + 1}\t{ln}" for i, ln in enumerate(window))
    return text_result(numbered)


read = Tool(
    name="read",
    description="Read a file from the local filesystem. Returns content with line "
    "numbers. Use offset/limit (in lines) for large files.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path"},
            "offset": {"type": "integer", "description": "0-based line to start from"},
            "limit": {"type": "integer", "description": f"Max lines (default {DEFAULT_LIMIT})"},
        },
        "required": ["path"],
    },
    handler=read_handler,
    parallel_safe=True,
    tags=("read",),
)
