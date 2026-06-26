"""edit — exact string-replace in a file (the Claude-native edit format).

`old_string` must match exactly and uniquely, unless `replace_all` is set. This
is the format Claude models are best at; V4A patch mode (for GPT/Codex) is a
later addition steered by coding context, not built here.
"""

from __future__ import annotations

from pathlib import Path

from harness.core.types import Tool
from harness.tools._util import err, text_result


def edit_handler(
    cancel=None,
    path: str = "",
    old_string: str = "",
    new_string: str = "",
    replace_all: bool = False,
) -> "object":
    if not path:
        return err("edit: 'path' is required")
    if old_string == new_string:
        return err("edit: old_string and new_string are identical")
    p = Path(path)
    if not p.exists():
        return err(f"edit: no such file: {path}")
    try:
        original = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return err(f"edit: {type(e).__name__}: {e}")

    count = original.count(old_string)
    if count == 0:
        return err("edit: old_string not found in file")
    if count > 1 and not replace_all:
        return err(
            f"edit: old_string is not unique ({count} matches). Add more context "
            "to make it unique, or set replace_all=true."
        )

    updated = original.replace(old_string, new_string)
    try:
        p.write_text(updated, encoding="utf-8")
    except OSError as e:
        return err(f"edit: write failed: {type(e).__name__}: {e}")

    n = count if replace_all else 1
    return text_result(f"edit: replaced {n} occurrence(s) in {path}")


edit = Tool(
    name="edit",
    description="Replace an exact string in a file. old_string must match exactly "
    "and be unique unless replace_all=true.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace every match"},
        },
        "required": ["path", "old_string", "new_string"],
    },
    handler=edit_handler,
    parallel_safe=False,  # writes — don't run concurrently with other edits
    execution_mode="sequential",
    requires_approval=True,
    tags=("edit",),
)
