"""Tool-call repair.

Models regularly emit tool calls as plain text instead of structured tool_use
blocks, and emit structurally-valid calls with wrongly-typed arguments
(string "42" for an integer field). Repair is first-class, not an afterthought:
the loop runs it on every call before approval hooks see them.

Two layers:
1. `promote_plaintext_tool_calls(response, allowed)` — RESPONSE level. When the
   model wrote tool calls as prose (three syntax families below), extract them
   into real ToolCall objects and flip stop_reason to "tool_use".
2. `coerce_arguments(args, schema)` — PER-CALL. Coerce arg types against the
   tool's JSON Schema (string->number/bool, null->default) before validation.

Syntax families handled (from OpenClaw's tool-call-repair):
  - Bracket:  [tool_name]\n{...json...}\n[/tool_name]   (also [tool:name]{...})
  - Harmony:  <|channel|>commentary to=tool_name <|message|> {...} <|call|>
  - XML-ish:  <function=tool_name><parameter=p>value</parameter></function>
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from harness.core.types import Response, TextContent, ToolCall

MAX_PAYLOAD_BYTES = 256 * 1024  # cap on a single repaired JSON payload


# --------------------------------------------------------------------------- #
# Balanced-brace JSON finder
# --------------------------------------------------------------------------- #


def find_json_object_end(text: str, start: int) -> int | None:
    """Given `text` and the index of an opening '{', return the index just past
    the matching '}'. Respects strings and escapes. Returns None if unbalanced.
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _extract_json_after(text: str, pos: int) -> tuple[dict[str, Any] | None, int]:
    """Find the next '{...}' object at/after `pos`, parse it. Returns (obj, end)."""
    brace = text.find("{", pos)
    if brace == -1:
        return None, pos
    end = find_json_object_end(text, brace)
    if end is None:
        return None, pos
    blob = text[brace:end]
    if len(blob.encode("utf-8", "replace")) > MAX_PAYLOAD_BYTES:
        return None, end
    try:
        obj = json.loads(blob)
        return (obj if isinstance(obj, dict) else None), end
    except (json.JSONDecodeError, ValueError):
        return None, end


# --------------------------------------------------------------------------- #
# Syntax-family parsers
# --------------------------------------------------------------------------- #

_BRACKET_OPEN = re.compile(r"\[(?:tool:)?([a-zA-Z_][\w.-]*)\]")
_HARMONY = re.compile(r"<\|channel\|>\s*\w+\s+to=([a-zA-Z_][\w.-]*)\b")
_XML_FUNC = re.compile(r"<function=([a-zA-Z_][\w.-]*)>(.*?)</function>", re.DOTALL)
_XML_PARAM = re.compile(r"<parameter=([a-zA-Z_][\w.-]*)>(.*?)</parameter>", re.DOTALL)


def _parse_with_spans(
    text: str, allowed: set[str] | None = None
) -> list[tuple[ToolCall, int, int]]:
    """Like parse_plaintext_tool_calls but also returns each call's (start, end)
    span in `text`, so callers (promotion) can strip the matched prose."""
    found: list[tuple[ToolCall, int, int]] = []

    def ok(name: str) -> bool:
        return allowed is None or name in allowed

    # Bracket: [name] {json}  (the json follows the opening tag)
    for m in _BRACKET_OPEN.finditer(text):
        name = m.group(1)
        if not ok(name):
            continue
        obj, end = _extract_json_after(text, m.end())
        if obj is not None:
            found.append((ToolCall(id=_new_id(), name=name, arguments=obj), m.start(), end))

    # Harmony: <|channel|>commentary to=name <|message|> {json} <|call|>
    for m in _HARMONY.finditer(text):
        name = m.group(1)
        if not ok(name):
            continue
        obj, end = _extract_json_after(text, m.end())
        if obj is not None:
            found.append((ToolCall(id=_new_id(), name=name, arguments=obj), m.start(), end))

    # XML-ish: <function=name><parameter=p>v</parameter>...</function>
    for m in _XML_FUNC.finditer(text):
        name = m.group(1)
        if not ok(name):
            continue
        args = {p: _scalarize(v.strip()) for p, v in _XML_PARAM.findall(m.group(2))}
        found.append((ToolCall(id=_new_id(), name=name, arguments=args), m.start(), m.end()))

    return found


def parse_plaintext_tool_calls(
    text: str, allowed: set[str] | None = None
) -> list[ToolCall]:
    """Scan free text for tool calls in any of the three syntaxes. `allowed`, if
    given, filters to known tool names (rejects hallucinated names)."""
    return _dedup([call for call, _, _ in _parse_with_spans(text, allowed)])


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove the given [start, end) spans from text and tidy whitespace."""
    if not spans:
        return text
    keep: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start >= cursor:
            keep.append(text[cursor:start])
            cursor = end
    keep.append(text[cursor:])
    return "".join(keep).strip()


def _new_id() -> str:
    return f"repair_{uuid.uuid4().hex[:16]}"


def _dedup(calls: list[ToolCall]) -> list[ToolCall]:
    seen: set[tuple[str, str]] = set()
    out: list[ToolCall] = []
    for c in calls:
        key = (c.name, json.dumps(c.arguments, sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _scalarize(v: str) -> Any:
    """Best-effort scalar parse for XML param text (no schema available there)."""
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


# --------------------------------------------------------------------------- #
# Response-level promotion
# --------------------------------------------------------------------------- #


def promote_plaintext_tool_calls(
    response: Response, allowed: set[str] | None = None
) -> Response:
    """If the model emitted no structured tool_calls but wrote them as text,
    extract them and flip stop_reason to "tool_use". No-op if real tool_calls
    already exist.

    The matched tool-call spans are STRIPPED from the text content, so history
    reconstruction (the raw=None fallback) can't double-emit the same call as
    both prose and a structured tool_use block. Any surrounding prose is kept.
    """
    if response.tool_calls:
        return response
    text_blocks = [b for b in response.content if isinstance(b, TextContent)]
    text = "\n".join(b.text for b in text_blocks)
    if not text:
        return response
    spans_calls = _parse_with_spans(text, allowed)
    if not spans_calls:
        return response

    response.tool_calls = _dedup([c for c, _, _ in spans_calls])
    response.stop_reason = "tool_use"

    leftover = _strip_spans(text, [(s, e) for _, s, e in spans_calls])
    non_text = [b for b in response.content if not isinstance(b, TextContent)]
    response.content = non_text + ([TextContent(leftover)] if leftover else [])
    return response


# --------------------------------------------------------------------------- #
# Per-call argument coercion
# --------------------------------------------------------------------------- #


def coerce_arguments(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Coerce argument types against a JSON Schema's `properties`. Handles the
    common model mistakes: numbers/bools sent as strings, missing optionals
    filled from `default`. Unknown keys pass through untouched."""
    props: dict[str, Any] = schema.get("properties", {}) if schema else {}
    out = dict(args)

    for key, spec in props.items():
        if key not in out:
            if "default" in spec:
                out[key] = spec["default"]
            continue
        out[key] = _coerce_value(out[key], spec)

    return out


def _coerce_value(value: Any, spec: dict[str, Any]) -> Any:
    expected = spec.get("type")
    types = expected if isinstance(expected, list) else [expected]

    if "integer" in types and isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return value
    if "number" in types and isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    if "boolean" in types and isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        return value
    if "null" in types and isinstance(value, str) and value.strip().lower() in ("null", "none"):
        return None
    if "array" in types and isinstance(value, str):
        # models sometimes send a JSON array as a string
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return value


def repair_call(call: ToolCall, schema: dict[str, Any]) -> ToolCall:
    """Per-call repair used by the loop: coerce arguments against the tool's
    schema. Structural promotion happens earlier (promote_plaintext_tool_calls).
    """
    call.arguments = coerce_arguments(call.arguments, schema)
    return call
