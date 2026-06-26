"""Tool-call repair — the three syntax families, coercion, and promotion."""

from __future__ import annotations

from harness.core.repair import (
    coerce_arguments,
    find_json_object_end,
    parse_plaintext_tool_calls,
    promote_plaintext_tool_calls,
)
from harness.core.types import Response, TextContent, ToolCall


def test_balanced_brace_finder_handles_strings_and_escapes():
    s = r'{"a": "}{ tricky \" still in", "b": {"c": 1}}'
    assert find_json_object_end(s, 0) == len(s)


def test_balanced_brace_unbalanced_returns_none():
    assert find_json_object_end('{"a": 1', 0) is None


def test_bracket_syntax():
    calls = parse_plaintext_tool_calls('[read]\n{"path": "a.txt"}', {"read"})
    assert calls and calls[0].name == "read" and calls[0].arguments == {"path": "a.txt"}


def test_harmony_syntax():
    text = '<|channel|>commentary to=bash <|message|> {"cmd": "ls"} <|call|>'
    calls = parse_plaintext_tool_calls(text, {"bash"})
    assert calls and calls[0].name == "bash" and calls[0].arguments == {"cmd": "ls"}


def test_xml_syntax_with_scalar_coercion():
    text = "<function=edit><parameter=line>42</parameter><parameter=keep>true</parameter></function>"
    calls = parse_plaintext_tool_calls(text, {"edit"})
    assert calls[0].arguments == {"line": 42, "keep": True}


def test_hallucinated_name_rejected():
    assert parse_plaintext_tool_calls("[ghost]\n{}", {"read"}) == []


def test_coercion_against_schema():
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "f": {"type": "number"},
            "b": {"type": "boolean"},
            "opt": {"type": "string", "default": "x"},
        },
    }
    out = coerce_arguments({"n": "42", "f": "3.5", "b": "false"}, schema)
    assert out == {"n": 42, "f": 3.5, "b": False, "opt": "x"}


def test_coercion_leaves_unparseable_alone():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert coerce_arguments({"n": "not-a-number"}, schema) == {"n": "not-a-number"}


def test_promotion_flips_stop_reason():
    r = Response(
        content=[TextContent('[read]\n{"path":"x"}')], tool_calls=[], stop_reason="end_turn"
    )
    promote_plaintext_tool_calls(r, {"read"})
    assert r.stop_reason == "tool_use" and r.tool_calls[0].name == "read"


def test_promotion_noop_when_real_calls_present():
    r = Response(content=[], tool_calls=[ToolCall("i", "read", {})], stop_reason="tool_use")
    promote_plaintext_tool_calls(r, {"read"})
    assert len(r.tool_calls) == 1
