"""Cache-breakpoint placement and Claude provider wiring (no live API calls)."""

from __future__ import annotations

from harness.providers import (
    ClaudeProvider,
    apply_cache_breakpoints,
    default_profile,
    mark_system,
    to_anthropic_tools,
)
from harness.providers.cache import _EPHEMERAL
from harness.core.types import Tool


def test_breakpoints_on_last_three_only_and_copy():
    msgs = [{"role": "user", "content": [{"type": "text", "text": str(i)}]} for i in range(5)]
    marked = apply_cache_breakpoints(msgs)
    # live list untouched
    assert all("cache_control" not in m["content"][-1] for m in msgs)
    cc = [m["content"][-1].get("cache_control") for m in marked]
    assert cc == [None, None, _EPHEMERAL, _EPHEMERAL, _EPHEMERAL]


def test_string_content_normalized_and_marked():
    out = apply_cache_breakpoints([{"role": "user", "content": "hi"}])
    block = out[0]["content"][0]
    assert block["type"] == "text" and block["cache_control"] == _EPHEMERAL


def test_mark_system():
    blocks = mark_system("SYS")
    assert blocks[0]["text"] == "SYS" and blocks[0]["cache_control"] == _EPHEMERAL


def test_profile_reasoning_is_top_level_for_anthropic():
    p = default_profile()
    assert p.reasoning_in_extra_body is False
    assert "claude-opus-4-8" in p.supported_models


def test_api_kwargs_split():
    prov = ClaudeProvider(model="claude-opus-4-8", api_key="x")
    extra_body, top_level = prov.hooks.build_api_kwargs()
    assert extra_body == {}
    assert top_level == {"thinking": {"type": "adaptive"}}


def test_tool_conversion_passes_json_schema():
    tools = [
        Tool(
            name="read",
            description="read",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    ]
    converted = to_anthropic_tools(tools)
    assert converted[0]["name"] == "read"
    assert converted[0]["input_schema"]["properties"]["path"]["type"] == "string"
