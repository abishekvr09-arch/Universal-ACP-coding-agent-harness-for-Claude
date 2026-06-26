"""Shared helpers for the built-in tools."""

from __future__ import annotations

from harness.core.types import TextContent, ToolResult


def text_result(text: str, *, is_error: bool = False, terminate: bool = False) -> ToolResult:
    return ToolResult(content=[TextContent(text)], is_error=is_error, terminate=terminate)


def err(message: str) -> ToolResult:
    return text_result(message, is_error=True)
