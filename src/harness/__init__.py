"""harness — a focused, Claude-first coding agent harness.

A composable tool-calling loop with three extension seams (tools, providers,
hooks), built around the one-tool_result-per-tool_use invariant and strict
prompt-cache discipline. See CLAUDE.md for the full design.
"""

__version__ = "0.0.1"
