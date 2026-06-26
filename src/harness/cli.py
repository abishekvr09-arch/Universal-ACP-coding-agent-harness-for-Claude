"""harness CLI — drive the agent directly from a terminal.

In-process: drives `Agent.run()` with no ACP bridge (ACP is for cross-process).
Streams assistant text to stdout via `on_chunk`, persists to the same SQLite store
the ACP server uses (so `--session-id` resumes either), and gates dangerous tools
through a TTY prompt (safe-by-default; non-TTY auto-denies unless `--yes`).

Exit codes: 0 clean, 130 cancelled (Ctrl-C), 1 uncaught error.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import uuid
from typing import Any, TextIO

from harness.core.budget import IterationBudget
from harness.core.loop import Agent, AgentConfig, reconcile_dangling_tool_calls, resolve_tool_timeout
from harness.core.types import CancelToken

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CANCEL = 130

_MODELS = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}

DEFAULT_SYSTEM = (
    "You are a coding agent operating in a terminal. Be precise and concise. Use the "
    "provided tools (read, edit, bash, glob, grep) to inspect and modify the workspace. "
    "Prefer minimal, correct edits. State what you changed."
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description="A Claude-first coding agent.")
    p.add_argument("prompt", nargs="?", help="the task; if omitted, read from stdin")
    p.add_argument("--session-id", help="resume an existing session (new one if omitted)")
    p.add_argument("--model", default="opus", help="opus|sonnet|haiku or a full model id")
    p.add_argument("--system", help="override the system prompt")
    p.add_argument("--max-iterations", type=int, default=90, help="iteration budget cap")
    p.add_argument("--yes", action="store_true", help="auto-approve gated tools (non-interactive)")
    p.add_argument("--no-color", action="store_true", help="disable ANSI output")
    p.add_argument("--db", default=os.environ.get("HARNESS_DB", "harness.db"), help="session DB path")
    p.add_argument("--mcp", help="path to an mcpServers JSON config; adds those tools")
    return p


def resolve_model(name: str) -> str:
    return _MODELS.get(name, name)


def _install_sigint(cancel: CancelToken, err: TextIO) -> None:
    """First Ctrl-C requests cancellation; a second restores default (force-quit).
    No-op if not on the main thread (signals can't be set there)."""
    state = {"n": 0}
    try:
        original = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            state["n"] += 1
            if state["n"] == 1:
                cancel.set()
                err.write("\n[cancelling… press Ctrl-C again to force-quit]\n")
                err.flush()
            else:
                signal.signal(signal.SIGINT, original)
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):
        pass  # not main thread / unsupported — cancellation still works via the API


def run(
    args: argparse.Namespace,
    *,
    provider: Any,
    store: Any,
    stdout: TextIO | None = None,
    stdin: TextIO | None = None,
    stderr: TextIO | None = None,
    install_signals: bool = True,
    interactive: bool | None = None,
    tools: list | None = None,
) -> int:
    from harness.hooks.cli_approval import from_tools as cli_approval_from_tools
    from harness.tools import default_tools

    stdout = stdout or sys.stdout
    stdin = stdin or sys.stdin
    stderr = stderr or sys.stderr
    tools = default_tools() if tools is None else tools

    # MCP tools join the FROZEN set BEFORE the first model call (Law 1). Loading is
    # failure-isolated: a bad server is dropped with a WARN, the run proceeds.
    mcp_clients: list = []
    mcp_runtime = None
    if getattr(args, "mcp", None):
        from harness.mcp import load_mcp_config, load_mcp_tools

        servers = load_mcp_config(args.mcp)
        mcp_tools_list, mcp_clients, mcp_runtime = load_mcp_tools(servers)
        if mcp_tools_list:
            tools = list(tools) + mcp_tools_list
            stderr.write(f"harness: loaded {len(mcp_tools_list)} MCP tool(s)\n")

    prompt_text = args.prompt if args.prompt is not None else stdin.read()
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        stderr.write("harness: no prompt given (argument or stdin)\n")
        return EXIT_ERROR

    # Session resume vs. new — restore the system prompt byte-for-byte on resume.
    session_id = args.session_id or f"cli-{uuid.uuid4().hex[:12]}"
    system = args.system or DEFAULT_SYSTEM
    existing_system = store.get_system(session_id) if args.session_id else None
    if existing_system is None:
        store.create_session(session_id, system, model=resolve_model(args.model))
        messages: list[dict[str, Any]] = []
    else:
        system = existing_system
        messages = store.load_messages(session_id)

    # Persist the new user turn (the loop persists assistant + tool_result turns).
    user_msg = {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
    messages.append(user_msg)
    # If a prior turn was interrupted mid-tool-execution (crash between persisting the
    # tool_use turn and its tool_result turn), fold synthetic 'interrupted' results
    # into THIS user turn so the resumed request satisfies one-result-per-tool_use.
    # Persist the turn in its final (reconciled) form — single append, idempotent.
    reconcile_dangling_tool_calls(messages)
    store.append_message(session_id, messages[-1])

    if interactive is None:
        interactive = stdout.isatty() if hasattr(stdout, "isatty") else False
    approval = cli_approval_from_tools(tools, assume_yes=args.yes, interactive=interactive)

    streamed = {"any": False}

    def on_chunk(text: str) -> None:
        streamed["any"] = True
        stdout.write(text)
        stdout.flush()

    cancel = CancelToken()
    if install_signals:
        _install_sigint(cancel, stderr)

    budget = IterationBudget(args.max_iterations)
    agent = Agent(AgentConfig(
        provider=provider,
        tools=tools,
        system=system,
        hooks=[approval],
        budget=budget,
        persist=store.persist_fn(session_id),
        on_chunk=on_chunk,
        tool_timeout=resolve_tool_timeout(),
    ))

    try:
        final = agent.run(messages, cancel=cancel)
    except KeyboardInterrupt:
        stderr.write("\nharness: force-quit\n")
        return EXIT_CANCEL
    except Exception as e:  # noqa: BLE001 — top-level: report, don't traceback-dump
        stderr.write(f"harness: error: {type(e).__name__}: {e}\n")
        return EXIT_ERROR
    finally:
        for c in mcp_clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        if mcp_runtime is not None:
            mcp_runtime.shutdown()

    # If the provider didn't stream (e.g. a non-streaming backend), print the final.
    if not streamed["any"]:
        text = "".join(
            getattr(b, "text", "") for b in final.content if getattr(b, "type", None) == "text"
        )
        if text:
            stdout.write(text)
    stdout.write("\n")
    stdout.flush()

    if cancel.is_set():
        return EXIT_CANCEL
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    from harness.providers import ClaudeProvider
    from harness.session import SessionStore

    provider = ClaudeProvider(model=resolve_model(args.model))
    store = SessionStore(args.db)
    return run(args, provider=provider, store=store)


if __name__ == "__main__":
    sys.exit(main())
