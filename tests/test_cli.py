"""CLI surface: argparse, stdin, approval paths, resume roundtrip, exit codes."""

from __future__ import annotations

import io
from pathlib import Path

from conftest import FakeProvider, assistant_text, assistant_tool_use, echo_tool, tool_use

from harness.cli import EXIT_CANCEL, EXIT_ERROR, EXIT_OK, build_arg_parser, resolve_model, run
from harness.hooks.cli_approval import from_tools as cli_approval_from_tools
from harness.core.types import Deny, Tool, ToolCall, ToolResult, TextContent
from harness.session import SessionStore


# --------------------------------------------------------------------------- #
# argparse + model resolution
# --------------------------------------------------------------------------- #


def test_argparse_defaults():
    a = build_arg_parser().parse_args(["do it"])
    assert a.prompt == "do it" and a.model == "opus" and a.yes is False


def test_model_resolution():
    assert resolve_model("opus") == "claude-opus-4-8"
    assert resolve_model("haiku") == "claude-haiku-4-5"
    assert resolve_model("claude-custom-1") == "claude-custom-1"  # pass-through


# --------------------------------------------------------------------------- #
# approval hook paths
# --------------------------------------------------------------------------- #


def _gated_tool():
    return Tool(
        name="bash", description="run", input_schema={"type": "object"},
        handler=lambda cancel=None, **k: ToolResult(content=[TextContent("ran")]),
        requires_approval=True, tags=("execute",),
    )


def test_cli_approval_yes_allows():
    hook = cli_approval_from_tools([_gated_tool()], ask=lambda p: "y")
    assert isinstance(hook.before_tool(ToolCall("i", "bash", {"command": "ls"})), ToolCall)


def test_cli_approval_no_denies():
    hook = cli_approval_from_tools([_gated_tool()], ask=lambda p: "n")
    assert isinstance(hook.before_tool(ToolCall("i", "bash", {"command": "ls"})), Deny)


def test_cli_approval_always_grants_for_session():
    asked = {"n": 0}

    def ask(p):
        asked["n"] += 1
        return "a"

    hook = cli_approval_from_tools([_gated_tool()], ask=ask)
    c1 = hook.before_tool(ToolCall("i1", "bash", {"command": "ls"}))
    c2 = hook.before_tool(ToolCall("i2", "bash", {"command": "pwd"}))
    assert isinstance(c1, ToolCall) and isinstance(c2, ToolCall)
    assert asked["n"] == 1  # only prompted once; 'always' remembered


def test_cli_approval_non_interactive_auto_denies():
    hook = cli_approval_from_tools([_gated_tool()], interactive=False)
    assert isinstance(hook.before_tool(ToolCall("i", "bash", {"command": "ls"})), Deny)


def test_cli_approval_non_interactive_with_yes_allows():
    hook = cli_approval_from_tools([_gated_tool()], interactive=False, assume_yes=True)
    assert isinstance(hook.before_tool(ToolCall("i", "bash", {"command": "ls"})), ToolCall)


def test_cli_approval_ungated_tool_passes():
    hook = cli_approval_from_tools([_gated_tool()], ask=lambda p: "n")
    # 'read' is not gated -> always allowed
    assert isinstance(hook.before_tool(ToolCall("i", "read", {"path": "x"})), ToolCall)


# --------------------------------------------------------------------------- #
# run(): prompt sources, exit codes, streaming, resume
# --------------------------------------------------------------------------- #


def _args(**kw):
    base = dict(
        prompt=None, session_id=None, model="opus", system=None,
        max_iterations=90, yes=False, no_color=True, db=":memory:",
    )
    base.update(kw)
    return type("A", (), base)()


def test_run_reads_prompt_from_stdin(tmp_path: Path):
    store = SessionStore(tmp_path / "s.db")
    prov = FakeProvider([assistant_text("hello back")])
    out = io.StringIO()
    code = run(
        _args(prompt=None), provider=prov, store=store,
        stdout=out, stdin=io.StringIO("from stdin"), install_signals=False, interactive=False,
    )
    assert code == EXIT_OK
    # provider saw the stdin prompt
    seen = prov.seen_messages[0]
    assert seen[0]["content"][0]["text"] == "from stdin"


def test_run_empty_prompt_is_error(tmp_path: Path):
    store = SessionStore(tmp_path / "s.db")
    err = io.StringIO()
    code = run(
        _args(prompt=None), provider=FakeProvider([]), store=store,
        stdout=io.StringIO(), stdin=io.StringIO("   "), stderr=err, install_signals=False,
    )
    assert code == EXIT_ERROR and "no prompt" in err.getvalue()


def test_run_prints_final_when_not_streamed(tmp_path: Path):
    # FakeProvider doesn't stream -> run() prints the final text itself
    store = SessionStore(tmp_path / "s.db")
    out = io.StringIO()
    code = run(
        _args(prompt="hi"), provider=FakeProvider([assistant_text("the answer")]),
        store=store, stdout=out, install_signals=False, interactive=False,
    )
    assert code == EXIT_OK and "the answer" in out.getvalue()


def test_run_session_resume_roundtrip(tmp_path: Path):
    db = tmp_path / "s.db"
    # turn 1: new session
    store1 = SessionStore(db)
    run(
        _args(prompt="first", session_id="resume-me", system="CUSTOM SYS"),
        provider=FakeProvider([assistant_text("ok1")]),
        store=store1, stdout=io.StringIO(), install_signals=False, interactive=False,
    )
    # turn 2: resume -> must load prior history AND restore the system prompt
    store2 = SessionStore(db)
    prov2 = FakeProvider([assistant_text("ok2")])
    run(
        _args(prompt="second", session_id="resume-me"),
        provider=prov2, store=store2, stdout=io.StringIO(),
        install_signals=False, interactive=False,
    )
    # the provider on turn 2 saw the prior turn's messages + the new prompt
    seen = prov2.seen_messages[0]
    texts = [b.get("text") for m in seen for b in m["content"] if isinstance(b, dict)]
    assert "first" in texts and "ok1" in texts and "second" in texts
    # system prompt was restored byte-for-byte (Law 1), not the default
    assert store2.get_system("resume-me") == "CUSTOM SYS"


def test_run_gated_tool_denied_non_interactive(tmp_path: Path):
    # bash is gated; non-interactive without --yes -> denied -> error_result, no crash
    store = SessionStore(tmp_path / "s.db")
    prov = FakeProvider([assistant_tool_use(tool_use("bash", command="rm -rf /")), assistant_text("done")])
    out = io.StringIO()
    code = run(
        _args(prompt="delete"), provider=prov, store=store, tools=[_gated_tool()],
        stdout=out, install_signals=False, interactive=False,
    )
    assert code == EXIT_OK  # ran cleanly; the tool was denied internally
    results = [
        b for m in store.load_messages(_sid(store)) if m["role"] == "user"
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert any(r["is_error"] for r in results)


def _sid(store) -> str:
    import sqlite3
    return sqlite3.connect(store.path).execute("SELECT id FROM sessions LIMIT 1").fetchone()[0]


def test_run_error_path_returns_exit_error(tmp_path: Path):
    class Boom:
        profile = None
        hooks = None

        def stream(self, *a, **k):
            raise RuntimeError("provider exploded")

    store = SessionStore(tmp_path / "s.db")
    err = io.StringIO()
    code = run(
        _args(prompt="hi"), provider=Boom(), store=store,
        stdout=io.StringIO(), stderr=err, install_signals=False, interactive=False,
    )
    assert code == EXIT_ERROR and "provider exploded" in err.getvalue()


# --------------------------------------------------------------------------- #
# --mcp config wiring
# --------------------------------------------------------------------------- #


def test_mcp_config_loader(tmp_path: Path):
    from harness.mcp import load_mcp_config

    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        '{"mcpServers": {"git": {"command": "uvx", "args": ["mcp-git"]}, '
        '"bad": {"no_command": true}}}',
        encoding="utf-8",
    )
    servers = load_mcp_config(cfg)
    assert "git" in servers and servers["git"]["command"] == "uvx"
    assert "bad" not in servers  # entries without a command are dropped


def test_cli_mcp_arg_parses():
    a = build_arg_parser().parse_args(["go", "--mcp", "mcp.json"])
    assert a.mcp == "mcp.json"
