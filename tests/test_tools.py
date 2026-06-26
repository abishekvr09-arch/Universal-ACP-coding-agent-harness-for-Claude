"""The 5 built-in tools against a real temp workspace."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools import bash, default_tools, edit, glob, grep, read


def test_default_tool_set():
    assert [t.name for t in default_tools()] == ["read", "edit", "bash", "glob", "grep"]


def test_dangerous_tools_are_gated_and_sequential():
    assert bash.requires_approval and bash.execution_mode == "sequential"
    assert not bash.parallel_safe and "execute" in bash.tags
    assert edit.requires_approval and not edit.parallel_safe


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "hello.py").write_text(
        'def greet():\n    return "hi"\n\nprint(greet())\n', encoding="utf-8"
    )
    return tmp_path


def test_read_with_line_numbers(workspace: Path):
    r = read.handler(path=str(workspace / "hello.py"))
    assert "1\tdef greet():" in r.content[0].text and not r.is_error


def test_read_offset_limit(workspace: Path):
    r = read.handler(path=str(workspace / "hello.py"), offset=2, limit=1)
    assert r.content[0].text == "3\t"  # blank line 3


def test_read_missing_file():
    r = read.handler(path="does-not-exist.xyz")
    assert r.is_error and "no such file" in r.content[0].text


def test_edit_unique_replace(workspace: Path):
    f = workspace / "hello.py"
    r = edit.handler(path=str(f), old_string='return "hi"', new_string='return "hello"')
    assert not r.is_error and "hello" in f.read_text(encoding="utf-8")


def test_edit_non_unique_rejected(workspace: Path):
    f = workspace / "dup.txt"
    f.write_text("x\nx\n", encoding="utf-8")
    r = edit.handler(path=str(f), old_string="x", new_string="y")
    assert r.is_error and "not unique" in r.content[0].text


def test_edit_replace_all(workspace: Path):
    f = workspace / "dup.txt"
    f.write_text("x\nx\n", encoding="utf-8")
    r = edit.handler(path=str(f), old_string="x", new_string="y", replace_all=True)
    assert not r.is_error and f.read_text(encoding="utf-8") == "y\ny\n"


def test_glob(workspace: Path):
    r = glob.handler(pattern="*.py", path=str(workspace))
    assert "hello.py" in r.content[0].text


def test_grep(workspace: Path):
    r = grep.handler(pattern="greet", path=str(workspace))
    assert "greet" in r.content[0].text and not r.is_error


def test_bash_success():
    r = bash.handler(command="echo harness-ok")
    assert "harness-ok" in r.content[0].text and "exit code: 0" in r.content[0].text


def test_bash_failure_is_error():
    r = bash.handler(command="exit 3")
    assert r.is_error and "exit code: 3" in r.content[0].text
