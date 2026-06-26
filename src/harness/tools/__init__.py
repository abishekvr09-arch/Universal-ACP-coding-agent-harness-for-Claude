"""Built-in tools. `default_tools()` returns the standard 5-tool MVP set."""

from harness.core.types import Tool
from harness.tools.bash import bash
from harness.tools.edit import edit
from harness.tools.glob import glob
from harness.tools.grep import grep
from harness.tools.read import read


def default_tools() -> list[Tool]:
    return [read, edit, bash, glob, grep]


__all__ = ["bash", "default_tools", "edit", "glob", "grep", "read"]
