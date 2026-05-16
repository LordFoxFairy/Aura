"""Compatibility façade for the capabilities-owned builtin tool catalogue.

Runtime stays lazy to avoid a package-init cycle:
``aura.capabilities.tools.catalog`` imports concrete ``aura.tools.<tool>``
modules, and importing a submodule always runs ``aura.tools.__init__`` first.
Static type checking still needs the exported names visible, so TYPE_CHECKING
gets explicit imports while runtime uses ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aura.capabilities.tools.catalog import (
        BUILTIN_STATEFUL_TOOLS,
        BUILTIN_TOOLS,
        AskUserQuestion,
        Bash,
        BashBackground,
        EditFile,
        EnterPlanMode,
        ExitPlanMode,
        Glob,
        Grep,
        MCPReadResourceTool,
        ReadFile,
        SendMessage,
        SkillTool,
        TaskCreate,
        TaskGet,
        TaskList,
        TaskOutput,
        TaskStop,
        TodoWrite,
        ToolError,
        ToolResult,
        WebFetch,
        WebSearch,
        WriteFile,
        assemble_tool_pool,
        bash,
        build_tool,
        edit_file,
        glob,
        grep,
        read_file,
        tool_metadata,
        web_fetch,
        write_file,
    )

_CATALOG_MODULE = "aura.capabilities.tools.catalog"

__all__ = [
    "BUILTIN_STATEFUL_TOOLS",
    "BUILTIN_TOOLS",
    "AskUserQuestion",
    "Bash",
    "BashBackground",
    "EditFile",
    "EnterPlanMode",
    "ExitPlanMode",
    "Glob",
    "Grep",
    "MCPReadResourceTool",
    "ReadFile",
    "SendMessage",
    "SkillTool",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TodoWrite",
    "ToolError",
    "ToolResult",
    "WebFetch",
    "WebSearch",
    "WriteFile",
    "assemble_tool_pool",
    "bash",
    "build_tool",
    "edit_file",
    "glob",
    "grep",
    "read_file",
    "tool_metadata",
    "web_fetch",
    "write_file",
]


def _catalog() -> Any:
    return import_module(_CATALOG_MODULE)


def __getattr__(name: str) -> Any:
    try:
        return getattr(_catalog(), name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
