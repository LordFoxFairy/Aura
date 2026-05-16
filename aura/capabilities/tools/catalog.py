"""Capabilities-owned builtin tool catalogue.

The actual tool implementations remain under ``aura.tools.*``; this module
owns the assembly of the builtin stateless and stateful tool catalogues so
other layers can import a single authoritative source.
"""

from collections.abc import Iterable

from langchain_core.tools import BaseTool

from aura.schemas.tool import ToolError, ToolResult, tool_metadata
from aura.tools.ask_user import AskUserQuestion
from aura.tools.base import build_tool
from aura.tools.bash import Bash, bash
from aura.tools.bash_background import BashBackground
from aura.tools.edit_file import EditFile, edit_file
from aura.tools.enter_plan_mode import EnterPlanMode
from aura.tools.exit_plan_mode import ExitPlanMode
from aura.tools.glob import Glob, glob
from aura.tools.grep import Grep, grep
from aura.tools.mcp_read_resource import MCPReadResourceTool
from aura.tools.read_file import ReadFile, read_file
from aura.tools.send_message import SendMessage
from aura.tools.skill import SkillTool
from aura.tools.task_create import TaskCreate
from aura.tools.task_get import TaskGet
from aura.tools.task_list import TaskList
from aura.tools.task_output import TaskOutput
from aura.tools.task_stop import TaskStop
from aura.tools.todo_write import TodoWrite
from aura.tools.web_fetch import WebFetch, web_fetch
from aura.tools.web_search import WebSearch
from aura.tools.write_file import WriteFile, write_file

BUILTIN_TOOLS: dict[str, BaseTool] = {
    "bash": bash,
    "edit_file": edit_file,
    "glob": glob,
    "grep": grep,
    "read_file": read_file,
    "web_fetch": web_fetch,
    "write_file": write_file,
}

BUILTIN_STATEFUL_TOOLS: dict[str, type[BaseTool]] = {
    "todo_write": TodoWrite,
    "ask_user_question": AskUserQuestion,
    "task_create": TaskCreate,
    "task_output": TaskOutput,
    "task_get": TaskGet,
    "task_list": TaskList,
    "task_stop": TaskStop,
    "web_search": WebSearch,
    "enter_plan_mode": EnterPlanMode,
    "exit_plan_mode": ExitPlanMode,
    "bash_background": BashBackground,
    "skill": SkillTool,
    "send_message": SendMessage,
}


def assemble_tool_pool(
    builtins: Iterable[BaseTool],
    mcp_tools: Iterable[BaseTool],
    *,
    mcp_overrides: bool = False,
) -> dict[str, BaseTool]:
    """Merge builtin + MCP tools into a stable, dedup'd, ordered mapping."""
    from aura.core import journal  # noqa: PLC0415

    builtin_sorted = sorted(builtins, key=lambda t: t.name)
    mcp_sorted = sorted(mcp_tools, key=lambda t: t.name)

    pool: dict[str, BaseTool] = {}
    builtin_names: set[str] = set()
    for tool in builtin_sorted:
        if tool.name in pool:
            continue
        pool[tool.name] = tool
        builtin_names.add(tool.name)

    for tool in mcp_sorted:
        if tool.name in pool:
            collides_with_builtin = tool.name in builtin_names
            if collides_with_builtin and mcp_overrides:
                pool[tool.name] = tool
                journal.write(
                    "mcp_tool_shadowed",
                    tool=tool.name,
                    winner="mcp",
                    shadowed_by="mcp",
                )
                continue
            winner = "builtin" if collides_with_builtin else "mcp"
            journal.write(
                "mcp_tool_shadowed",
                tool=tool.name,
                winner=winner,
                shadowed_by=winner,
            )
            continue
        pool[tool.name] = tool

    return pool


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
