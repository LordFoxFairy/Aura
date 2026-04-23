"""Built-in tools — stateless singletons and stateful classes.

Stateless tools are shared module-level singletons (Pythonic: one canonical
instance, cheap to reuse, no per-Agent state). They live in ``BUILTIN_TOOLS``.

Stateful tools declare a ``state: LoopState`` pydantic field; they cannot be
module-level because each Agent needs its own instance bound to its own
``LoopState``. Their classes live in ``BUILTIN_STATEFUL_TOOLS``; the Agent
constructs an instance per session.

``Agent.__init__`` composes the two: start with ``BUILTIN_TOOLS`` (or the
caller-supplied ``available_tools``), then instantiate each stateful class
with ``self._state`` and overlay it into the same dict.
"""

from langchain_core.tools import BaseTool

from aura.schemas.tool import ToolError, ToolResult, tool_metadata
from aura.tools.ask_user import AskUserQuestion
from aura.tools.base import build_tool
from aura.tools.bash import Bash, bash
from aura.tools.edit_file import EditFile, edit_file
from aura.tools.glob import Glob, glob
from aura.tools.grep import Grep, grep
from aura.tools.read_file import ReadFile, read_file
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
}

__all__ = [
    "BUILTIN_STATEFUL_TOOLS",
    "BUILTIN_TOOLS",
    "AskUserQuestion",
    "Bash",
    "EditFile",
    "Glob",
    "Grep",
    "ReadFile",
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
