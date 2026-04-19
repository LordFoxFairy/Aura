"""Tool infrastructure (ToolResult + build_tool) and built-in tools.

Tools are LangChain ``BaseTool`` instances; use ``build_tool`` or
``langchain_core.tools.StructuredTool`` directly.

``BUILTIN_TOOLS`` is the name → tool map that the Agent facade resolves
``config.tools.enabled`` against. Keep extension points tool-package-local
so ``aura/core/agent.py`` stays about orchestration, not tool inventory.
"""

from langchain_core.tools import BaseTool

from aura.tools.base import ToolError, ToolResult, build_tool
from aura.tools.bash import bash
from aura.tools.edit_file import edit_file
from aura.tools.glob import glob
from aura.tools.grep import grep
from aura.tools.read_file import read_file
from aura.tools.web_fetch import web_fetch
from aura.tools.write_file import write_file

BUILTIN_TOOLS: dict[str, BaseTool] = {
    "bash": bash,
    "edit_file": edit_file,
    "glob": glob,
    "grep": grep,
    "read_file": read_file,
    "web_fetch": web_fetch,
    "write_file": write_file,
}

__all__ = [
    "BUILTIN_TOOLS",
    "ToolError",
    "ToolResult",
    "bash",
    "build_tool",
    "edit_file",
    "glob",
    "grep",
    "read_file",
    "web_fetch",
    "write_file",
]
