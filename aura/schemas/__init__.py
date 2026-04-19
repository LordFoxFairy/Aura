"""Cross-layer data types — the foundation leaf of the aura package.

Everything in this package is pure data (or data + serialization helpers):
event types emitted by the loop, the mutable ``LoopState`` carried across
turns, typed pydantic schemas (``TodoItem``), and the tool execution
protocol (``ToolResult`` / ``ToolError`` / ``tool_metadata``).

Invariant: nothing under ``aura/`` imports back into here. Core, tools, and
cli all import FROM ``aura.schemas``; no module here imports any other
``aura`` module. This leaves the dependency direction unambiguous and
eliminates the "tools depend on core, core imports from tools" confusion
that arises when shared types hide inside a domain package.
"""

from aura.schemas.events import (
    AgentEvent,
    AssistantDelta,
    Final,
    ToolCallCompleted,
    ToolCallStarted,
)
from aura.schemas.state import LoopState
from aura.schemas.todos import TodoItem, TodoStatus
from aura.schemas.tool import ToolError, ToolResult, tool_metadata

__all__ = [
    "AgentEvent",
    "AssistantDelta",
    "Final",
    "LoopState",
    "TodoItem",
    "TodoStatus",
    "ToolCallCompleted",
    "ToolCallStarted",
    "ToolError",
    "ToolResult",
    "tool_metadata",
]
