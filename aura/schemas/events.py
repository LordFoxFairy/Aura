"""Event types emitted by the agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AssistantDelta:
    """Incremental text output from the assistant."""

    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    """Tool invocation initiated."""

    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolCallCompleted:
    """Tool invocation finished with result or error."""

    name: str
    output: Any
    error: str | None = None


@dataclass(frozen=True)
class Final:
    """Final concatenated text response from the agent loop."""

    message: str


AgentEvent = AssistantDelta | ToolCallStarted | ToolCallCompleted | Final
"""Type alias for all events emitted by the agent loop."""
