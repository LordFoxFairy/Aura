"""Core types for the slash-command abstraction.

``Command`` is a ``Protocol`` — any object with the required attributes and
``async def handle(...)`` satisfies it. This keeps command implementations
structurally typed without forcing inheritance from an Aura base class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from aura.core.agent import Agent


CommandKind = Literal["print", "view", "exit", "noop"]
CommandSource = Literal["builtin", "skill", "mcp"]


@dataclass(frozen=True)
class CommandResult:
    """Outcome of dispatching a user input line."""

    handled: bool
    kind: CommandKind
    text: str


class Command(Protocol):
    """Structural type for a slash command."""

    name: str
    description: str
    source: CommandSource
    allowed_tools: tuple[str, ...]
    argument_hint: str | None

    async def handle(
        self, arg: str, agent: Agent
    ) -> CommandResult:  # pragma: no cover - protocol stub
        ...


__all__ = [
    "Command",
    "CommandKind",
    "CommandResult",
    "CommandSource",
]
