"""Core types for the slash-command abstraction.

``Command`` is a ``Protocol`` — any object with the required attributes and
``async def handle(...)`` satisfies it. This is deliberate: Skills and MCP
integrations will plug in their own command objects without inheriting from
an Aura ABC. Structural typing only; no ``@runtime_checkable`` (we don't
need ``isinstance`` checks, and enabling it would slow ``dict[str, Command]``
lookups with no real benefit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from aura.core.agent import Agent


CommandKind = Literal["print", "exit", "noop"]
CommandSource = Literal["builtin", "skill", "mcp"]


@dataclass(frozen=True)
class CommandResult:
    """Outcome of dispatching a user input line.

    - ``handled=False, kind="noop"`` → the line is a normal prompt; the REPL
      should send it to the agent.
    - ``handled=True, kind="print"`` → REPL prints ``text`` and re-prompts.
    - ``handled=True, kind="exit"`` → REPL exits cleanly.
    """

    handled: bool
    kind: CommandKind
    text: str


class Command(Protocol):
    """Structural type for a slash command.

    Attributes:
        name: invocation token including the leading slash (e.g. ``/help``).
        description: one-line summary rendered by ``/help``.
        source: where the command came from — shown grouped by ``/help``
            once Skills/MCP commands exist.
    """

    name: str
    description: str
    source: CommandSource

    async def handle(
        self, arg: str, agent: Agent
    ) -> CommandResult:  # pragma: no cover - protocol stub
        ...
