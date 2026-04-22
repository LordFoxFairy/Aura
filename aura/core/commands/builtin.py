"""Built-in slash commands — the four migrated from the old if-else.

Each class satisfies the :class:`aura.core.commands.types.Command`
Protocol by duck-typing (no ABC inheritance).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.core.commands.registry import CommandRegistry
from aura.core.commands.types import CommandResult, CommandSource
from aura.core.llm import UnknownModelSpecError

if TYPE_CHECKING:
    from aura.core.agent import Agent


class HelpCommand:
    """``/help`` — enumerate every registered command."""

    name = "/help"
    description = "show this message"
    source: CommandSource = "builtin"

    def __init__(self, *, registry: CommandRegistry) -> None:
        # Keep a reference to the registry so /help always reflects the
        # live set of commands (critical once Skills/MCP register at runtime).
        self._registry = registry

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        commands = self._registry.list()
        # Column-align name -> description; width 14 covers "/model <spec>".
        lines = ["Available commands:"]
        for cmd in commands:
            lines.append(f"  {cmd.name:<14} {cmd.description}")
        lines.append("")
        lines.append("Anything else is sent as a prompt to the agent.")
        return CommandResult(
            handled=True, kind="print", text="\n".join(lines)
        )


class ExitCommand:
    """``/exit`` — signal the REPL to terminate."""

    name = "/exit"
    description = "exit the REPL (Ctrl+D also works)"
    source: CommandSource = "builtin"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        return CommandResult(handled=True, kind="exit", text="")


class ClearCommand:
    """``/clear`` — reset the current session's history."""

    name = "/clear"
    description = "clear the current session's history"
    source: CommandSource = "builtin"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        agent.clear_session()
        return CommandResult(
            handled=True, kind="print", text="session cleared"
        )


class ModelCommand:
    """``/model [spec]`` — show or switch the current model."""

    name = "/model"
    description = "show or switch model (no arg = status)"
    source: CommandSource = "builtin"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        if not arg:
            return CommandResult(
                handled=True, kind="print", text=_model_status(agent)
            )
        try:
            agent.switch_model(arg)
        except UnknownModelSpecError as exc:
            return CommandResult(
                handled=True, kind="print", text=f"error: {exc}"
            )
        return CommandResult(
            handled=True, kind="print", text=f"switched to {arg}"
        )


def _model_status(agent: Agent) -> str:
    default = agent.current_model or "?"
    aliases = sorted(agent.router_aliases)
    lines = [f"current default: {default}"]
    if aliases:
        lines.append("aliases:")
        for alias in aliases:
            lines.append(f"  {alias} -> {agent.router_aliases[alias]}")
    return "\n".join(lines)
