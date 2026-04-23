"""Built-in slash commands — the four migrated from the old if-else.

Each class satisfies the :class:`aura.core.commands.types.Command`
Protocol by duck-typing (no ABC inheritance).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.config.schema import AuraConfigError
from aura.core.commands.registry import CommandRegistry
from aura.core.commands.types import CommandResult, CommandSource

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
        lines.append(
            "Keybindings: shift+tab cycles permission mode "
            "(default -> accept_edits -> plan) · esc resets to default."
        )
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


class CompactCommand:
    """``/compact`` — summarize old history + preserve session state."""

    name = "/compact"
    description = "summarize history + preserve state"
    source: CommandSource = "builtin"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        result = await agent.compact(source="manual")
        return CommandResult(
            handled=True,
            kind="print",
            text=(
                f"compact applied ({result.before_tokens} -> "
                f"{result.after_tokens} tokens)"
            ),
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
        old = agent.current_model or "?"
        try:
            agent.switch_model(arg)
        except AuraConfigError as exc:
            # Catches UnknownModelSpecError, MissingCredentialError,
            # MissingProviderDependencyError — any resolve/create failure
            # surfaces as a printable error instead of crashing the REPL.
            return CommandResult(
                handled=True, kind="print", text=f"error: {exc}"
            )
        new = agent.current_model or arg
        return CommandResult(
            handled=True, kind="print", text=f"model: {old} → {new}"
        )


def _model_status(agent: Agent) -> str:
    current = agent.current_model or "?"
    aliases = sorted(agent.router_aliases)
    lines = [f"current: {current}"]
    if aliases:
        # Align alias names so the arrows line up — matches claude-code's
        # ``/model`` picker formatting. Width from the longest alias name.
        width = max(len(a) for a in aliases)
        lines.append("aliases:")
        for alias in aliases:
            lines.append(
                f"  {alias:<{width}} → {agent.router_aliases[alias]}"
            )
    return "\n".join(lines)
