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
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    def __init__(self, *, registry: CommandRegistry) -> None:
        # Keep a reference to the registry so /help always reflects the
        # live set of commands (critical once Skills/MCP register at runtime).
        self._registry = registry

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        commands = self._registry.list()
        # Group by source so the picker mirrors claude-code: Builtins →
        # Skills → MCP. Skipping an empty group keeps the output tight for
        # users who haven't wired any skills or MCP yet. Alignment is
        # preserved WITHIN each group (24-col label column, same as the
        # pre-grouping format); descriptions collapse to the first
        # non-empty line so a multi-line ``description:`` frontmatter
        # doesn't shatter the column. Mirrors the slash-completer fix
        # that already applies ``.split("\n", 1)[0].strip()`` to
        # ``display_meta``.
        sections: list[tuple[str, CommandSource]] = [
            ("Builtins", "builtin"),
            ("Skills", "skill"),
            ("MCP", "mcp"),
        ]
        lines = ["Available commands:"]
        for heading, source in sections:
            group = [c for c in commands if c.source == source]
            if not group:
                continue
            lines.append("")
            lines.append(f"  {heading}:")
            for cmd in group:
                hint = getattr(cmd, "argument_hint", None)
                label = f"{cmd.name} {hint}" if hint else cmd.name
                # Collapse multi-line description to its first non-empty
                # line so a stray ``description:`` that spans multiple
                # paragraphs (common when skills are authored by LLMs)
                # doesn't shatter the 24-col alignment. Mirrors the
                # slash-completer fix at ``aura/cli/completion.py``
                # (``display_meta`` gets the same treatment).
                description = cmd.description.split("\n", 1)[0].strip()
                lines.append(f"    {label:<24} {description}")
        lines.append("")
        lines.append(
            "Keybindings: shift+tab cycles permission mode "
            "(default -> accept_edits -> plan) · esc resets to default."
        )
        lines.append("Anything else is sent as a prompt to the agent.")
        return CommandResult(
            handled=True, kind="view", text="\n".join(lines)
        )


class ExitCommand:
    """``/exit`` — signal the REPL to terminate."""

    name = "/exit"
    description = "exit the REPL (Ctrl+D also works)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        return CommandResult(handled=True, kind="exit", text="")


class ClearCommand:
    """``/clear`` — reset the current session's history."""

    name = "/clear"
    description = "clear the current session's history"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

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
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

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
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[spec]"

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
