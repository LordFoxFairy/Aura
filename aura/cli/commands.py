"""Slash command façade for the REPL.

This module used to contain a hardcoded if-else dispatcher. v0.1.1 moved
the real implementation to :mod:`aura.core.commands` (registry + builtins)
so Skills and MCP integrations can register commands dynamically. This
file remains as a thin façade to keep the ``aura.cli.commands`` import
path stable for existing callers.
"""

from __future__ import annotations

from aura.core.agent import Agent
from aura.core.commands import Command, CommandRegistry, CommandResult
from aura.core.commands.builtin import (
    ClearCommand,
    ExitCommand,
    HelpCommand,
    ModelCommand,
)

__all__ = [
    "Command",
    "CommandRegistry",
    "CommandResult",
    "build_default_registry",
    "dispatch",
]


def build_default_registry() -> CommandRegistry:
    """Return a registry pre-populated with Aura's built-in commands."""
    r = CommandRegistry()
    # HelpCommand needs the registry to enumerate commands at /help time.
    r.register(HelpCommand(registry=r))
    r.register(ExitCommand())
    r.register(ClearCommand())
    r.register(ModelCommand())
    return r


async def dispatch(
    line: str, agent: Agent, registry: CommandRegistry
) -> CommandResult:
    """Dispatch a REPL input line via the given registry."""
    return await registry.dispatch(line, agent)
