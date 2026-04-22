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
    CompactCommand,
    ExitCommand,
    HelpCommand,
    ModelCommand,
)
from aura.core.commands.tasks import TasksCommand
from aura.core.skills.command import SkillCommand

__all__ = [
    "Command",
    "CommandRegistry",
    "CommandResult",
    "build_default_registry",
    "dispatch",
]


def build_default_registry(agent: Agent | None = None) -> CommandRegistry:
    """Return a registry pre-populated with Aura's built-in commands.

    If ``agent`` is provided, also register one :class:`SkillCommand` per
    skill loaded by the Agent (user + project layers). The optional kwarg
    preserves the zero-arg call for callers that still build a registry
    without an Agent (e.g. legacy tests).
    """
    r = CommandRegistry()
    # HelpCommand needs the registry to enumerate commands at /help time.
    r.register(HelpCommand(registry=r))
    r.register(ExitCommand())
    r.register(ClearCommand())
    r.register(CompactCommand())
    r.register(ModelCommand())
    r.register(TasksCommand())
    if agent is not None:
        for skill in agent._skill_registry.list():
            r.register(SkillCommand(skill=skill, agent=agent))
        # MCP commands were collected at aconnect() time; register them
        # last so a name collision with a built-in / skill is flagged
        # rather than silently shadowed.
        for cmd in agent._mcp_commands:
            r.register(cmd)  # type: ignore[arg-type]
    return r


async def dispatch(
    line: str, agent: Agent, registry: CommandRegistry
) -> CommandResult:
    """Dispatch a REPL input line via the given registry."""
    return await registry.dispatch(line, agent)
