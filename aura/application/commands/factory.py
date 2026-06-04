"""Default command-registry assembly; keeps CommandRegistry free of concrete-command deps."""

from aura.application.commands.builtin import (
    ClearCommand,
    CompactCommand,
    ContextCommand,
    ExitCommand,
    HelpCommand,
    ModelCommand,
    ResumeCommand,
)
from aura.application.commands.export import ExportCommand
from aura.application.commands.git import GitDiffCommand, GitLogCommand, GitStatusCommand
from aura.application.commands.mcp import MCPCommand
from aura.application.commands.registry import CommandRegistry
from aura.application.commands.stats import StatsCommand
from aura.application.commands.tasks import TaskGetCommand, TasksCommand, TaskStopCommand
from aura.application.commands.team import TeamCommand
from aura.application.commands.types import Command
from aura.application.session import AgentSession
from aura.infrastructure.skills.command import SkillCommand


def build_default_registry(agent: AgentSession | None = None) -> CommandRegistry:
    """Pre-populate built-in commands; with an agent, also the dynamic skill/MCP surfaces."""
    registry = CommandRegistry()
    registry.register(HelpCommand(registry=registry))
    for cmd in (
        ExitCommand(),
        ClearCommand(),
        CompactCommand(),
        ContextCommand(),
        ModelCommand(),
        ExportCommand(),
        StatsCommand(),
        TasksCommand(),
        TaskGetCommand(),
        TaskStopCommand(),
        GitStatusCommand(),
        GitDiffCommand(),
        GitLogCommand(),
        MCPCommand(),
        ResumeCommand(),
    ):
        registry.register(cmd)
    if agent is None:
        return registry
    if agent.config.teams.enabled:
        registry.register(TeamCommand())
    for skill in agent.skill_registry.user_invocable():
        registry.register(SkillCommand(skill=skill, agent=agent))
    for mcp_cmd in agent.mcp_commands:
        if isinstance(mcp_cmd, Command):
            registry.register(mcp_cmd)
    return registry
