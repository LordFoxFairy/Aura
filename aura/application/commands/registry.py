"""Command registry + default-registry assembly for Aura's slash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.application.commands.types import Command, CommandResult

if TYPE_CHECKING:
    from aura.core.agent import Agent


class CommandRegistry:
    """Registry + dispatcher for slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, cmd: Command) -> None:
        """Add ``cmd``; raises ``ValueError`` on duplicate name."""
        if cmd.name in self._commands:
            raise ValueError(f"command {cmd.name!r} is already registered")
        self._commands[cmd.name] = cmd

    def unregister(self, name: str) -> None:
        """Remove ``name`` if present (idempotent — MCP/Skill reload safe)."""
        self._commands.pop(name, None)

    def list(self) -> list[Command]:
        """Return all commands sorted by name (deterministic ``/help``)."""
        return [self._commands[k] for k in sorted(self._commands)]

    async def dispatch(self, line: str, agent: Agent) -> CommandResult:
        """Route ``line`` to the matching command, or pass through as prompt.

        Non-slash input → ``handled=False`` (REPL treats as a prompt).
        Unknown ``/name`` → ``handled=True`` with a help hint.
        """
        stripped = line.strip()
        if not stripped.startswith("/"):
            return CommandResult(handled=False, kind="noop", text="")
        parts = stripped.split(None, 1)
        name = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        cmd = self._commands.get(name)
        if cmd is None:
            return CommandResult(
                handled=True, kind="print",
                text=f"unknown command: {name} (try /help)",
            )
        return await cmd.handle(arg, agent)


def build_default_registry(agent: Agent | None = None) -> CommandRegistry:
    """Pre-populate a registry with Aura's built-in commands.

    When ``agent`` is provided, also registers one ``SkillCommand`` per
    user-invocable skill and every MCP slash command harvested during
    ``Agent.aconnect()``. Zero-arg form returns the static builtin set —
    used by tests and callers that don't need the dynamic surfaces.
    """
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
    from aura.application.commands.git import (
        GitDiffCommand,
        GitLogCommand,
        GitStatusCommand,
    )
    from aura.application.commands.mcp import MCPCommand
    from aura.application.commands.stats import StatsCommand
    from aura.application.commands.tasks import (
        TaskGetCommand,
        TasksCommand,
        TaskStopCommand,
    )
    from aura.application.commands.team import TeamCommand
    from aura.infrastructure.skills.command import SkillCommand

    registry = CommandRegistry()
    registry.register(HelpCommand(registry=registry))
    for cmd in (
        ExitCommand(), ClearCommand(), CompactCommand(), ContextCommand(),
        ModelCommand(), ExportCommand(), StatsCommand(),
        TasksCommand(), TaskGetCommand(), TaskStopCommand(),
        GitStatusCommand(), GitDiffCommand(), GitLogCommand(),
        MCPCommand(), ResumeCommand(),
    ):
        registry.register(cmd)
    if agent is None:
        return registry
    if agent.config.teams.enabled:
        registry.register(TeamCommand())
    for skill in agent._skill_registry.user_invocable():
        registry.register(SkillCommand(skill=skill, agent=agent))
    for mcp_cmd in agent._mcp_commands:
        registry.register(mcp_cmd)  # type: ignore[arg-type]  # deliberately off-type arg to exercise path
    return registry


async def dispatch(
    line: str, agent: Agent, registry: CommandRegistry,
) -> CommandResult:
    """Dispatch a REPL input line via the given registry."""
    return await registry.dispatch(line, agent)
