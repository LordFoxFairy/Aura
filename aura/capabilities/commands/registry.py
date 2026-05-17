"""Command registry ownership for Aura's slash-command surface.

This module owns the command registry abstraction plus default-registry
assembly. Compatibility façades in ``aura.core.commands`` and
``aura.cli.commands`` re-export these symbols so legacy import paths stay
stable while ownership lives under ``aura.capabilities``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.capabilities.commands.types import Command, CommandResult

if TYPE_CHECKING:
    from aura.core.agent import Agent


class CommandRegistry:
    """Registry + dispatcher for slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, cmd: Command) -> None:
        """Add ``cmd`` to the registry.

        Raises:
            ValueError: if ``cmd.name`` is already registered.
        """
        if cmd.name in self._commands:
            raise ValueError(
                f"command {cmd.name!r} is already registered"
            )
        self._commands[cmd.name] = cmd

    def unregister(self, name: str) -> None:
        """Remove the command with the given ``name``.

        Idempotent — does nothing if ``name`` is not registered. This is
        important for MCP disconnects and Skill reloads where the caller
        may not know the current state.
        """
        self._commands.pop(name, None)

    def list(self) -> list[Command]:
        """Return all registered commands, sorted by name.

        Stable ordering keeps ``/help`` output deterministic across runs.
        Every element exposes the full :class:`Command` surface —
        ``name``, ``description``, ``source``, ``allowed_tools`` (possibly
        empty tuple), and ``argument_hint`` (possibly ``None``) — so
        callers (``/help``, future completion UI, permission layer) can
        render or inspect the metadata without a second registry hop.
        """
        return [self._commands[k] for k in sorted(self._commands)]

    async def dispatch(self, line: str, agent: Agent) -> CommandResult:
        """Route ``line`` to the matching command (or pass through).

        Contract:
        - Non-slash input (including empty/whitespace) → ``handled=False``,
          ``kind="noop"``. The REPL treats this as a normal prompt.
        - ``/unknown`` → ``handled=True``, ``kind="print"`` with a hint.
        - ``/known [arg...]`` → delegate to ``cmd.handle(arg, agent)``.
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
                handled=True,
                kind="print",
                text=f"unknown command: {name} (try /help)",
            )
        return await cmd.handle(arg, agent)


def build_default_registry(agent: Agent | None = None) -> CommandRegistry:
    """Return a registry pre-populated with Aura's built-in commands.

    If ``agent`` is provided, also register one SkillCommand per
    user-invocable skill loaded by the Agent and any MCP slash commands
    harvested during ``Agent.aconnect()``. The optional ``agent`` keeps the
    legacy zero-arg build path intact for tests and callers that only need
    the static builtin set.
    """
    from aura.capabilities.commands.buddy import BuddyCommand
    from aura.capabilities.commands.builtin import (
        ClearCommand,
        CompactCommand,
        ContextCommand,
        ExitCommand,
        HelpCommand,
        ModelCommand,
        ResumeCommand,
    )
    from aura.capabilities.commands.export import ExportCommand
    from aura.capabilities.commands.git import (
        GitDiffCommand,
        GitLogCommand,
        GitStatusCommand,
    )
    from aura.capabilities.commands.mcp import MCPCommand
    from aura.capabilities.commands.stats import StatsCommand
    from aura.capabilities.commands.tasks import (
        TaskGetCommand,
        TasksCommand,
        TaskStopCommand,
    )
    from aura.capabilities.commands.team import TeamCommand
    from aura.capabilities.skills_runtime.command import SkillCommand

    registry = CommandRegistry()
    registry.register(HelpCommand(registry=registry))
    registry.register(ExitCommand())
    registry.register(ClearCommand())
    registry.register(CompactCommand())
    registry.register(ContextCommand())
    registry.register(ModelCommand())
    registry.register(ExportCommand())
    registry.register(StatsCommand())
    registry.register(TasksCommand())
    registry.register(TaskGetCommand())
    registry.register(TaskStopCommand())
    registry.register(GitStatusCommand())
    registry.register(GitDiffCommand())
    registry.register(GitLogCommand())
    registry.register(MCPCommand())
    registry.register(BuddyCommand())
    registry.register(ResumeCommand())
    if agent is not None and agent._config.teams.enabled:
        registry.register(TeamCommand())
    if agent is not None:
        for skill in agent._skill_registry.user_invocable():
            registry.register(SkillCommand(skill=skill, agent=agent))
        for cmd in agent._mcp_commands:
            registry.register(cmd)  # type: ignore[arg-type]
    return registry


async def dispatch(
    line: str, agent: Agent, registry: CommandRegistry
) -> CommandResult:
    """Dispatch a REPL input line via the given registry."""
    return await registry.dispatch(line, agent)


__all__ = ["CommandRegistry", "build_default_registry", "dispatch"]
