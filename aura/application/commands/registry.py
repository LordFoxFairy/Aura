"""Slash-command registry and REPL dispatch helper."""

from __future__ import annotations

from aura.application.commands.types import Command, CommandResult
from aura.application.session import AgentSession


class CommandRegistry:
    """Registry + dispatcher for slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, Command[AgentSession]] = {}

    def register(self, cmd: Command[AgentSession]) -> None:
        """Add ``cmd``; raises ``ValueError`` on duplicate name."""
        if cmd.name in self._commands:
            raise ValueError(f"command {cmd.name!r} is already registered")
        self._commands[cmd.name] = cmd

    def unregister(self, name: str) -> None:
        """Remove ``name`` if present (idempotent — MCP/Skill reload safe)."""
        self._commands.pop(name, None)

    def list(self) -> list[Command[AgentSession]]:
        """Return all commands sorted by name (deterministic ``/help``)."""
        return [self._commands[k] for k in sorted(self._commands)]

    async def dispatch(self, line: str, agent: AgentSession) -> CommandResult:
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


async def dispatch(
    line: str, agent: AgentSession, registry: CommandRegistry,
) -> CommandResult:
    """Dispatch a REPL input line via the given registry."""
    return await registry.dispatch(line, agent)
