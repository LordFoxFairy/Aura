"""CommandRegistry — holds slash commands and dispatches input lines.

Thread-safety: not safe for concurrent ``register``/``unregister`` during
``dispatch``. The REPL is single-threaded, so this is fine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.core.commands.types import Command, CommandResult

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
