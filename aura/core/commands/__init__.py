"""Unified slash-command abstraction.

Exports:
    Command       — structural Protocol for command objects.
    CommandRegistry — holds commands, routes input lines.
    CommandResult — dispatch outcome (handled/kind/text).
    CommandKind   — Literal["print", "exit", "noop"].
    CommandSource — Literal["builtin", "skill", "mcp"].

Built-in commands live in :mod:`aura.core.commands.builtin`.
"""

from __future__ import annotations

from aura.core.commands.registry import CommandRegistry
from aura.core.commands.types import (
    Command,
    CommandKind,
    CommandResult,
    CommandSource,
)

__all__ = [
    "Command",
    "CommandKind",
    "CommandRegistry",
    "CommandResult",
    "CommandSource",
]
