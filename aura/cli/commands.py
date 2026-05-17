"""Slash command façade for the REPL.

This import path remains stable for callers, but ownership of command
implementations and default-registry assembly now lives under
``aura.capabilities.commands``.
"""

from aura.capabilities.commands import Command, CommandResult
from aura.capabilities.commands.registry import (
    CommandRegistry,
    build_default_registry,
    dispatch,
)

__all__ = [
    "Command",
    "CommandRegistry",
    "CommandResult",
    "build_default_registry",
    "dispatch",
]
