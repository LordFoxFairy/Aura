"""Compatibility façade for Aura's slash-command abstraction.

Ownership now lives under :mod:`aura.capabilities.commands`; this package
re-exports the public command protocol and registry surface so historical
imports from ``aura.core.commands`` remain stable.
"""

from __future__ import annotations

from aura.capabilities.commands.registry import CommandRegistry
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
