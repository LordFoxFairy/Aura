"""Capabilities-owned slash command surface.

This package owns Aura's builtin slash-command implementations and registry,
but keeps package import side effects minimal so core compatibility façades can
import specific submodules without triggering the whole command graph.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "ClearCommand": "aura.capabilities.commands.builtin",
    "Command": "aura.capabilities.commands.types",
    "CommandKind": "aura.capabilities.commands.types",
    "CommandRegistry": "aura.capabilities.commands.registry",
    "CommandResult": "aura.capabilities.commands.types",
    "CommandSource": "aura.capabilities.commands.types",
    "CompactCommand": "aura.capabilities.commands.builtin",
    "ContextCommand": "aura.capabilities.commands.builtin",
    "ExitCommand": "aura.capabilities.commands.builtin",
    "GitDiffCommand": "aura.capabilities.commands.git",
    "GitLogCommand": "aura.capabilities.commands.git",
    "GitStatusCommand": "aura.capabilities.commands.git",
    "HelpCommand": "aura.capabilities.commands.builtin",
    "BuddyCommand": "aura.capabilities.commands.buddy",
    "ExportCommand": "aura.capabilities.commands.export",
    "MCPCommand": "aura.capabilities.commands.mcp",
    "ModelCommand": "aura.capabilities.commands.builtin",
    "ResumeCommand": "aura.capabilities.commands.builtin",
    "StatsCommand": "aura.capabilities.commands.stats",
    "TaskGetCommand": "aura.capabilities.commands.tasks",
    "TaskStopCommand": "aura.capabilities.commands.tasks",
    "TasksCommand": "aura.capabilities.commands.tasks",
    "TeamCommand": "aura.capabilities.commands.team",
    "build_default_registry": "aura.capabilities.commands.registry",
    "dispatch": "aura.capabilities.commands.registry",
    "format_relative_time": "aura.capabilities.commands.builtin",
    "restore_session_into": "aura.capabilities.commands.builtin",
    "session_label": "aura.capabilities.commands.builtin",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
