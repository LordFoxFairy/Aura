"""Application-layer slash command surface."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "ClearCommand": "aura.application.commands.builtin",
    "Command": "aura.application.commands.types",
    "CommandKind": "aura.application.commands.types",
    "CommandRegistry": "aura.application.commands.registry",
    "CommandResult": "aura.application.commands.types",
    "CommandSource": "aura.application.commands.types",
    "CompactCommand": "aura.application.commands.builtin",
    "ContextCommand": "aura.application.commands.builtin",
    "ExitCommand": "aura.application.commands.builtin",
    "GitDiffCommand": "aura.application.commands.git",
    "GitLogCommand": "aura.application.commands.git",
    "GitStatusCommand": "aura.application.commands.git",
    "HelpCommand": "aura.application.commands.builtin",
    "ExportCommand": "aura.application.commands.export",
    "MCPCommand": "aura.application.commands.mcp",
    "ModelCommand": "aura.application.commands.builtin",
    "ResumeCommand": "aura.application.commands.builtin",
    "StatsCommand": "aura.application.commands.stats",
    "TaskGetCommand": "aura.application.commands.tasks",
    "TaskStopCommand": "aura.application.commands.tasks",
    "TasksCommand": "aura.application.commands.tasks",
    "TeamCommand": "aura.application.commands.team",
    "build_default_registry": "aura.application.commands.factory",
    "dispatch": "aura.application.commands.registry",
    "format_relative_time": "aura.application.commands.builtin",
    "session_label": "aura.application.commands.builtin",
}

__all__ = list(_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]  # lazy-load: __getattr__ resolves each name on demand from _EXPORTS


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return vars(import_module(module_name))[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
