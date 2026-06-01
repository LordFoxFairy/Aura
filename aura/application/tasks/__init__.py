"""Fire-and-forget subagent dispatch primitives."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "LocalAgentTask": "aura.application.tasks.runners",
    "SubagentSpawner": "aura.application.tasks.spawn",
    "TaskRecord": "aura.domain.task",
    "TaskStatus": "aura.domain.task",
    "TasksStore": "aura.application.tasks.store",
    "run_task": "aura.application.tasks.run",
}

__all__ = list(_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]  # lazy-load: __getattr__ resolves each name on demand


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return vars(import_module(module_name))[name]
