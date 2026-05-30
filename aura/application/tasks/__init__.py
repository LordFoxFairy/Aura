"""Fire-and-forget subagent dispatch primitives."""

from aura.application.tasks.run import run_task
from aura.application.tasks.runners import LocalAgentTask
from aura.application.tasks.spawn import SubagentSpawner
from aura.application.tasks.store import TasksStore
from aura.domain.task import TaskRecord, TaskStatus

__all__ = [
    "LocalAgentTask",
    "SubagentSpawner",
    "TaskRecord",
    "TaskStatus",
    "TasksStore",
    "run_task",
]
