"""Subagent dispatch — fire-and-forget TaskCreate / TaskOutput primitives.

The ``task_create`` tool returns a ``task_id`` the moment the subagent is
scheduled, without awaiting completion. The subagent is an independent
:class:`aura.core.agent.Agent` with its own :class:`LoopState` and
:class:`Context` (no shared mutable state with the parent), but inherits
the parent's LLM router alias.

Cancellation is cooperative: the parent Agent keeps a ``task_id ->
asyncio.Task`` map, and ``Agent.close()`` cancels the running handles.
``run_task`` turns ``CancelledError`` into ``status=cancelled`` on the
record before re-raising.
"""

from aura.application.tasks.factory import SubagentFactory
from aura.application.tasks.run import run_task
from aura.application.tasks.runners import LocalAgentTask
from aura.application.tasks.store import TasksStore
from aura.domain.task import TaskRecord, TaskStatus

__all__ = [
    "LocalAgentTask",
    "SubagentFactory",
    "TaskRecord",
    "TaskStatus",
    "TasksStore",
    "run_task",
]
