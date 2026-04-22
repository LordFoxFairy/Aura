"""Subagent dispatch — fire-and-forget TaskCreate / TaskOutput primitives.

Phase E (0.5.0) wiring. The ``task_create`` tool mirrors claude-code's
``shouldDefer: true`` semantics — it returns a ``task_id`` the moment the
subagent is scheduled, without awaiting completion. The subagent is an
independent :class:`aura.core.agent.Agent` instance with its own
:class:`LoopState` and :class:`Context` (no shared mutable state with the
parent), but shares the parent's LLM router alias so the same model handles
both levels.

Cancellation is cooperative: the parent Agent keeps a ``task_id ->
asyncio.Task`` map, and ``Agent.close()`` cancels the running handles.
``run_task`` turns ``CancelledError`` into ``status=cancelled`` on the
record before re-raising.
"""

from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from aura.core.tasks.types import TaskRecord, TaskStatus

__all__ = [
    "SubagentFactory",
    "TaskRecord",
    "TaskStatus",
    "TasksStore",
    "run_task",
]
