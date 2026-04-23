"""TaskRecord — the one dataclass persisted per subagent invocation.

Mutation rules (enforced by :class:`TasksStore`, not by the dataclass
itself):

- ``status``, ``final_result``, ``error``, ``finished_at`` are set exactly
  once, when a terminal transition is taken (completed / failed / cancelled).
- ``messages`` is append-only during the ``running`` window — the subagent's
  transcript accretes, never rewrites.
- ``progress`` is mutated in place while the subagent is ``running`` — it
  holds a rolling snapshot of child tool activity so ``task_get`` can
  surface "what is the child doing right now" without replaying the whole
  transcript. The progress dataclass is deliberately mutable (``recent_activities``
  is a bounded ring) and not frozen; treat it as owned by the store.
- Anything else (``description``, ``prompt``, ``parent_id``, ``started_at``,
  ``id``) is write-at-create and read-only thereafter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import BaseMessage

TaskStatus = Literal["running", "completed", "failed", "cancelled"]

# Keep the rolling window tight — this gets serialised into task_get output
# and the child could fire hundreds of tool calls in a long subagent run.
# 5 is enough to show what the agent is currently doing without flooding
# the parent's context.
_RECENT_ACTIVITIES_CAP = 5


@dataclass
class TaskProgress:
    """Rolling snapshot of a running subagent's tool activity.

    ``recent_activities`` is a bounded list (last 5 tool names); older
    entries are dropped as new ones arrive. ``token_count`` is left at 0
    today — we don't have per-subagent usage reporting wired through the
    child loop yet, but the field is here so wiring it later doesn't
    break the task_get contract. ``last_activity_at`` is the wall-clock
    timestamp of the most recent increment (``time.time()`` seconds).
    """

    tool_count: int = 0
    token_count: int = 0
    last_activity_at: float | None = None
    recent_activities: list[str] = field(default_factory=list)


@dataclass
class TaskRecord:
    id: str
    parent_id: str | None
    description: str
    prompt: str
    status: TaskStatus = "running"
    messages: list[BaseMessage] = field(default_factory=list)
    final_result: str | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    progress: TaskProgress = field(default_factory=TaskProgress)


def _append_recent(progress: TaskProgress, activity: str) -> None:
    """Append ``activity`` to ``progress.recent_activities``, capping at 5.

    Module-private helper shared by :class:`TasksStore` and :func:`run_task`
    so the ring-buffer contract lives in one place.
    """
    progress.recent_activities.append(activity)
    if len(progress.recent_activities) > _RECENT_ACTIVITIES_CAP:
        del progress.recent_activities[: -_RECENT_ACTIVITIES_CAP]
