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
TaskKind = Literal["subagent", "shell"]

# Keep the subagent rolling window tight — this gets serialised into
# task_get output and the child could fire hundreds of tool calls in a
# long subagent run. 5 is enough to show what the agent is currently
# doing without flooding the parent's context.
_RECENT_ACTIVITIES_CAP = 5
# Shell tasks emit line-by-line output (stdout + stderr interleaved), which
# is much chattier than tool names — 20 keeps useful tail without drowning
# task_get output.
_SHELL_RECENT_ACTIVITIES_CAP = 20


@dataclass
class TaskProgress:
    """Rolling snapshot of a running task's activity.

    ``recent_activities`` is a bounded list; older entries are dropped as
    new ones arrive. For subagents we cap at 5 tool names; for shell
    tasks we cap at 20 output lines (``[out] ``/``[err] `` prefixed).
    ``token_count`` is left at 0 today — we don't have per-subagent usage
    reporting wired through the child loop yet. ``line_count`` is the
    monotonic total for shell tasks (analogous to ``tool_count`` for
    subagents); stays at 0 for subagents. ``last_activity_at`` is the
    wall-clock timestamp of the most recent increment.
    """

    tool_count: int = 0
    token_count: int = 0
    line_count: int = 0
    last_activity_at: float | None = None
    recent_activities: list[str] = field(default_factory=list)


@dataclass
class TaskRecord:
    id: str
    parent_id: str | None
    description: str
    prompt: str
    status: TaskStatus = "running"
    kind: TaskKind = "subagent"
    # Orthogonal to ``kind``: ``kind`` distinguishes subagent-vs-shell, while
    # ``agent_type`` selects the flavor of subagent (general-purpose / explore
    # / verify / plan). Only meaningful when ``kind == "subagent"``; shell
    # tasks leave it ``None``. Stored as a free-form ``str`` (not the
    # Literal) so the store layer doesn't have to re-import the registry
    # just to persist a known-valid name.
    agent_type: str | None = None
    messages: list[BaseMessage] = field(default_factory=list)
    final_result: str | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    progress: TaskProgress = field(default_factory=TaskProgress)
    metadata: dict[str, object] = field(default_factory=dict)


def _append_recent(
    progress: TaskProgress,
    activity: str,
    *,
    cap: int = _RECENT_ACTIVITIES_CAP,
) -> None:
    """Append ``activity`` to ``progress.recent_activities``, bounded by ``cap``.

    Module-private helper shared by :class:`TasksStore`, :func:`run_task`,
    and the ``bash_background`` tool so the ring-buffer contract lives in
    one place. ``cap`` defaults to 5 (subagent default); shell tasks pass
    ``_SHELL_RECENT_ACTIVITIES_CAP`` (20).
    """
    progress.recent_activities.append(activity)
    if len(progress.recent_activities) > cap:
        del progress.recent_activities[:-cap]
