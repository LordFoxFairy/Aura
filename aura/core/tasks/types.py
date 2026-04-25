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
from pathlib import Path
from typing import Literal

from langchain_core.messages import BaseMessage

TaskStatus = Literal["running", "completed", "failed", "cancelled"]
# Round 6L extended to cover ``teammate`` — a long-lived team member runtime
# whose lifecycle TeamManager owns. Distinct from a one-shot subagent +
# distinct from a shell task; same TaskRecord shape, different orchestrator.
TaskKind = Literal["subagent", "shell", "teammate"]

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

    Round 7QS — token tracking. ``input_tokens`` / ``output_tokens`` are
    cumulative-since-spawn counters fed by the post_model hook installed
    on the child Agent inside :func:`run_task`. ``token_count`` is the
    total (input + output) — kept as a derived alias because callers
    want one display number and we don't want every site to recompute
    the sum.

    Round 7QS — periodic summary. ``latest_summary`` is overwritten on
    every :class:`AgentSummarizer` tick; ``summary_updated_at`` is the
    wall-clock timestamp of the most recent overwrite. Both stay
    ``None`` when summarization is disabled (interval=0).

    ``line_count`` is the monotonic total for shell tasks (analogous to
    ``tool_count`` for subagents); stays at 0 for subagents.
    ``last_activity_at`` is the wall-clock timestamp of the most recent
    increment.
    """

    tool_count: int = 0
    token_count: int = 0
    line_count: int = 0
    last_activity_at: float | None = None
    recent_activities: list[str] = field(default_factory=list)
    # Round 7QS — separate input / output buckets so the LLM-facing
    # task_get can show the breakdown the way provider dashboards do.
    input_tokens: int = 0
    output_tokens: int = 0
    # Round 7QS — last summary text written by the AgentSummarizer.
    latest_summary: str | None = None
    summary_updated_at: float | None = None


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
    # Round 4F — JSONL transcript path on parent's storage root. Set by
    # :func:`run_task` after a successful flush; ``None`` for tasks that
    # ran without a transcript_storage (legacy callers, in-memory tests).
    transcript_path: Path | None = None
    # Round 7R — the model spec the child actually ran on. Pinned at
    # task_create time (override or inherited parent spec). Empty
    # string for legacy callers that didn't route through factory's
    # spec-aware spawn.
    model_spec: str = ""


@dataclass
class TaskNotification:
    """Round 4F — terminal-transition push from a child to the parent.

    Producers: :class:`TasksStore` listeners (registered by Agent.__init__).
    Consumers: :class:`Context.build` drains via Agent's
    ``_drain_task_notifications``. The dataclass is deliberately small —
    everything the parent's prompt envelope needs to render a
    ``<task-notification>`` block without re-fetching the record.

    ``description`` echoes ``TaskRecord.description`` so the parent's
    notification block can reference the human-readable name; ``summary``
    carries the final_result on success or the error message on failure;
    ``exit_code`` is the shell-task return code (always ``None`` for
    subagents).
    """

    task_id: str
    status: TaskStatus
    summary: str | None
    description: str = ""
    exit_code: int | None = None


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
