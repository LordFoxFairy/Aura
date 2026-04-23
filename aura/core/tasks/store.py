"""TasksStore — in-memory keyed store for :class:`TaskRecord`.

Single-process, single-event-loop. No cross-process sharing (subagents
always run inside the parent's event loop), so a plain dict is sufficient.
If that changes we'll revisit with a proper concurrency primitive — until
then keeping the surface minimal avoids premature complexity.
"""

from __future__ import annotations

import time
import uuid

from langchain_core.messages import BaseMessage

from aura.core.tasks.types import (
    _SHELL_RECENT_ACTIVITIES_CAP,
    TaskKind,
    TaskRecord,
    TaskStatus,
    _append_recent,
)


class TasksStore:
    def __init__(self) -> None:
        self._records: dict[str, TaskRecord] = {}

    def create(
        self,
        description: str,
        prompt: str,
        parent_id: str | None = None,
        *,
        kind: TaskKind = "subagent",
        metadata: dict[str, object] | None = None,
    ) -> TaskRecord:
        task_id = uuid.uuid4().hex
        rec = TaskRecord(
            id=task_id,
            parent_id=parent_id,
            description=description,
            prompt=prompt,
            kind=kind,
            metadata=dict(metadata) if metadata is not None else {},
        )
        self._records[task_id] = rec
        return rec

    def get(self, task_id: str) -> TaskRecord | None:
        return self._records.get(task_id)

    def list(
        self,
        *,
        status: TaskStatus | None = None,
        kind: TaskKind | None = None,
        limit: int | None = None,
    ) -> list[TaskRecord]:
        records = list(self._records.values())
        if status is not None:
            records = [r for r in records if r.status == status]
        if kind is not None:
            records = [r for r in records if r.kind == kind]
        if limit is not None:
            # Newest-first slice — matches /tasks ordering so callers
            # sharing this list (e.g. task_list tool) don't have to resort.
            records = sorted(records, key=lambda r: -r.started_at)[:limit]
        return records

    def record_activity(self, task_id: str, activity: str) -> None:
        """Note a child-agent tool event on the task's progress record.

        Called from :func:`run_task` as the subagent streams
        ``ToolCallStarted`` events. Tight contract: bump ``tool_count``,
        push onto the bounded ``recent_activities`` ring, update
        ``last_activity_at``. No-op for unknown task_ids so a racing
        cancel can't KeyError here.
        """
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.progress.tool_count += 1
        rec.progress.last_activity_at = time.time()
        _append_recent(rec.progress, activity)

    def record_shell_line(self, task_id: str, line: str) -> None:
        """Append a shell output line to the task's progress ring.

        Bumps ``line_count`` + ``last_activity_at``; bounds
        ``recent_activities`` at :data:`_SHELL_RECENT_ACTIVITIES_CAP`.
        No-op for unknown ids (matches ``record_activity`` semantics).
        """
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.progress.line_count += 1
        rec.progress.last_activity_at = time.time()
        _append_recent(rec.progress, line, cap=_SHELL_RECENT_ACTIVITIES_CAP)

    def record_shell_marker(self, task_id: str, marker: str) -> None:
        """Append a non-counting marker (e.g. ``"[stalled?]"``) to the ring.

        Unlike :meth:`record_shell_line`, this does NOT bump ``line_count``
        or ``last_activity_at`` — stall detection uses ``last_activity_at``
        to decide when to fire, and bumping it would defeat the detector.
        """
        rec = self._records.get(task_id)
        if rec is None:
            return
        _append_recent(rec.progress, marker, cap=_SHELL_RECENT_ACTIVITIES_CAP)

    def append_message(self, task_id: str, msg: BaseMessage) -> None:
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.messages.append(msg)

    def mark_completed(self, task_id: str, result: str) -> None:
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.status = "completed"
        rec.final_result = result
        rec.finished_at = time.time()

    def mark_failed(self, task_id: str, error: str) -> None:
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.status = "failed"
        rec.error = error
        rec.finished_at = time.time()

    def mark_cancelled(self, task_id: str) -> None:
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.status = "cancelled"
        rec.finished_at = time.time()
