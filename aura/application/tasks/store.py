"""In-memory keyed store for :class:`TaskRecord`."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import BaseMessage

from aura.domain.task import (
    SHELL_RECENT_ACTIVITIES_CAP,
    TaskKind,
    TaskRecord,
    TaskStatus,
    append_recent,
)
from aura.infrastructure.persistence import journal


class _LazyEvent:
    """Event proxy that defers loop binding until first ``wait`` / ``set``."""

    __slots__ = ("_event", "_pre_set")

    def __init__(self) -> None:
        self._event: asyncio.Event | None = None
        self._pre_set: bool = False

    def _ensure(self) -> asyncio.Event:
        if self._event is None:
            self._event = asyncio.Event()
            if self._pre_set:
                self._event.set()
        return self._event

    def set(self) -> None:
        if self._event is None:
            self._pre_set = True
            return
        self._event.set()

    def is_set(self) -> bool:
        if self._event is None:
            return self._pre_set
        return self._event.is_set()

    async def wait(self) -> None:
        await self._ensure().wait()


# Sync on purpose: fired from ``mark_*`` which may run outside an event loop.
TerminalListener = Callable[[TaskRecord], None]
StartedListener = Callable[[TaskRecord], None]
ActivityListener = Callable[[TaskRecord, str], None]


class TasksStore:
    def __init__(self) -> None:
        self._records: dict[str, TaskRecord] = {}
        self._terminal_events: dict[str, _LazyEvent] = {}
        self._terminal_listeners: list[TerminalListener] = []
        self._started_listeners: list[StartedListener] = []
        self._activity_listeners: list[ActivityListener] = []

    def create(
        self,
        description: str,
        prompt: str,
        *,
        kind: TaskKind = "subagent",
        agent_type: str | None = None,
        metadata: dict[str, object] | None = None,
        model_spec: str = "",
    ) -> TaskRecord:
        task_id = uuid.uuid4().hex
        rec = TaskRecord(
            id=task_id,
            description=description,
            prompt=prompt,
            kind=kind,
            agent_type=agent_type,
            metadata=dict(metadata) if metadata is not None else {},
            model_spec=model_spec,
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
            # Newest-first slice matches /tasks ordering.
            records = sorted(records, key=lambda r: -r.started_at)[:limit]
        return records

    def record_activity(self, task_id: str, activity: str) -> None:
        """Note a child-agent tool event on the task's progress record."""
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.progress.tool_count += 1
        rec.progress.last_activity_at = time.time()
        append_recent(rec.progress, activity)
        for listener in list(self._activity_listeners):
            try:
                listener(rec, activity)
            except Exception as exc:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
                journal.write(
                    "tasks_activity_listener_error",
                    task_id=rec.id,
                    error=f"{type(exc).__name__}: {exc}",
                )

    def record_started(self, task_id: str) -> None:
        """Fire ``started_listeners`` for a freshly-running task."""
        rec = self._records.get(task_id)
        if rec is None:
            return
        for listener in list(self._started_listeners):
            try:
                listener(rec)
            except Exception as exc:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
                journal.write(
                    "tasks_started_listener_error",
                    task_id=rec.id,
                    error=f"{type(exc).__name__}: {exc}",
                )

    def record_activity_note(self, task_id: str, activity: str) -> None:
        """Note non-tool activity; preserves the stricter ``tool_count`` semantics."""
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.progress.last_activity_at = time.time()
        append_recent(rec.progress, activity)

    def record_shell_line(self, task_id: str, line: str) -> None:
        """Append a shell output line; bumps ``line_count`` and bounds the ring."""
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.progress.line_count += 1
        rec.progress.last_activity_at = time.time()
        append_recent(rec.progress, line, cap=SHELL_RECENT_ACTIVITIES_CAP)

    def record_shell_marker(self, task_id: str, marker: str) -> None:
        """Append a marker without touching ``last_activity_at`` (stall detector reads it)."""
        rec = self._records.get(task_id)
        if rec is None:
            return
        append_recent(rec.progress, marker, cap=SHELL_RECENT_ACTIVITIES_CAP)

    def append_message(self, task_id: str, msg: BaseMessage) -> None:
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.messages.append(msg)

    def record_token_usage(
        self,
        task_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Add ``input_tokens`` / ``output_tokens`` to progress; clamps negatives to 0."""
        rec = self._records.get(task_id)
        if rec is None:
            return
        in_t = max(0, int(input_tokens))
        out_t = max(0, int(output_tokens))
        rec.progress.input_tokens += in_t
        rec.progress.output_tokens += out_t
        rec.progress.token_count = (
            rec.progress.input_tokens + rec.progress.output_tokens
        )

    def mark_observed(self, task_id: str) -> float | None:
        """Stamp when a terminal task's result is first observed; stable on repeated reads."""
        rec = self._records.get(task_id)
        if rec is None or rec.status == "running":
            return None
        if rec.observed_at is None:
            rec.observed_at = time.time()
        return rec.observed_at

    def set_transcript_path(self, task_id: str, path: Path) -> None:
        """Pin the on-disk JSONL transcript location on the record."""
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.transcript_path = path

    def terminal_event(self, task_id: str) -> _LazyEvent:
        """Per-task lazy terminal event; pre-set if the record is already terminal."""
        ev = self._terminal_events.get(task_id)
        if ev is None:
            ev = _LazyEvent()
            self._terminal_events[task_id] = ev
            rec = self._records.get(task_id)
            if rec is not None and rec.status != "running":
                ev.set()
        return ev

    def add_terminal_listener(self, callback: TerminalListener) -> None:
        """Register a sync callback fired AFTER the record's terminal fields are set."""
        self._terminal_listeners.append(callback)

    def remove_terminal_listener(self, callback: TerminalListener) -> None:
        try:
            self._terminal_listeners.remove(callback)
        except ValueError:
            return

    def add_started_listener(self, callback: StartedListener) -> None:
        self._started_listeners.append(callback)

    def remove_started_listener(self, callback: StartedListener) -> None:
        try:
            self._started_listeners.remove(callback)
        except ValueError:
            return

    def add_activity_listener(self, callback: ActivityListener) -> None:
        self._activity_listeners.append(callback)

    def remove_activity_listener(self, callback: ActivityListener) -> None:
        try:
            self._activity_listeners.remove(callback)
        except ValueError:
            return

    def _fire_terminal(self, rec: TaskRecord) -> None:
        """Trigger the terminal event and dispatch listeners; exceptions journaled."""
        ev = self._terminal_events.get(rec.id)
        if ev is not None:
            ev.set()
        else:
            # Pre-set so a later consumer asking after the terminal mark short-circuits.
            placeholder = _LazyEvent()
            placeholder.set()
            self._terminal_events[rec.id] = placeholder
        for listener in list(self._terminal_listeners):
            try:
                listener(rec)
            except Exception as exc:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
                journal.write(
                    "tasks_terminal_listener_error",
                    task_id=rec.id,
                    error=f"{type(exc).__name__}: {exc}",
                )

    def _terminal_record(self, task_id: str) -> TaskRecord | None:
        rec = self._records.get(task_id)
        if rec is None or rec.status != "running":
            return None
        return rec

    def mark_completed(self, task_id: str, result: str) -> None:
        rec = self._terminal_record(task_id)
        if rec is None:
            return
        rec.status = "completed"
        rec.final_result = result
        rec.finished_at = time.time()
        self._fire_terminal(rec)

    def mark_failed(self, task_id: str, error: str) -> None:
        rec = self._terminal_record(task_id)
        if rec is None:
            return
        rec.status = "failed"
        rec.error = error
        rec.finished_at = time.time()
        self._fire_terminal(rec)

    def mark_cancelled(self, task_id: str) -> None:
        rec = self._terminal_record(task_id)
        if rec is None:
            return
        rec.status = "cancelled"
        rec.finished_at = time.time()
        self._fire_terminal(rec)


__all__ = [
    "ActivityListener",
    "StartedListener",
    "TasksStore",
    "TerminalListener",
]
