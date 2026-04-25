"""TasksStore — in-memory keyed store for :class:`TaskRecord`.

Single-process, single-event-loop. No cross-process sharing (subagents
always run inside the parent's event loop), so a plain dict is sufficient.

Round 4F adds two cross-cutting bits:

- **Lazy terminal events**: per-task :class:`asyncio.Event` allocated on
  first :meth:`terminal_event` lookup. ``task_output(wait=True)`` parks
  on these so a parent can block on a child's terminal transition
  without polling. ``_LazyEvent`` defers Event construction until we're
  inside a running event loop — constructing one at ``store.create``
  time would bind the Event to the wrong loop in tests that recycle
  loops between cases.

- **Terminal listener registry**: callables fired AFTER a record reaches
  a terminal state. Agent.__init__ registers a listener that pushes a
  :class:`TaskNotification` onto its ``_pending_notifications`` queue
  so the next :meth:`Context.build` can render it as a
  ``<task-notification>`` block.

Round 7QS adds:

- :meth:`record_token_usage` — accumulates per-invoke usage_metadata
  from the child's post_model hook.
- :meth:`update_summary` — overwrites the rolling summary text the
  AgentSummarizer writes every interval.

Round 4F transcript wiring:

- :meth:`set_transcript_path` — ``run_task`` calls this after the JSONL
  flush succeeds so ``task_get`` can surface the on-disk location.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import BaseMessage

from aura.core.persistence import journal
from aura.core.tasks.types import (
    _SHELL_RECENT_ACTIVITIES_CAP,
    TaskKind,
    TaskNotification,
    TaskRecord,
    TaskStatus,
    _append_recent,
)


class _LazyEvent:
    """asyncio.Event proxy that defers loop binding until first use.

    ``asyncio.Event`` ties itself to the event loop active at construction
    time. Building one in ``store.create`` (which usually runs synchronously
    on import / fixture setup) would bind it to a loop that may not be
    the same loop as the eventual ``await event.wait()`` consumer. We
    sidestep by building the real Event lazily on the first ``.wait()`` /
    ``.set()`` call and pre-marking ``_pre_set=True`` for tasks that
    became terminal before any consumer parked on the event (race fix:
    a parent calling ``task_output(wait=True)`` after the child already
    completed must short-circuit instead of hanging forever on a
    fresh, never-set Event).
    """

    __slots__ = ("_event", "_pre_set")

    def __init__(self) -> None:
        self._event: asyncio.Event | None = None
        # ``_pre_set`` carries the "should this Event already be set when
        # we lazily build it" sentinel. Flipped by :meth:`set` when no
        # real Event has been built yet; honored by :meth:`_ensure`.
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


# A terminal listener gets the full TaskRecord so it can read every
# field it needs (description, status, final_result, error, ...). Sync
# callable on purpose — listeners run inside ``mark_*`` methods which
# may be called from outside an event loop in tests; making the
# contract sync keeps every call site compatible.
TerminalListener = Callable[[TaskRecord], None]


class TasksStore:
    def __init__(self) -> None:
        self._records: dict[str, TaskRecord] = {}
        # Lazy per-task terminal events for ``task_output(wait=True)``.
        self._terminal_events: dict[str, _LazyEvent] = {}
        # Terminal listeners — fired AFTER a record reaches terminal
        # state. Each listener gets the full TaskRecord; exceptions are
        # caught + journaled so a buggy listener can't strand the
        # mark_* call.
        self._terminal_listeners: list[TerminalListener] = []

    def create(
        self,
        description: str,
        prompt: str,
        parent_id: str | None = None,
        *,
        kind: TaskKind = "subagent",
        agent_type: str | None = None,
        metadata: dict[str, object] | None = None,
        model_spec: str = "",
    ) -> TaskRecord:
        task_id = uuid.uuid4().hex
        rec = TaskRecord(
            id=task_id,
            parent_id=parent_id,
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

    # ------------------------------------------------------------------
    # Round 7QS — token usage + summary
    # ------------------------------------------------------------------

    def record_token_usage(
        self,
        task_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Add ``input_tokens`` / ``output_tokens`` to the task's progress.

        Negative values clamp to 0 (defensive — a malformed usage payload
        from a misconfigured provider must NOT subtract the running
        total). No-op for unknown task_ids so a racing cancel cleanup
        between the post_model hook firing and the record being
        looked up doesn't KeyError.
        """
        rec = self._records.get(task_id)
        if rec is None:
            return
        in_t = max(0, int(input_tokens))
        out_t = max(0, int(output_tokens))
        rec.progress.input_tokens += in_t
        rec.progress.output_tokens += out_t
        # Keep ``token_count`` in sync — it's the LLM-facing aggregate
        # number consumed by ``task_get`` / ``task_list``.
        rec.progress.token_count = (
            rec.progress.input_tokens + rec.progress.output_tokens
        )

    def update_summary(self, task_id: str, summary: str) -> None:
        """Overwrite ``progress.latest_summary`` and stamp the timestamp.

        Called by the :class:`AgentSummarizer` loop on every interval
        tick. Empty summaries are kept (the summarizer already filters
        whitespace), but ``None`` is rejected — use
        ``progress.latest_summary = None`` directly if a caller ever
        needs to clear it (no current site does).
        """
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.progress.latest_summary = summary
        rec.progress.summary_updated_at = time.time()

    # ------------------------------------------------------------------
    # Round 4F — transcript path + lazy terminal event + listener registry
    # ------------------------------------------------------------------

    def set_transcript_path(self, task_id: str, path: Path) -> None:
        """Pin the on-disk JSONL transcript location on the record.

        Called by :func:`run_task` after the transcript flush succeeds
        — the path is what ``task_get`` returns so the parent (or the
        operator scrolling ``/transcripts``) can find a child's history
        on disk after the parent process has restarted.
        """
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.transcript_path = path

    def terminal_event(self, task_id: str) -> _LazyEvent:
        """Per-task lazy terminal event.

        Allocated on first lookup. If the record is ALREADY in a
        terminal state at lookup time, the returned event is pre-set so
        a consumer that calls ``await store.terminal_event(...).wait()``
        right after observing the state short-circuits instead of
        hanging forever.
        """
        ev = self._terminal_events.get(task_id)
        if ev is None:
            ev = _LazyEvent()
            self._terminal_events[task_id] = ev
            rec = self._records.get(task_id)
            if rec is not None and rec.status != "running":
                # Race fix: record went terminal before any consumer
                # asked for the event. Pre-set so the next ``await wait()``
                # returns immediately.
                ev.set()
        return ev

    def add_terminal_listener(self, callback: TerminalListener) -> None:
        """Register a sync callback to fire on every terminal transition.

        Listeners are called AFTER the record's ``status`` /
        ``final_result`` / ``error`` / ``finished_at`` are set, so they
        observe a fully-built terminal record. Exceptions are journaled
        and swallowed — a buggy listener must NOT strand the mark_*
        call.
        """
        self._terminal_listeners.append(callback)

    def remove_terminal_listener(self, callback: TerminalListener) -> None:
        """Unregister a previously-registered listener; no-op if absent."""
        try:
            self._terminal_listeners.remove(callback)
        except ValueError:
            return

    def _fire_terminal(self, rec: TaskRecord) -> None:
        """Common terminal-fanout: trigger event + dispatch listeners.

        Called by every ``mark_*`` terminal method. Listener exceptions
        caught + journaled so one bad listener can't poison the rest.
        """
        ev = self._terminal_events.get(rec.id)
        if ev is not None:
            ev.set()
        else:
            # No consumer ever called ``terminal_event`` for this task
            # — eagerly create + pre-set so a LATER consumer that comes
            # in after the terminal mark still short-circuits. Cheap:
            # _LazyEvent is two slots and defers the real Event build.
            placeholder = _LazyEvent()
            placeholder.set()
            self._terminal_events[rec.id] = placeholder
        for listener in list(self._terminal_listeners):
            try:
                listener(rec)
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "tasks_terminal_listener_error",
                    task_id=rec.id,
                    error=f"{type(exc).__name__}: {exc}",
                )

    def mark_completed(self, task_id: str, result: str) -> None:
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.status = "completed"
        rec.final_result = result
        rec.finished_at = time.time()
        self._fire_terminal(rec)

    def mark_failed(self, task_id: str, error: str) -> None:
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.status = "failed"
        rec.error = error
        rec.finished_at = time.time()
        self._fire_terminal(rec)

    def mark_cancelled(self, task_id: str) -> None:
        rec = self._records.get(task_id)
        if rec is None:
            return
        rec.status = "cancelled"
        rec.finished_at = time.time()
        self._fire_terminal(rec)


# Type re-export so ``from aura.core.tasks.store import TaskNotification``
# remains a stable import path even though TaskNotification's home is
# ``types.py``. Several CLI / Context.build call sites historically
# imported from store; keep both addresses live.
__all__ = [
    "TasksStore",
    "TaskNotification",
    "TerminalListener",
]
