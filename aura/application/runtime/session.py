"""Session lifecycle + persistence sidecar for one :class:`Agent`."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage

from aura.domain.permission.session import SessionRuleSet
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.wire.event_dto import WireEvent
from aura.schemas.state import ReadCarryover

if TYPE_CHECKING:
    from aura.domain.task import TaskNotification


class SessionRuntime:
    """One per Agent; mirrors Agent's lifetime exactly."""

    def __init__(
        self,
        *,
        storage: SessionStorage,
        session_id: str,
        session_log_dir: Path | None = None,
        session_rules: SessionRuleSet | None = None,
        carryover: ReadCarryover | None = None,
    ) -> None:
        self._storage = storage
        self._session_id = session_id
        if session_log_dir is not None:
            session_log_dir.mkdir(parents=True, exist_ok=True)
            self._session_log_path: Path | None = (
                session_log_dir / f"{self._session_id}.jsonl"
            )
        else:
            self._session_log_path = None
        self._session_rules = session_rules
        # Flushed on abort as one last AssistantDelta so half-streamed reasoning isn't lost.
        self._partial_assistant_text: str = ""
        # SessionStart fires exactly once per session; re-armed by clear/resume.
        self._session_start_fired: bool = False
        self._pending_notifications: list[TaskNotification] = []
        self._pending_protocol_events: list[WireEvent] = []
        # One-shot: flows into the FIRST Context build only; clear/resume drop it
        # so long-gone parent reads never resurrect.
        self._carryover: ReadCarryover | None = carryover

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        # resume_session is the legitimate writer; setter retained for tests.
        self._session_id = value

    @property
    def storage(self) -> SessionStorage:
        return self._storage

    @property
    def session_log_path(self) -> Path | None:
        return self._session_log_path

    @property
    def session_rules(self) -> SessionRuleSet | None:
        return self._session_rules

    @property
    def session_start_fired(self) -> bool:
        return self._session_start_fired

    @property
    def partial_assistant_text(self) -> str:
        return self._partial_assistant_text

    @partial_assistant_text.setter
    def partial_assistant_text(self, value: str) -> None:
        self._partial_assistant_text = value

    @property
    def pending_notifications(self) -> tuple[TaskNotification, ...]:
        return tuple(self._pending_notifications)

    @property
    def pending_notifications_live(self) -> list[TaskNotification]:
        """Live mutable list — callers .append / .clear directly on the runtime's queue."""
        return self._pending_notifications

    @property
    def pending_protocol_events(self) -> tuple[WireEvent, ...]:
        return tuple(self._pending_protocol_events)

    @property
    def carryover(self) -> ReadCarryover | None:
        return self._carryover

    def load_history(self) -> list[BaseMessage]:
        return self._storage.load(self._session_id)

    def save_history(self, history: list[BaseMessage]) -> None:
        self._storage.save(self._session_id, history)

    def buffer_partial_assistant_text(self, text: str) -> None:
        self._partial_assistant_text += text

    def reset_partial_assistant_text(self) -> None:
        self._partial_assistant_text = ""

    def take_partial_assistant_text(self) -> str:
        """Return + clear atomically so the abort flush isn't double-counted."""
        text = self._partial_assistant_text
        self._partial_assistant_text = ""
        return text

    def enqueue_task_notification(self, notif: TaskNotification) -> None:
        self._pending_notifications.append(notif)

    def drain_task_notifications(self) -> list[TaskNotification]:
        """Pop every queued notification; oldest first."""
        drained = list(self._pending_notifications)
        self._pending_notifications.clear()
        return drained

    def enqueue_protocol_event(self, event: WireEvent) -> None:
        self._pending_protocol_events.append(event)

    def drain_protocol_events(self) -> list[WireEvent]:
        """Pop every queued protocol event; oldest first."""
        drained = list(self._pending_protocol_events)
        self._pending_protocol_events.clear()
        return drained

    def mark_session_start_fired(self) -> None:
        self._session_start_fired = True

    def rearm_session_start(self) -> None:
        self._session_start_fired = False

    def clear(self) -> None:
        """Wipe the live session; does NOT rebuild Context / hooks / loop."""
        self._storage.clear(self._session_id)
        if self._session_rules is not None:
            self._session_rules.clear()
        self._pending_notifications.clear()
        self._pending_protocol_events.clear()
        self._partial_assistant_text = ""
        self._session_start_fired = False
        self._carryover = None

    def resume(self, session_id: str) -> int:
        """Swap the live session_id; return the loaded message count.

        Raises :class:`KeyError` if the requested session has no rows.
        """
        history = self._storage.load(session_id)
        if not history:
            raise KeyError(
                f"session {session_id!r} has no persisted history"
            )
        # Re-target the log path BEFORE flipping session_id so journal
        # attribution lands under the new session.
        if self._session_log_path is not None:
            self._session_log_path = (
                self._session_log_path.parent / f"{session_id}.jsonl"
            )
        self._session_id = session_id
        self._partial_assistant_text = ""
        self._session_start_fired = False
        self._pending_notifications.clear()
        self._pending_protocol_events.clear()
        self._carryover = None
        journal.write(
            "session_resumed",
            session=session_id,
            message_count=len(history),
        )
        return len(history)

    def close_storage(self) -> None:
        """Flush + close the underlying storage; idempotent."""
        self._storage.close()
