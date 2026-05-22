"""Session lifecycle runtime — extracted from :class:`Agent` (Phase 1 §5).

`Agent` used to inline session_id management, storage init + history
load/save, the partial-assistant streaming buffer, the SessionStart
re-arm flag, the parent→child read-record carry-over, and the
clear / resume / aclose lifecycle. Phase 1 splits all of that out so
:class:`Agent` shrinks toward its ≤600-line ultimate target and so
lifecycle behaviour can be exercised in isolation (no LangChain model,
no HookChain, no Context construction needed for unit tests).

`SessionRuntime` does NOT touch:
- the loop / turn lifecycle (``astream`` stays on Agent)
- the model / bound model / hook chain construction
- MCP wiring (Phase 2's territory — `McpRuntime`)
- Subagent factory wiring (Phase 6's territory)
- Permissions enforcement

It DOES own:
- ``session_id`` (uuid-or-default at construction; mutable via :meth:`resume`)
- the :class:`SessionStorage` reference
- per-session journal log path (when ``session_log_dir`` was passed)
- a snapshot of the runtime :class:`SessionRuleSet` (so :meth:`clear` can
  drop dynamically-added per-session permission rules)
- the partial-assistant streaming text buffer
- the ``session_start_fired`` re-arm flag
- the queued :class:`TaskNotification` list (subagent terminal events)
- the parent-read :class:`ReadCarryover` (snapshot for the FIRST
  Context build only; :meth:`clear` and :meth:`resume` deliberately
  drop it so long-gone parent reads never resurrect)
"""
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
    """Lifecycle + persistence sidecar for a single :class:`Agent`.

    One instance per Agent; mirrors Agent's lifetime exactly. Methods
    are all sync — async ``aclose`` is a coroutine the caller awaits
    so storage flushing happens deterministically.
    """

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
        # Session-scoped journal: when ``session_log_dir`` is passed, every
        # :func:`journal.write` made under :func:`journal.session_scope`
        # routes to a per-session JSONL file. Enables two concurrent
        # Agents in the same process (subagents, server workers) to keep
        # their audit trails fully separate. ``mkdir`` is idempotent.
        if session_log_dir is not None:
            session_log_dir.mkdir(parents=True, exist_ok=True)
            self._session_log_path: Path | None = (
                session_log_dir / f"{self._session_id}.jsonl"
            )
        else:
            self._session_log_path = None
        # ``session_rules``: CLI hands in the SessionRuleSet that was used
        # to build the permission hook; :meth:`clear` drops its runtime
        # rules alongside history + state so /clear is coherent.
        self._session_rules = session_rules
        # F-05-003 partial-text buffer. ``Agent.astream`` appends to
        # this on every AssistantDelta event; if abort fires before the
        # final AIMessage we yield this as one last AssistantDelta so
        # the user doesn't lose half-streamed reasoning. Reset at the
        # start of each astream call.
        self._partial_assistant_text: str = ""
        # F-04-014: SessionStart fires exactly once per session.
        # Re-armed by :meth:`clear` / :meth:`resume`.
        self._session_start_fired: bool = False
        # Round 4F notification queue. Populated by external producers
        # (TasksStore terminal-listener, registered by :class:`Agent`),
        # drained by ``Context.build`` at the start of each prompt
        # envelope. Owned here so /clear can wipe it.
        self._pending_notifications: list[TaskNotification] = []
        # Unified coordination pipeline: external transports can drain
        # live coordination wire events from here without changing the
        # parent-facing prompt/context path. Producers append in parallel
        # with existing behavior; transports decide when to flush.
        self._pending_protocol_events: list[WireEvent] = []
        # Workstream G8 + Phase 3 Task 4 — ``carryover`` only flows
        # into the FIRST Context construction. ``clear_session`` and
        # the post-compact rebuild build their own fresh Contexts and
        # must NOT resurrect a long-gone parent's read fingerprints.
        # We hold the snapshot so :meth:`Agent` can read it once at
        # construction; subagent spawn re-snapshots the parent at each
        # ``SubagentFactory.spawn``.
        self._carryover: ReadCarryover | None = carryover

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

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

    @property
    def pending_notifications(self) -> tuple[TaskNotification, ...]:
        """Snapshot of queued :class:`TaskNotification` records."""
        return tuple(self._pending_notifications)

    @property
    def pending_protocol_events(self) -> tuple[WireEvent, ...]:
        """Snapshot of queued external coordination wire events."""
        return tuple(self._pending_protocol_events)

    @property
    def carryover(self) -> ReadCarryover | None:
        """One-shot carryover — meant for the FIRST Context build only."""
        return self._carryover

    # ------------------------------------------------------------------
    # History persistence — thin pass-through to :class:`SessionStorage`
    # ------------------------------------------------------------------

    def load_history(self) -> list[BaseMessage]:
        """Load the live session's persisted history."""
        return self._storage.load(self._session_id)

    def save_history(self, history: list[BaseMessage]) -> None:
        """Persist ``history`` under the live session_id."""
        self._storage.save(self._session_id, history)

    # ------------------------------------------------------------------
    # Streaming buffer + notification queue
    # ------------------------------------------------------------------

    def buffer_partial_assistant_text(self, text: str) -> None:
        """Append ``text`` to the partial-assistant buffer."""
        self._partial_assistant_text += text

    def reset_partial_assistant_text(self) -> None:
        """Clear the partial-assistant buffer (start of every astream)."""
        self._partial_assistant_text = ""

    def take_partial_assistant_text(self) -> str:
        """Return + clear the partial-assistant buffer atomically.

        Used by the abort path — the caller flushes the buffered text
        as one last AssistantDelta and the buffer is empty afterwards
        so re-entry through :meth:`buffer_partial_assistant_text` doesn't
        double-count what the renderer already saw.
        """
        text = self._partial_assistant_text
        self._partial_assistant_text = ""
        return text

    def enqueue_task_notification(self, notif: TaskNotification) -> None:
        """External producer hook — append ``notif`` to the queue.

        Unbounded at the queue level — the build-time renderer caps the
        emitted block at 5 entries (FIFO) and collapses the tail to a
        ``(N more earlier)`` line, so the parent's prompt envelope stays
        compact while the queue itself preserves order.
        """
        self._pending_notifications.append(notif)

    def drain_task_notifications(self) -> list[TaskNotification]:
        """Pop every queued notification and return them, oldest first."""
        drained = list(self._pending_notifications)
        self._pending_notifications.clear()
        return drained

    def enqueue_protocol_event(self, event: WireEvent) -> None:
        """Append one external coordination wire event for transport drains."""
        self._pending_protocol_events.append(event)

    def drain_protocol_events(self) -> list[WireEvent]:
        """Pop every queued protocol event and return them, oldest first."""
        drained = list(self._pending_protocol_events)
        self._pending_protocol_events.clear()
        return drained

    # ------------------------------------------------------------------
    # SessionStart re-arm flag
    # ------------------------------------------------------------------

    def mark_session_start_fired(self) -> None:
        self._session_start_fired = True

    def rearm_session_start(self) -> None:
        self._session_start_fired = False

    # ------------------------------------------------------------------
    # Lifecycle — clear / resume / close
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Wipe the live session.

        Drops persisted history, re-arms SessionStart, clears the
        partial-assistant buffer + notification queue, and removes
        runtime per-session permission rules (if a SessionRuleSet was
        registered). Does NOT rebuild Context / hooks / loop — those
        belong to :class:`Agent` because they need model + hook + skill
        wiring that lives outside the session lifecycle.
        """
        self._storage.clear(self._session_id)
        if self._session_rules is not None:
            self._session_rules.clear()
        self._pending_notifications.clear()
        self._pending_protocol_events.clear()
        self._partial_assistant_text = ""
        self._session_start_fired = False
        # /clear starts a fresh session — long-gone parent reads must
        # not resurrect into the new Context the Agent will rebuild.
        self._carryover = None

    def resume(self, session_id: str) -> int:
        """Swap the live session_id; return the loaded message count.

        Loads ``session_id``'s history from storage, updates the live
        session_id, drops partial buffers, re-arms SessionStart so the
        lifecycle fires again on the next astream, and updates the
        per-session journal log path when one is configured. Raises
        :class:`KeyError` if the requested session has no rows.
        """
        history = self._storage.load(session_id)
        if not history:
            raise KeyError(
                f"session {session_id!r} has no persisted history"
            )
        # Re-target the per-session journal log path BEFORE flipping
        # session_id so the new ``session_resumed`` event is correctly
        # attributed (we journal AFTER the assignment but BEFORE the
        # caller's first astream).
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
        """Flush + close the underlying :class:`SessionStorage`.

        Idempotent — :meth:`SessionStorage.close` tolerates repeat calls.
        """
        self._storage.close()
