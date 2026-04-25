"""AbortController — global cancel signal for one ``Agent.astream`` call.

Mirrors claude-code's ``ToolUseContext.abortController`` (TypeScript
``AbortController`` from ``query.ts``). One controller is created per
``astream`` invocation and threaded into the loop, tool dispatch, and any
spawned subagent so a single ``.abort(reason)`` call collapses every level
of in-flight work.

Why a contextvar instead of plumbing the signal through every signature:
the loop already uses :mod:`aura.tools.progress` for the same shape (tools
that opt into a cross-cutting per-task channel), tools that want to poll
``aborted`` shouldn't have to mutate their kwargs surface, and subagents
spawned inside ``task_create`` inherit the parent's contextvar by virtue
of running on the same task tree. Direct kwarg threading is still used at
the structural seams (``AgentLoop.run_turn``, ``run_task``) so the static
ownership trail stays explicit.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar


class AbortException(Exception):
    """Raised when an in-flight tool / model call hits ``AbortController.abort``.

    Distinct from ``asyncio.CancelledError`` so the loop can tell "we tore
    this down ourselves" apart from "the asyncio runtime cancelled us."
    Both branches end up balancing history, but the journal events differ.
    """

    def __init__(self, reason: str = "aborted") -> None:
        super().__init__(reason)
        self.reason = reason


class AbortController:
    """Per-astream cancel signal.

    Cheap to construct, owned by ``Agent.astream``, surfaced to deeply
    nested code through :data:`current_abort_signal`. ``.abort(reason)``
    is idempotent — calling it twice keeps the first reason.
    """

    __slots__ = ("_event", "_reason")

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str = ""

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def signal(self) -> asyncio.Event:
        # Exposed so callers that want to ``await controller.signal.wait()``
        # for fan-in (race a subprocess against the abort) can do so without
        # reaching into a private attribute.
        return self._event

    def abort(self, reason: str = "aborted") -> None:
        if self._event.is_set():
            # First reason wins — the second call is a duplicate fan-in
            # (e.g. parent abort + child timeout firing concurrently).
            return
        self._reason = reason
        self._event.set()


current_abort_signal: ContextVar[AbortController | None] = ContextVar(
    "aura_abort_signal", default=None,
)
"""Live AbortController for the running ``astream`` task, or ``None``.

Tools that loop / poll long-running work should read this once at the
start of each iteration and bail with :class:`AbortException` (or by
raising :class:`asyncio.CancelledError`) when ``aborted`` flips. The
contextvar propagates across ``await`` and across child tasks created via
``asyncio.create_task`` from inside the same task — exactly the inheritance
shape we want for subagents (which are detached tasks but spawned from
inside the parent's context)."""


__all__ = [
    "AbortController",
    "AbortException",
    "current_abort_signal",
]
