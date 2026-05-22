"""AbortController — per-astream cancel signal threaded via contextvar."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar


class AbortException(Exception):
    """Raised when an in-flight call hits ``AbortController.abort``.

    Distinct from ``asyncio.CancelledError`` so the loop can tell self-
    abort from runtime cancellation; both balance history.
    """

    def __init__(self, reason: str = "aborted") -> None:
        super().__init__(reason)
        self.reason = reason


class AbortController:
    """Idempotent: second ``abort()`` keeps the first reason."""

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
        return self._event

    def abort(self, reason: str = "aborted") -> None:
        if self._event.is_set():
            return
        self._reason = reason
        self._event.set()


current_abort_signal: ContextVar[AbortController | None] = ContextVar(
    "aura_abort_signal", default=None,
)
"""Live AbortController for the running ``astream`` task, or ``None``.

Tools polling long-running work should read this once per iteration and
raise :class:`AbortException` (or :class:`asyncio.CancelledError`) when
``aborted`` flips. The contextvar propagates across ``await`` and child
tasks created via ``asyncio.create_task``.
"""


__all__ = [
    "AbortController",
    "AbortException",
    "current_abort_signal",
]
