"""AbortController — per-astream cancel signal threaded via contextvar."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar


class AbortException(Exception):
    # Distinct from asyncio.CancelledError so the loop tells self-abort apart.

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


# Live AbortController for the running astream task; tools poll .aborted.
current_abort_signal: ContextVar[AbortController | None] = ContextVar(
    "aura_abort_signal",
    default=None,
)


__all__ = [
    "AbortController",
    "AbortException",
    "current_abort_signal",
]
