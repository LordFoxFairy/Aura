"""Minimal progress-callback plumbing for tools that want to stream output.

Only used by ``bash`` today. The agent loop sets a per-step callback via
:func:`set_progress_callback` right before invoking the tool, and clears it
after. Tools read the current callback via :func:`get_progress_callback` and
call it with ``(stream, chunk)`` to surface a ``ToolCallProgress`` event.

Stored in a ``ContextVar`` so concurrent tool calls in a single batch each
see their own callback without threading one through every tool signature
— which would churn the Tool public API.

Callers MUST NOT rely on progress ordering between concurrent tools; each
callback is independent. The callback itself is synchronous + non-blocking
(it just ``put_nowait`` onto a queue the loop drains).
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Literal

ProgressCallback = Callable[[Literal["stdout", "stderr"], str], None]

_progress_cb: ContextVar[ProgressCallback | None] = ContextVar(
    "aura_progress_cb", default=None,
)


def set_progress_callback(cb: ProgressCallback | None) -> object:
    """Install ``cb`` as the current progress callback. Returns a token the
    caller MUST pass to :func:`reset_progress_callback` in a ``finally``
    block so a failing tool invocation doesn't leak the callback to the
    next step in the same task."""
    return _progress_cb.set(cb)


def reset_progress_callback(token: object) -> None:
    """Restore the previous progress callback (paired with ``set``)."""
    _progress_cb.reset(token)  # type: ignore[arg-type]


def get_progress_callback() -> ProgressCallback | None:
    """Return the current progress callback, or ``None`` when no loop is
    listening. Tools SHOULD check this before formatting the chunk — a
    ``None`` return means we can skip the decode entirely."""
    return _progress_cb.get()
