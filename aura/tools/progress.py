"""Progress-callback ContextVar for streaming-output tools."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Literal

ProgressCallback = Callable[[Literal["stdout", "stderr"], str], None]

_progress_cb: ContextVar[ProgressCallback | None] = ContextVar(
    "aura_progress_cb", default=None,
)


def set_progress_callback(cb: ProgressCallback | None) -> object:
    return _progress_cb.set(cb)


def reset_progress_callback(token: object) -> None:
    _progress_cb.reset(token)  # type: ignore[arg-type]


def get_progress_callback() -> ProgressCallback | None:
    return _progress_cb.get()
