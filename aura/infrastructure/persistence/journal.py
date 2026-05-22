"""Append-only WAL JSONL — every event flushed + fsync'd, crash-safe."""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

_path: Path | None = None

# Per-task override; asyncio.Task copies the current Context on creation
# so child tasks inherit the parent's session scope.
_SESSION_PATH: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "journal_session_path", default=None,
)


def configure(path: Path) -> None:
    global _path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # mkdir failure on a bad path must not crash startup — degrade silently.
        _path = None
        print(
            f"aura: audit log disabled — cannot prepare {path.parent}: {exc}",
            file=sys.stderr,
        )
        return
    _path = path


def reset() -> None:
    global _path
    _path = None


@contextlib.contextmanager
def session_scope(path: Path) -> Generator[None, None, None]:
    """Route journal writes to ``path`` within this context; nests via contextvars."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover
        print(
            f"aura: session audit disabled — cannot prepare {path.parent}: {exc}",
            file=sys.stderr,
        )
        token = _SESSION_PATH.set(None)
        try:
            yield
        finally:
            _SESSION_PATH.reset(token)
        return
    token = _SESSION_PATH.set(path)
    try:
        yield
    finally:
        _SESSION_PATH.reset(token)


def write(event: str, /, **fields: Any) -> None:
    active = _SESSION_PATH.get() or _path
    if active is None:
        return
    try:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "event": event,
            **fields,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with active.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            # tmpfs / nfs may reject fsync; suppress to keep agent running.
            with contextlib.suppress(OSError, ValueError):
                os.fsync(f.fileno())
    except Exception:  # noqa: BLE001
        # Contract: audit failure must NEVER crash the agent.
        pass
