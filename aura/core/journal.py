"""Append-only JSONL journal with fsync-per-write for crash-durable audit."""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol


class Journal(Protocol):
    def write(self, event: str, /, **fields: Any) -> None: ...
    def close(self) -> None: ...


class _NoOpJournal:
    def write(self, event: str, /, **fields: Any) -> None:
        return

    def close(self) -> None:
        return


class _FileJournal:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = path.open("a", encoding="utf-8", buffering=1)
        self._closed = False

    def write(self, event: str, /, **fields: Any) -> None:
        if self._closed:
            return
        try:
            payload: dict[str, Any] = {
                "ts": round(time.time(), 3),
                "event": event,
                **fields,
            }
            line = json.dumps(payload, ensure_ascii=False, default=str)
            self._fd.write(line + "\n")
            self._fd.flush()
            with contextlib.suppress(OSError, ValueError):
                os.fsync(self._fd.fileno())
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):  # noqa: BLE001
            self._fd.close()


_current: Journal = _NoOpJournal()


def set_journal(j: Journal) -> None:
    global _current
    _current = j


def journal() -> Journal:
    return _current


def setup_file_journal(path: Path) -> Journal:
    j = _FileJournal(path)
    set_journal(j)
    return j


def reset_journal() -> None:
    global _current
    current = _current
    if isinstance(current, _FileJournal):
        current.close()
    _current = _NoOpJournal()
