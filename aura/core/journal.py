"""Append-only JSONL journal — write with fsync per event for crash-durable audit."""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

_path: Path | None = None


def configure(path: Path) -> None:
    global _path
    path.parent.mkdir(parents=True, exist_ok=True)
    _path = path


def reset() -> None:
    global _path
    _path = None


def is_configured() -> bool:
    return _path is not None


def write(event: str, /, **fields: Any) -> None:
    if _path is None:
        return
    try:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "event": event,
            **fields,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with _path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            with contextlib.suppress(OSError, ValueError):
                os.fsync(f.fileno())
    except Exception:  # noqa: BLE001
        pass
