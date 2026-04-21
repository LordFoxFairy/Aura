"""WAL 风格 JSONL 审计日志 — append-only，每事件 flush + fsync，进程/机器崩溃不丢数据。"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_path: Path | None = None


def configure(path: Path) -> None:
    # Invariant: audit failures never crash the agent (see write() below).
    # mkdir on a read-only FS, a path-under-a-file, or a non-existent root
    # raises OSError; honor the same contract here by falling through to
    # disabled state + a one-line stderr warning. Otherwise a bad log path
    # in config would crash startup before we ever reached write().
    global _path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
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
        # 每事件独立开关 fd，不留 handle 泄漏；flush 后 fsync 保证内核缓冲落盘。
        with _path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            # tmpfs / 网络挂载可能拒绝 fsync，suppress 避免因此中断 agent。
            with contextlib.suppress(OSError, ValueError):
                os.fsync(f.fileno())
    except Exception:  # noqa: BLE001
        # contract：日志失败绝不崩 agent — 审计是可选的，业务流程不依赖它。
        pass
