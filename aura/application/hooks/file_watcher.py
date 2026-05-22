"""Polling file watcher → :class:`FileChangedHook`."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from aura.application.hooks import FileChangeKind, HookChain
from aura.infrastructure.persistence import journal
from aura.schemas.state import LoopState

_DEFAULT_POLL_INTERVAL_SEC = 1.0


@dataclass(frozen=True)
class _Snap:
    mtime: float | None


class FileWatcher:
    """Async polling watcher. ``start`` / ``stop`` are idempotent."""

    def __init__(
        self,
        *,
        paths: Iterable[Path],
        chain: HookChain,
        state: LoopState,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SEC,
    ) -> None:
        self._roots: list[Path] = [Path(p).expanduser().resolve() for p in paths]
        self._chain = chain
        self._state = state
        self._poll_interval = poll_interval
        self._task: asyncio.Task[None] | None = None
        self._snapshots: dict[Path, _Snap] = {}

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        # Take initial snapshot synchronously so the first tick has a baseline.
        self._snapshots = self._take_snapshot()
        self._task = asyncio.create_task(
            self._poll_loop(), name="aura-file-watcher",
        )
        journal.write(
            "file_watcher_started",
            paths=[str(p) for p in self._roots],
            poll_interval=self._poll_interval,
        )

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        self._task = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        journal.write("file_watcher_stopped")

    async def _poll_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._poll_interval)
                await self._tick()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            journal.write(
                "file_watcher_error",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    async def _tick(self) -> None:
        new_snap = self._take_snapshot()
        events: list[tuple[Path, FileChangeKind]] = []

        for path, snap in new_snap.items():
            prev = self._snapshots.get(path)
            if prev is None or prev.mtime is None:
                if snap.mtime is not None:
                    events.append((path, "created"))
            else:
                if snap.mtime is None:
                    events.append((path, "deleted"))
                elif snap.mtime != prev.mtime:
                    events.append((path, "modified"))

        # Files inside a watched dir that vanished between snapshots.
        for path, prev in self._snapshots.items():
            if path in new_snap:
                continue
            if prev.mtime is not None:
                events.append((path, "deleted"))

        self._snapshots = new_snap

        for path, kind in events:
            try:
                await self._chain.run_file_changed(
                    path=path, kind=kind, state=self._state,
                )
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "file_watcher_consumer_error",
                    path=str(path),
                    kind=kind,
                    error=f"{type(exc).__name__}: {exc}",
                )

    def _take_snapshot(self) -> dict[Path, _Snap]:
        snap: dict[Path, _Snap] = {}
        for root in self._roots:
            try:
                if root.is_dir():
                    for child in root.rglob("*"):
                        if child.is_file():
                            try:
                                mtime = child.stat().st_mtime
                            except OSError:
                                continue
                            snap[child.resolve()] = _Snap(mtime=mtime)
                    snap[root] = _Snap(mtime=root.stat().st_mtime)
                elif root.exists():
                    snap[root] = _Snap(mtime=root.stat().st_mtime)
                else:
                    snap[root] = _Snap(mtime=None)
            except OSError:
                continue
        return snap


def default_watch_paths(cwd: Path) -> list[Path]:
    home = Path.home()
    return [
        home / ".aura" / "AURA.md",
        home / ".aura" / "skills",
        cwd / ".aura",
    ]
