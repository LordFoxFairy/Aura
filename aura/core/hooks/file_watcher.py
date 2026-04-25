"""Pure-stdlib polling file watcher → fires :class:`FileChangedHook`.

Why polling, not ``watchdog``: keeping Aura's dep tree tight is a
deliberate trade-off (see ``feedback_no_compromise``). The set of
watched paths is small (project AURA.md, ~/.aura/AURA.md, the skills
dir, the project ``.aura/`` dir) and the polling interval is human-
scale, so an mtime-stat loop costs effectively nothing even on a
laptop with a tight battery budget.

The watcher:

- Snapshots ``(path, mtime, exists)`` for every path it's told to
  watch on each tick.
- For directories, descends one level (non-recursive on subdirs of
  subdirs is fine for ``.aura/skills/`` which is a flat-ish tree;
  ``rglob`` keeps recursion correct without fighting the stdlib).
- Compares against the previous snapshot to derive
  ``created`` / ``modified`` / ``deleted`` events.
- Awaits :meth:`HookChain.run_file_changed` for each event in turn so
  a slow consumer can't be raced by a follow-up tick.

Errors (a watched path that became permission-denied, a transient
``OSError`` on stat) are journalled and skipped, never re-raised —
the watcher must never crash the Agent.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from aura.core.hooks import FileChangeKind, HookChain
from aura.core.persistence import journal
from aura.schemas.state import LoopState

_DEFAULT_POLL_INTERVAL_SEC = 1.0


@dataclass(frozen=True)
class _Snap:
    """Single snapshot entry — None mtime means "did not exist this tick"."""

    mtime: float | None


class FileWatcher:
    """Async polling file watcher.

    The instance owns one :class:`asyncio.Task` between :meth:`start`
    and :meth:`stop`; both are idempotent so the CLI's outer ``finally``
    can call ``stop()`` regardless of the start path's success.
    """

    def __init__(
        self,
        *,
        paths: Iterable[Path],
        chain: HookChain,
        state: LoopState,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SEC,
    ) -> None:
        # ``Path.resolve`` so ``~/...`` and relative paths are normalized
        # before snapshotting; equality keys must be stable.
        self._roots: list[Path] = [Path(p).expanduser().resolve() for p in paths]
        self._chain = chain
        self._state = state
        self._poll_interval = poll_interval
        self._task: asyncio.Task[None] | None = None
        self._snapshots: dict[Path, _Snap] = {}

    async def start(self) -> None:
        """Begin polling. Idempotent — second start is a no-op."""
        if self._task is not None and not self._task.done():
            return
        # Take the initial snapshot synchronously so the very first tick
        # of the polling loop has a baseline. Without this, the first
        # tick would (correctly) flag every existing watched file as
        # ``modified`` against an empty prior — noisy and wrong on
        # startup.
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
        """Cancel the polling task. Idempotent + safe under recwarn."""
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
            # An unexpected raise here would silently kill the task. Log
            # before propagating so an operator sees the trail; then
            # let the exception escape so a real bug isn't masked.
            journal.write(
                "file_watcher_error",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    async def _tick(self) -> None:
        new_snap = self._take_snapshot()
        events: list[tuple[Path, FileChangeKind]] = []

        # Detect created + modified by walking the new snapshot.
        for path, snap in new_snap.items():
            prev = self._snapshots.get(path)
            if prev is None or prev.mtime is None:
                # Path didn't exist last tick (or wasn't tracked). If
                # it exists now, that's a creation.
                if snap.mtime is not None:
                    events.append((path, "created"))
            else:
                # Existed last tick.
                if snap.mtime is None:
                    # Has been removed.
                    events.append((path, "deleted"))
                elif snap.mtime != prev.mtime:
                    events.append((path, "modified"))

        # Detect deletions for paths that were present last tick but not
        # discovered this tick (e.g. a file inside a watched dir
        # disappears entirely between snapshots — handled here, not in
        # the loop above, because such a path won't appear in new_snap
        # at all).
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
                # A buggy consumer must not stop the watcher. Journal
                # and continue with the next event.
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
                    # Track every regular file under the directory.
                    # ``rglob`` quietly skips broken symlinks and
                    # permission-denied subtrees on most platforms.
                    for child in root.rglob("*"):
                        if child.is_file():
                            try:
                                mtime = child.stat().st_mtime
                            except OSError:
                                continue
                            snap[child.resolve()] = _Snap(mtime=mtime)
                    # Also record the directory itself so deletion of
                    # the entire dir is observable. mtime of dir is
                    # noisy but we don't use it as a change signal —
                    # deletions of children handle that.
                    snap[root] = _Snap(mtime=root.stat().st_mtime)
                elif root.exists():
                    # Plain file watch.
                    snap[root] = _Snap(mtime=root.stat().st_mtime)
                else:
                    # Path doesn't exist (yet) — record it explicitly so
                    # a future creation flips from None → mtime and
                    # generates a ``created`` event.
                    snap[root] = _Snap(mtime=None)
            except OSError:
                # Permission denied / transient FS hiccup — skip this
                # root for this tick. Next tick may succeed.
                continue
        return snap


def default_watch_paths(cwd: Path) -> list[Path]:
    """Canonical set of paths Aura watches by default.

    Returned in deterministic order so journal output is stable across
    runs. Caller is free to extend or filter; this is just the
    out-of-the-box wiring.
    """
    home = Path.home()
    return [
        home / ".aura" / "AURA.md",
        home / ".aura" / "skills",
        cwd / ".aura",
    ]
