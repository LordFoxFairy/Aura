"""JSONL mailbox + ``.seen`` cursor + notifier strategies.

Append-only JSONL stays the source of record (so ``/team view`` and
post-mortem replay both work uniformly), but new-message wake-up runs
through a :class:`MailboxNotifier` strategy:

- :class:`QueueMailboxNotifier` — per-recipient ``asyncio.Event`` flipped
  on every ``signal()``; used by the in-process backend where the
  publisher and consumer share a loop, eliminating the 200 ms poll.
- :class:`FileMailboxNotifier` — coarse-grained sleep + filesystem
  re-check; used by the pane backend where the consumer is a separate
  Python process and the JSONL is the only IPC channel.

``fcntl.flock`` guards the append because TeamMessage lines (~700 B
after Pydantic dump) sit above macOS APFS's 512-byte atomicity floor.
Exactly one reader per recipient.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import time
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from aura.domain.team import TeamMessage
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage


class Mailbox:
    """JSONL mailbox bound to one team_id. Stateless — no open fds."""

    def __init__(self, storage: SessionStorage, team_id: str) -> None:
        self._storage = storage
        self._team_id = team_id

    def _inbox(self, member: str) -> Path:
        return self._storage.team_inbox_path(self._team_id, member)

    def _seen_path(self, member: str) -> Path:
        return self._inbox(member).with_suffix(".seen")

    def append(self, msg: TeamMessage) -> None:
        """Atomically append ``msg`` to the recipient's JSONL."""
        path = self._inbox(msg.recipient)
        line = msg.model_dump_json() + "\n"
        with path.open("a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(line)
                f.flush()
                with contextlib.suppress(OSError, ValueError):
                    os.fsync(f.fileno())
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        journal.write(
            "team_message_appended",
            team_id=self._team_id, sender=msg.sender,
            recipient=msg.recipient, kind=msg.kind, msg_id=msg.msg_id,
        )

    def read_all(self, member: str) -> list[TeamMessage]:
        """Return every message in ``member``'s JSONL, oldest first.

        Malformed lines are skipped (journaled) rather than aborting.
        """
        path = self._inbox(member)
        if not path.exists():
            return []
        out: list[TeamMessage] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        out.append(TeamMessage.model_validate_json(line))
                    except (ValidationError, ValueError):
                        journal.write(
                            "team_message_skipped_malformed",
                            team_id=self._team_id, recipient=member,
                        )
        except OSError:
            return []
        return out

    def read_unseen(self, member: str) -> list[TeamMessage]:
        """Return messages whose ``msg_id`` is not yet in ``.seen``."""
        seen = self._load_seen(member)
        return [m for m in self.read_all(member) if m.msg_id not in seen]

    def ack(self, member: str, msg_ids: list[str]) -> None:
        """Persist ``msg_ids`` into the ``.seen`` sidecar (atomic, idempotent)."""
        if not msg_ids:
            return
        path = self._seen_path(member)
        existing = self._load_seen(member)
        existing.update(msg_ids)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for mid in sorted(existing):
                    f.write(mid + "\n")
                f.flush()
                with contextlib.suppress(OSError):
                    os.fsync(f.fileno())
            tmp.replace(path)
        except OSError:
            # Cursor failure → re-deliver on next read; harmless because ack is idempotent.
            pass

    def _load_seen(self, member: str) -> set[str]:
        path = self._seen_path(member)
        if not path.exists():
            return set()
        try:
            with path.open("r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except OSError:
            return set()


class MailboxNotifier(Protocol):
    """Strategy for waking a teammate when a new message lands.

    Decouples wake-up cadence (which is backend-specific — instant for
    in-process, periodic for cross-process pane) from the JSONL storage
    of record.
    """

    async def wait_new(self, member: str, *, timeout: float) -> bool:
        """Return ``True`` when an unseen message is available for ``member``
        within ``timeout``; ``False`` on timeout. Must be cancel-safe."""
        ...

    def signal(self, member: str) -> None:
        """Hint that ``member`` may have a new message. Idempotent; no-op
        on notifiers that don't need explicit publish signals."""
        ...


class QueueMailboxNotifier:
    """Per-recipient ``asyncio.Event`` notifier — in-process publisher/consumer.

    ``signal(member)`` flips the event; ``wait_new`` awaits it and clears
    on return. The event is *edge-triggered*: a publisher that fires
    twice before the consumer wakes once still causes the consumer to
    drain the JSONL (read_unseen yields both messages on the same wake).
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}

    def _event(self, member: str) -> asyncio.Event:
        ev = self._events.get(member)
        if ev is None:
            ev = asyncio.Event()
            self._events[member] = ev
        return ev

    async def wait_new(self, member: str, *, timeout: float) -> bool:
        ev = self._event(member)
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
        except TimeoutError:
            return False
        ev.clear()
        return True

    def signal(self, member: str) -> None:
        self._event(member).set()


class FileMailboxNotifier:
    """Filesystem-polling notifier — cross-process pane backend.

    ``signal`` is a no-op (the publisher is in a different process and
    can't reach the consumer's event loop); ``wait_new`` re-reads the
    JSONL on a fixed 200 ms cadence until an unseen message appears or
    ``timeout`` elapses. Implemented on top of the existing
    :class:`Mailbox` cursor so we never have to mutate the JSONL.
    """

    _POLL_INTERVAL_SEC: float = 0.2

    def __init__(self, mailbox: Mailbox) -> None:
        self._mailbox = mailbox

    async def wait_new(self, member: str, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if self._mailbox.read_unseen(member):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(self._POLL_INTERVAL_SEC, remaining))

    def signal(self, member: str) -> None:
        # Cross-process: the publisher can't reach this notifier's loop.
        # Wake-up has to come from the polling cadence above.
        del member


__all__ = [
    "FileMailboxNotifier",
    "Mailbox",
    "MailboxNotifier",
    "QueueMailboxNotifier",
]
