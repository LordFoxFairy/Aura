"""JSONL mailbox + ``.seen`` cursor sidecar.

Writes are append-only; reads use a sidecar ``.seen`` file so we never
mutate the JSONL itself (no rewrite race, no torn writes).
``fcntl.flock`` guards the append because TeamMessage lines (~700 B
after Pydantic dump) sit above macOS APFS's 512-byte atomicity floor.
Exactly one reader per recipient.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path

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

    def wait_for_new_message(
        self,
        member: str,
        *,
        poll_interval: float = 0.2,
        timeout: float = 30.0,
    ) -> bool:
        """Block until an unseen message arrives or ``timeout`` elapses.

        Sync; called via ``asyncio.to_thread`` from the runtime loop.
        """
        deadline = time.monotonic() + timeout
        while True:
            if self.read_unseen(member):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(poll_interval, remaining))
