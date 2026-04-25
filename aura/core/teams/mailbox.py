"""JSONL mailbox + ``.seen`` cursor sidecar.

Per design §3.4 + open-question resolution #2: writes are append-only
JSONL, reads use a sidecar ``.seen`` file recording the last consumed
``msg_id`` so we never have to mutate the JSONL itself (no rewrite
race, no torn writes on the body).

Concurrency contract:

- Multiple writers may ``append`` concurrently. ``write(2)`` of a single
  ``TeamMessage`` line is small enough (<2KB after Pydantic dump) to be
  POSIX-atomic on Linux (PIPE_BUF=4096) and safe under
  ``fcntl.flock`` on macOS where the atomicity floor is 512 bytes.
- Exactly one reader per recipient (the teammate's runtime, or the
  leader's drain on the next turn). The sidecar simplifies "what's
  unread" without coordination.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path

from pydantic import ValidationError

from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.teams.types import TeamMessage


class Mailbox:
    """JSONL mailbox bound to one team_id.

    Cheap to construct — keeps no fds open between calls. ``append`` and
    ``read_unseen`` resolve the path via the supplied ``SessionStorage``
    helper at every call so dir creation is centralized.
    """

    def __init__(self, storage: SessionStorage, team_id: str) -> None:
        self._storage = storage
        self._team_id = team_id

    def _inbox(self, member: str) -> Path:
        return self._storage.team_inbox_path(self._team_id, member)

    def _seen_path(self, member: str) -> Path:
        # Sibling of the JSONL — keeps cursor + content adjacent so a
        # ``rm -rf <member>.*`` removes both at once.
        inbox = self._inbox(member)
        return inbox.with_suffix(".seen")

    def append(self, msg: TeamMessage) -> None:
        """Atomically append ``msg`` to the recipient's JSONL.

        Locking: ``fcntl.flock`` on the file descriptor while we write +
        flush. macOS APFS is strict about the 512-byte atomicity floor on
        ``write(2)``, and Pydantic-serialized TeamMessage lines can run
        ~700 bytes; the lock covers the gap. The lock is per-file, so two
        teammates writing to different recipients don't contend.
        """
        path = self._inbox(msg.recipient)
        line = msg.model_dump_json() + "\n"
        # ``a`` opens with O_APPEND; combined with flock it's safe for
        # concurrent senders. fsync best-effort — same robustness contract
        # as journal.write: the audit trail must not crash the agent.
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
            team_id=self._team_id,
            sender=msg.sender,
            recipient=msg.recipient,
            kind=msg.kind,
            msg_id=msg.msg_id,
        )

    def read_all(self, member: str) -> list[TeamMessage]:
        """Return every message in ``member``'s JSONL, oldest first.

        Robust to malformed lines: a line that fails JSON or pydantic
        validation is skipped (with a journal note) rather than crashing
        the reader. Used by tests and by ``read_unseen`` internally.
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
                            team_id=self._team_id,
                            recipient=member,
                        )
                        continue
        except OSError:
            return []
        return out

    def read_unseen(self, member: str) -> list[TeamMessage]:
        """Return messages with ``msg_id`` not yet recorded in ``.seen``.

        Implementation: load all, intersect against the cursor set. The
        cursor is a simple newline-separated msg_id list; bounded by
        teammate turn count so it stays O(messages-this-team) and
        well-formed without a separate index. Caller must ``ack`` after
        consumption to avoid replay on the next read.
        """
        seen = self._load_seen(member)
        return [m for m in self.read_all(member) if m.msg_id not in seen]

    def ack(self, member: str, msg_ids: list[str]) -> None:
        """Persist ``msg_ids`` into the ``.seen`` sidecar (idempotent).

        Sidecar is rewritten atomically (tmp+replace) — the previously-seen
        set unions with the new ids so the file always reflects "every
        message we've ever consumed". For Phase A teams the cursor file
        stays small (one id per line, hex uuid).
        """
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
            # Same robustness contract as journal: cursor failures
            # degrade to "we'll re-deliver next time" rather than
            # crash. Replay is harmless because we ack again.
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
        """Block until a new unseen message arrives or ``timeout`` elapses.

        Synchronous helper — the runtime calls this from inside its async
        loop via ``asyncio.to_thread`` so it doesn't block the event loop.
        Returns ``True`` if at least one unseen message exists, ``False``
        on timeout. Implementation polls the inbox + cursor; an inotify /
        kqueue path is a Phase B optimization (the design proposal §3.4
        explicitly accepts polling here).
        """
        deadline = time.monotonic() + timeout
        while True:
            if self.read_unseen(member):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(poll_interval, remaining))
