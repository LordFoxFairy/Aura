"""Mailbox JSONL append + .seen cursor + concurrent writers."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest

from aura.core.persistence.storage import SessionStorage
from aura.core.teams.mailbox import Mailbox
from aura.core.teams.types import MAX_BODY_CHARS, TeamMessage


def _msg(
    sender: str = "leader",
    recipient: str = "alice",
    body: str = "hi",
    kind: str = "text",
) -> TeamMessage:
    return TeamMessage(
        msg_id=uuid.uuid4().hex,
        sender=sender,
        recipient=recipient,
        body=body,
        kind=kind,  # type: ignore[arg-type]
    )


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "sessions.db")


def test_mailbox_append_persists_message(tmp_path: Path) -> None:
    box = Mailbox(_storage(tmp_path), "team-a")
    box.append(_msg(body="first"))
    assert [m.body for m in box.read_all("alice")] == ["first"]


def test_mailbox_read_unseen_returns_only_unread(tmp_path: Path) -> None:
    box = Mailbox(_storage(tmp_path), "team-a")
    m1 = _msg(body="one")
    m2 = _msg(body="two")
    box.append(m1)
    box.append(m2)
    unseen1 = box.read_unseen("alice")
    assert [m.body for m in unseen1] == ["one", "two"]
    box.ack("alice", [m1.msg_id])
    unseen2 = box.read_unseen("alice")
    assert [m.body for m in unseen2] == ["two"]


def test_mailbox_ack_idempotent(tmp_path: Path) -> None:
    box = Mailbox(_storage(tmp_path), "team-a")
    m1 = _msg(body="one")
    box.append(m1)
    box.ack("alice", [m1.msg_id])
    box.ack("alice", [m1.msg_id])  # second ack — should not duplicate
    assert box.read_unseen("alice") == []


def test_mailbox_concurrent_writers_no_lost_messages(tmp_path: Path) -> None:
    """fcntl.flock + O_APPEND must give us atomic line writes under contention."""
    box = Mailbox(_storage(tmp_path), "team-a")
    sender_count = 4
    msgs_per_sender = 25

    def writer(sender_id: int) -> None:
        for i in range(msgs_per_sender):
            box.append(_msg(
                sender=f"s{sender_id}",
                body=f"msg-{sender_id}-{i}",
            ))

    threads = [
        threading.Thread(target=writer, args=(i,))
        for i in range(sender_count)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    all_msgs = box.read_all("alice")
    assert len(all_msgs) == sender_count * msgs_per_sender
    # Every (sender, body) pair shows up exactly once.
    expected = {
        f"msg-{s}-{i}"
        for s in range(sender_count)
        for i in range(msgs_per_sender)
    }
    assert {m.body for m in all_msgs} == expected


def test_mailbox_skips_malformed_lines(tmp_path: Path) -> None:
    box = Mailbox(_storage(tmp_path), "team-a")
    box.append(_msg(body="first"))
    # Inject a malformed line by hand
    inbox = _storage(tmp_path).team_inbox_path("team-a", "alice")
    with inbox.open("a") as f:
        f.write("THIS IS NOT JSON\n")
    box.append(_msg(body="third"))
    bodies = [m.body for m in box.read_all("alice")]
    assert bodies == ["first", "third"]


def test_mailbox_body_oversize_rejected_by_pydantic() -> None:
    with pytest.raises(ValueError):
        TeamMessage(
            msg_id=uuid.uuid4().hex,
            sender="leader",
            recipient="alice",
            body="x" * (MAX_BODY_CHARS + 1),
        )


def test_mailbox_wait_for_new_message_returns_false_on_timeout(
    tmp_path: Path,
) -> None:
    box = Mailbox(_storage(tmp_path), "team-a")
    # Tight timeout so the test is fast.
    assert box.wait_for_new_message(
        "alice", poll_interval=0.05, timeout=0.2,
    ) is False


def test_mailbox_wait_for_new_message_returns_true_on_arrival(
    tmp_path: Path,
) -> None:
    box = Mailbox(_storage(tmp_path), "team-a")
    box.append(_msg(body="ready"))
    assert box.wait_for_new_message(
        "alice", poll_interval=0.05, timeout=1.0,
    ) is True
