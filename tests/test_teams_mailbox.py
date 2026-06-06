"""Mailbox JSONL append + .seen cursor + concurrent writers + notifiers."""

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path

import pytest

from aura.application.teams.mailbox import (
    FileMailboxNotifier,
    Mailbox,
    QueueMailboxNotifier,
)
from aura.domain.team import MAX_BODY_CHARS, TeamMessage, TeamMessageKind
from aura.infrastructure.persistence.storage import SessionStorage


def _msg(
    sender: str = "leader",
    recipient: str = "alice",
    body: str = "hi",
    kind: TeamMessageKind = "text",
) -> TeamMessage:
    return TeamMessage(
        msg_id=uuid.uuid4().hex,
        sender=sender,
        recipient=recipient,
        body=body,
        kind=kind,
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


@pytest.mark.asyncio
async def test_file_notifier_returns_false_on_timeout(tmp_path: Path) -> None:
    box = Mailbox(_storage(tmp_path), "team-a")
    notifier = FileMailboxNotifier(box)
    assert await notifier.wait_new("alice", timeout=0.1) is False


@pytest.mark.asyncio
async def test_file_notifier_returns_true_when_unseen_present(
    tmp_path: Path,
) -> None:
    box = Mailbox(_storage(tmp_path), "team-a")
    box.append(_msg(body="ready"))
    notifier = FileMailboxNotifier(box)
    assert await notifier.wait_new("alice", timeout=1.0) is True


@pytest.mark.asyncio
async def test_queue_notifier_signal_wakes_immediately() -> None:
    notifier = QueueMailboxNotifier()

    async def signal_after_delay() -> None:
        await asyncio.sleep(0.05)
        notifier.signal("alice")

    asyncio.create_task(signal_after_delay())
    # The 1s timeout is generous; signal arrives at 50 ms so the wait
    # should return well before deadline. If it didn't, we'd block ≥1 s
    # — pin that the queue path is truly async-native.
    assert await notifier.wait_new("alice", timeout=1.0) is True


@pytest.mark.asyncio
async def test_queue_notifier_per_member_isolation() -> None:
    notifier = QueueMailboxNotifier()
    notifier.signal("alice")
    # Bob's wait must time out — alice's signal is not broadcast.
    assert await notifier.wait_new("bob", timeout=0.1) is False
    assert await notifier.wait_new("alice", timeout=0.1) is True


@pytest.mark.asyncio
async def test_queue_notifier_edge_triggered_drain() -> None:
    """Two signals before a wait still cause exactly one wake — caller
    drains the mailbox JSONL on that wake."""
    notifier = QueueMailboxNotifier()
    notifier.signal("alice")
    notifier.signal("alice")
    assert await notifier.wait_new("alice", timeout=0.1) is True
    # No second wake without another signal.
    assert await notifier.wait_new("alice", timeout=0.1) is False


def _inbox_for(tmp_path: Path, member: str = "alice") -> Path:
    return _storage(tmp_path).team_inbox_path("team-a", member)


def _replace_with_dir(path: Path) -> None:
    """Occupy a file slot with a directory so open() raises IsADirectoryError
    (an OSError subclass) — exercises the I/O-failure branch without faking it."""
    if path.exists():
        path.unlink()
    path.mkdir()


def test_mailbox_read_all_skips_blank_lines(tmp_path: Path) -> None:
    """Blank/whitespace lines from interrupted appends must be skipped, not
    surface as empty TeamMessage rows."""
    box = Mailbox(_storage(tmp_path), "team-a")
    box.append(_msg(body="first"))
    inbox = _inbox_for(tmp_path)
    with inbox.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("   \n")
    box.append(_msg(body="third"))
    assert [m.body for m in box.read_all("alice")] == ["first", "third"]


def test_mailbox_read_all_unreadable_inbox_returns_empty(tmp_path: Path) -> None:
    """An inbox path that exists but can't be opened (corrupt/locked slot) must
    degrade to an empty read, never crash the caller's drain loop."""
    box = Mailbox(_storage(tmp_path), "team-a")
    _replace_with_dir(_inbox_for(tmp_path))
    assert box.read_all("alice") == []
    assert box.read_unseen("alice") == []


def test_mailbox_read_unseen_empty_inbox(tmp_path: Path) -> None:
    """A member with no inbox file yet has zero unseen — the missing-file path
    returns [] rather than raising."""
    box = Mailbox(_storage(tmp_path), "team-a")
    assert box.read_unseen("ghost") == []
    assert box.read_all("ghost") == []


def test_mailbox_ack_empty_list_is_noop(tmp_path: Path) -> None:
    """Acking zero ids must not create or touch the .seen cursor file."""
    box = Mailbox(_storage(tmp_path), "team-a")
    box.append(_msg(body="one"))
    seen_path = _inbox_for(tmp_path).with_suffix(".seen")
    box.ack("alice", [])
    assert not seen_path.exists()
    assert [m.body for m in box.read_unseen("alice")] == ["one"]


def test_mailbox_ack_tolerates_cursor_write_failure(tmp_path: Path) -> None:
    """If the temp cursor file can't be written, ack swallows the error and the
    message stays unseen — re-delivery keeps delivery at-least-once."""
    box = Mailbox(_storage(tmp_path), "team-a")
    m1 = _msg(body="one")
    box.append(m1)
    seen = _inbox_for(tmp_path).with_suffix(".seen")
    tmp = seen.with_suffix(seen.suffix + ".tmp")
    _replace_with_dir(tmp)
    box.ack("alice", [m1.msg_id])
    assert not seen.exists()
    assert [m.body for m in box.read_unseen("alice")] == ["one"]


def test_mailbox_load_seen_unreadable_cursor_treats_all_unseen(
    tmp_path: Path,
) -> None:
    """An unreadable .seen cursor must fall back to 'nothing acked' so messages
    are re-delivered rather than silently dropped."""
    box = Mailbox(_storage(tmp_path), "team-a")
    box.append(_msg(body="one"))
    _replace_with_dir(_inbox_for(tmp_path).with_suffix(".seen"))
    assert [m.body for m in box.read_unseen("alice")] == ["one"]


def test_mailbox_multiple_recipients_isolated(tmp_path: Path) -> None:
    """Each recipient owns a separate inbox — one member's messages and acks
    must never bleed into another's unseen view."""
    box = Mailbox(_storage(tmp_path), "team-a")
    a = _msg(recipient="alice", body="for-alice")
    b = _msg(recipient="bob", body="for-bob")
    box.append(a)
    box.append(b)
    assert [m.body for m in box.read_unseen("alice")] == ["for-alice"]
    assert [m.body for m in box.read_unseen("bob")] == ["for-bob"]
    box.ack("alice", [a.msg_id])
    assert box.read_unseen("alice") == []
    assert [m.body for m in box.read_unseen("bob")] == ["for-bob"]


def test_mailbox_ack_unknown_id_is_harmless(tmp_path: Path) -> None:
    """Acking an id that was never delivered must persist quietly without
    affecting real pending messages."""
    box = Mailbox(_storage(tmp_path), "team-a")
    m1 = _msg(body="real")
    box.append(m1)
    box.ack("alice", ["never-existed"])
    assert [m.body for m in box.read_unseen("alice")] == ["real"]
    box.ack("alice", [m1.msg_id, "another-ghost"])
    assert box.read_unseen("alice") == []


@pytest.mark.asyncio
async def test_queue_notifier_wait_times_out_without_signal() -> None:
    """A bare wait with no signal must return False at the deadline — the
    timeout boundary of the queue path."""
    notifier = QueueMailboxNotifier()
    assert await notifier.wait_new("alice", timeout=0.05) is False


@pytest.mark.asyncio
async def test_file_notifier_signal_is_noop(tmp_path: Path) -> None:
    """File-backed notifier polls the JSONL, so signal() is intentionally inert
    and must not raise or alter unseen state."""
    box = Mailbox(_storage(tmp_path), "team-a")
    notifier = FileMailboxNotifier(box)
    notifier.signal("alice")  # inert: must not raise or change unseen state
    box.append(_msg(body="ready"))
    assert await notifier.wait_new("alice", timeout=1.0) is True


@pytest.mark.asyncio
async def test_file_notifier_zero_timeout_returns_false(tmp_path: Path) -> None:
    """A zero/negative remaining budget must short-circuit to False without an
    extra poll sleep — the numeric deadline boundary."""
    box = Mailbox(_storage(tmp_path), "team-a")
    notifier = FileMailboxNotifier(box)
    assert await notifier.wait_new("alice", timeout=0.0) is False
