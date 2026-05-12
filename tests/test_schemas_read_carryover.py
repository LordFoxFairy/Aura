"""Tests for ``ReadRecord`` and ``ReadCarryover``.

Phase 3 Task 1 — schema contracts only. ``ReadCarryover`` replaces the
ad-hoc ``inherited_reads: dict[Path, _ReadRecord]`` propagation that
the subagent factory uses to share parent file-read records with a
spawned child. Task 4 wires the new type through the factory; Task 5
teaches ``must_read_first`` to call ``is_fresh`` before honoring an
inherited record. This file pins the schema invariants in isolation.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from aura.schemas.state import ReadCarryover, ReadRecord


def _make_record(path: Path, *, turn: int = 1) -> ReadRecord:
    """Stat ``path`` and produce a record matching its current state."""
    stat = path.stat()
    return ReadRecord(
        path=path,
        mtime_at_read=stat.st_mtime,
        size_at_read=stat.st_size,
        read_at_turn=turn,
    )


def test_read_carryover_constructs_empty() -> None:
    """A no-record carryover is the natural starting state for a parent
    that has not yet read any files. ``records`` defaults to an empty
    mapping; ``source_session_id`` is ``None`` until the spawn site
    fills it in."""
    carry = ReadCarryover(records={}, source_session_id=None, generated_at_turn=0)

    assert carry.records == {}
    assert carry.source_session_id is None
    assert carry.generated_at_turn == 0


def test_read_carryover_constructs_with_records(tmp_path: Path) -> None:
    """Round-trip: build a record from a real file, embed in a
    carryover, read it back via ``records[path]``."""
    f = tmp_path / "a.txt"
    f.write_text("hello")

    record = _make_record(f, turn=3)
    carry = ReadCarryover(
        records={f: record},
        source_session_id="parent-abc",
        generated_at_turn=3,
    )

    assert carry.records[f] is record
    assert carry.source_session_id == "parent-abc"
    assert carry.generated_at_turn == 3


def test_is_fresh_true_for_unchanged_file(tmp_path: Path) -> None:
    """The file on disk matches the recorded ``(mtime, size)`` — the
    inherited record is still trustworthy."""
    f = tmp_path / "a.txt"
    f.write_text("hello")

    carry = ReadCarryover(
        records={f: _make_record(f)},
        source_session_id=None,
        generated_at_turn=1,
    )

    assert carry.is_fresh(f) is True


def test_is_fresh_normalizes_lookup_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Records are keyed by resolved paths, but callers may ask with a
    relative path. Normalize before lookup so inheritance matches
    Context.read_status semantics."""
    f = tmp_path / "a.txt"
    f.write_text("hello")
    monkeypatch.chdir(tmp_path)

    carry = ReadCarryover(
        records={f.resolve(): _make_record(f.resolve())},
        source_session_id=None,
        generated_at_turn=1,
    )

    assert carry.is_fresh(Path("a.txt")) is True


def test_is_fresh_false_when_mtime_newer(tmp_path: Path) -> None:
    """Touching the file forwards mtime past the recorded value;
    freshness check must reject the record so the subagent re-reads."""
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record = _make_record(f)
    # Bump mtime forward by 10s — well past any filesystem granularity.
    new_mtime = record.mtime_at_read + 10.0
    os.utime(f, (new_mtime, new_mtime))

    carry = ReadCarryover(
        records={f: record}, source_session_id=None, generated_at_turn=1
    )

    assert carry.is_fresh(f) is False


def test_is_fresh_false_when_mtime_older_even_same_size(tmp_path: Path) -> None:
    """Freshness is exact ``(mtime, size)`` equality. A changed/restored
    file can have the same size and an older mtime; it still must be
    rejected."""
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record = _make_record(f)
    f.write_text("jello")
    older_mtime = record.mtime_at_read - 10.0
    os.utime(f, (older_mtime, older_mtime))

    carry = ReadCarryover(
        records={f: record}, source_session_id=None, generated_at_turn=1
    )

    assert carry.is_fresh(f) is False


def test_is_fresh_false_when_size_differs(tmp_path: Path) -> None:
    """Same mtime but different byte length — the file changed even if
    the OS clock didn't tick. ``is_fresh`` must catch this."""
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record = _make_record(f)
    # Rewrite with a different length, then restore the original mtime
    # so only ``size`` differs from the record.
    f.write_text("hello world")
    os.utime(f, (record.mtime_at_read, record.mtime_at_read))

    carry = ReadCarryover(
        records={f: record}, source_session_id=None, generated_at_turn=1
    )

    assert carry.is_fresh(f) is False


def test_is_fresh_false_when_file_missing(tmp_path: Path) -> None:
    """Parent read it, child arrives, but the file has been deleted —
    no defensible "fresh" answer; return False."""
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record = _make_record(f)
    f.unlink()

    carry = ReadCarryover(
        records={f: record}, source_session_id=None, generated_at_turn=1
    )

    assert carry.is_fresh(f) is False


def test_is_fresh_false_when_path_not_in_records(tmp_path: Path) -> None:
    """A path the parent never read is, by definition, not fresh in the
    carryover. The subagent must read it first."""
    f = tmp_path / "a.txt"
    f.write_text("hello")
    other = tmp_path / "b.txt"
    other.write_text("other")

    carry = ReadCarryover(
        records={f: _make_record(f)}, source_session_id=None, generated_at_turn=1
    )

    assert carry.is_fresh(other) is False


def test_read_carryover_is_frozen() -> None:
    """``frozen=True`` — once handed to a subagent, the carryover's
    fields cannot be silently rebound on the value object."""
    carry = ReadCarryover(records={}, source_session_id=None, generated_at_turn=0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        carry.source_session_id = "mutated"  # type: ignore[misc]


def test_read_record_is_frozen(tmp_path: Path) -> None:
    """``ReadRecord`` is also frozen — the (path, mtime, size, turn)
    tuple is the audit identity; rebinding any field would silently
    rewrite history."""
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record = _make_record(f)

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.size_at_read = 9999  # type: ignore[misc]


def test_records_mapping_rejects_mutation(tmp_path: Path) -> None:
    """``records`` is typed as ``Mapping`` (read-only protocol); the
    runtime container must also refuse mutation so a subagent that
    naively does ``carry.records[p] = ...`` fails loudly instead of
    polluting the parent's view."""
    f = tmp_path / "a.txt"
    f.write_text("hello")
    carry = ReadCarryover(
        records={f: _make_record(f)},
        source_session_id=None,
        generated_at_turn=1,
    )

    with pytest.raises(TypeError):
        carry.records[f] = _make_record(f, turn=99)  # type: ignore[index]
