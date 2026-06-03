"""Tests for ``ReadRecord`` and ``ReadCarryover``.

Phase 3 Task 1 — schema contracts only. ``ReadCarryover`` replaces the
ad-hoc ``inherited_reads: dict[Path, _ReadRecord]`` propagation that
the subagent factory uses to share parent file-read records with a
spawned child. This file pins the schema invariants in isolation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from aura.domain.state_values import ReadCarryover, ReadRecord


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


def test_read_carryover_is_frozen() -> None:
    """``frozen=True`` — once handed to a subagent, the carryover's
    fields cannot be silently rebound on the value object."""
    carry = ReadCarryover(records={}, source_session_id=None, generated_at_turn=0)

    carry_obj: Any = carry
    with pytest.raises(dataclasses.FrozenInstanceError):
        carry_obj.source_session_id = "mutated"


def test_read_record_is_frozen(tmp_path: Path) -> None:
    """``ReadRecord`` is also frozen — the (path, mtime, size, turn)
    tuple is the audit identity; rebinding any field would silently
    rewrite history."""
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record = _make_record(f)

    record_obj: Any = record
    with pytest.raises(dataclasses.FrozenInstanceError):
        record_obj.size_at_read = 9999


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
        records_any: Any = carry.records
        records_any[f] = _make_record(f, turn=99)
