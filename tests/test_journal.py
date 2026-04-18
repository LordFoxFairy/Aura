"""Tests for aura.core.journal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aura.core.journal import (
    journal,
    reset_journal,
    set_journal,
    setup_file_journal,
)


@pytest.fixture(autouse=True)
def _reset() -> Any:
    yield
    reset_journal()


def test_default_is_noop() -> None:
    reset_journal()
    journal().write("anything", a=1)


def test_setup_file_journal_writes_jsonl(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    j = setup_file_journal(log)
    journal().write("first_event", foo=1)
    journal().write("second_event", bar="x")
    j.close()

    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "first_event"
    assert first["foo"] == 1
    assert "ts" in first


def test_file_journal_never_raises_on_serialization_error(tmp_path: Path) -> None:
    j = setup_file_journal(tmp_path / "events.jsonl")

    class NotJsonable:
        def __repr__(self) -> str:
            raise RuntimeError("explode")

    journal().write("bad_payload", obj=NotJsonable())
    j.close()


def test_file_journal_fsyncs_on_each_write(tmp_path: Path) -> None:
    """Verify entries are durable — write N events, read back, count lines."""
    log = tmp_path / "events.jsonl"
    j = setup_file_journal(log)
    for i in range(5):
        journal().write(f"event_{i}")
    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5
    j.close()


def test_set_journal_allows_recording_fake() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def write(self, event: str, /, **fields: object) -> None:
            self.calls.append((event, dict(fields)))

        def close(self) -> None:
            return

    rec = Recorder()
    set_journal(rec)
    journal().write("hello", x=1)
    assert rec.calls == [("hello", {"x": 1})]
