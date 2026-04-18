"""Tests for aura.core.journal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aura.core import journal as journal_module


@pytest.fixture(autouse=True)
def _reset() -> Any:
    journal_module.reset()
    yield
    journal_module.reset()


def test_default_is_disabled() -> None:
    assert journal_module.is_configured() is False
    journal_module.write("anything", a=1)


def test_configure_enables_writes(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    assert journal_module.is_configured() is True
    journal_module.write("first", foo=1)
    journal_module.write("second", bar="x")
    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "first"
    assert first["foo"] == 1
    assert "ts" in first


def test_reset_disables_writes(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    journal_module.write("pre_reset")
    journal_module.reset()
    assert journal_module.is_configured() is False
    journal_module.write("post_reset")
    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "pre_reset"


def test_write_never_raises_on_serialization_error(tmp_path: Path) -> None:
    journal_module.configure(tmp_path / "events.jsonl")

    class NotJsonable:
        def __repr__(self) -> str:
            raise RuntimeError("explode")

    journal_module.write("bad", obj=NotJsonable())


def test_file_opened_and_closed_per_write(tmp_path: Path) -> None:
    """Using `with` per-event means no lingering fd."""
    journal_module.configure(tmp_path / "events.jsonl")
    for i in range(5):
        journal_module.write(f"event_{i}")

    lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
    assert len(lines) == 5


def test_write_appends_not_overwrites(tmp_path: Path) -> None:
    journal_module.configure(tmp_path / "events.jsonl")
    journal_module.write("a")
    journal_module.write("b")
    journal_module.reset()

    journal_module.configure(tmp_path / "events.jsonl")
    journal_module.write("c")

    lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3
    assert [json.loads(l)["event"] for l in lines] == ["a", "b", "c"]
