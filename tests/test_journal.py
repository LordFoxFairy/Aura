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
    # Before configure(), write() is a silent no-op — should not raise.
    journal_module.write("anything", a=1)


def test_configure_enables_writes(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
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
    assert [json.loads(line)["event"] for line in lines] == ["a", "b", "c"]


def test_configure_does_not_raise_on_unwritable_parent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # Contract: the journal's documented invariant is that audit failures
    # never crash the agent. Before this test, write() honored that but
    # configure() could raise OSError if mkdir failed (e.g. read-only FS).
    # End-to-end the promise was broken — if configure raised, you never
    # reached write(). Close the gap at the module boundary.
    existing_file = tmp_path / "blocker"
    existing_file.write_text("x")
    # Can't mkdir a child *under a file* — mkdir raises NotADirectoryError
    # (an OSError subclass). configure must swallow it.
    unreachable = existing_file / "nested" / "events.jsonl"
    journal_module.configure(unreachable)

    # State must be disabled — subsequent writes are no-ops.
    journal_module.write("post_failed_configure")
    assert not unreachable.exists()

    # Stderr carries a specific one-line warning so the user knows audit
    # is off rather than silently losing events. Match the canonical phrase
    # rather than loose substrings — a future refactor that changes the
    # message shape (e.g. "failed to log" → different user experience)
    # should force this test to update deliberately.
    captured = capsys.readouterr()
    assert "audit log disabled" in captured.err


def test_configure_failure_does_not_poison_later_configure(
    tmp_path: Path,
) -> None:
    # A failed configure leaves the module in disabled state — a subsequent
    # configure() to a good path must restore normal operation.
    existing_file = tmp_path / "blocker"
    existing_file.write_text("x")
    journal_module.configure(existing_file / "nope" / "events.jsonl")

    good_log = tmp_path / "good.jsonl"
    journal_module.configure(good_log)
    journal_module.write("recovered")

    lines = good_log.read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "recovered"
