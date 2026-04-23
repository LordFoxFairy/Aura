"""Tests for aura.cli.user_question — inline interactive asker.

The production path uses a pt ``Application`` (for multi-choice) or a
``PromptSession`` (for free-text). Tests stub the module-level helpers
so no live terminal is needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from aura.cli.user_question import make_cli_user_asker


def _stub_picker(return_value: str | None) -> Any:
    """Stub for ``_pick_choice_interactive``."""
    captured: dict[str, Any] = {}

    async def _picker(
        question: str,
        options: list[str],
        default: str | None,
        timeout: float | None = None,
    ) -> str | None:
        captured["question"] = question
        captured["options"] = options
        captured["default"] = default
        captured["timeout"] = timeout
        return return_value

    return _picker, captured


def _stub_free_text(return_value: str | None) -> Any:
    """Stub for ``_read_free_text``."""
    captured: dict[str, Any] = {}

    async def _reader(
        question: str,
        default: str | None,
        timeout: float | None = None,
    ) -> str | None:
        captured["question"] = question
        captured["default"] = default
        captured["timeout"] = timeout
        return return_value

    return _reader, captured


def _stub_raises(exc: BaseException) -> Any:
    async def _fail(*_: Any, **__: Any) -> Any:
        raise exc

    return _fail


# ---------------------------------------------------------------------------
# Multi-choice path
# ---------------------------------------------------------------------------
async def test_multi_choice_returns_chosen_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    picker, captured = _stub_picker("green")
    monkeypatch.setattr("aura.cli.user_question._pick_choice_interactive", picker)
    asker = make_cli_user_asker()
    result = await asker("Pick a color", ["red", "green", "blue"], "red")
    assert result == "green"
    assert captured["question"] == "Pick a color"
    assert captured["options"] == ["red", "green", "blue"]
    assert captured["default"] == "red"


async def test_multi_choice_cancel_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Picker returns None → asker returns "" so the LLM sees a
    # well-typed empty string.
    picker, _ = _stub_picker(None)
    monkeypatch.setattr("aura.cli.user_question._pick_choice_interactive", picker)
    asker = make_cli_user_asker()
    result = await asker("Pick", ["a", "b"], "a")
    assert result == ""


async def test_multi_choice_exception_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._pick_choice_interactive",
        _stub_raises(RuntimeError("no tty")),
    )
    asker = make_cli_user_asker()
    result = await asker("Pick", ["a", "b"], "a")
    assert result == ""


async def test_multi_choice_keyboard_interrupt_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._pick_choice_interactive",
        _stub_raises(KeyboardInterrupt()),
    )
    asker = make_cli_user_asker()
    result = await asker("Pick", ["a", "b"], "a")
    assert result == ""


# ---------------------------------------------------------------------------
# Free-text path
# ---------------------------------------------------------------------------
async def test_free_text_returns_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, captured = _stub_free_text("deploy to prod")
    monkeypatch.setattr("aura.cli.user_question._read_free_text", reader)
    asker = make_cli_user_asker()
    result = await asker("What next?", None, None)
    assert result == "deploy to prod"
    assert captured["question"] == "What next?"
    assert captured["default"] is None


async def test_free_text_none_from_reader_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reader returns None on Ctrl+C / EOF → asker normalizes to "".
    reader, _ = _stub_free_text(None)
    monkeypatch.setattr("aura.cli.user_question._read_free_text", reader)
    asker = make_cli_user_asker()
    result = await asker("name?", None, None)
    assert result == ""


async def test_free_text_passes_default_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, captured = _stub_free_text("")
    monkeypatch.setattr("aura.cli.user_question._read_free_text", reader)
    asker = make_cli_user_asker()
    await asker("name?", None, "anon")
    assert captured["default"] == "anon"


# ---------------------------------------------------------------------------
# Timeout — audit Finding A: unattended / stale sessions must not hang the
# turn forever. Non-response resolves to empty string (existing "no answer"
# shape) and writes a journal entry for the audit trail.
# ---------------------------------------------------------------------------
async def test_multi_choice_timeout_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(
        "aura.cli.user_question._pick_choice_interactive",
        _stub_raises(TimeoutError()),
    )
    asker = make_cli_user_asker(timeout=0.1)
    result = await asker("Pick", ["a", "b"], "a")
    assert result == ""


async def test_multi_choice_timeout_writes_journal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    from aura.core.persistence import journal as journal_module

    monkeypatch.setattr(
        "aura.cli.user_question._pick_choice_interactive",
        _stub_raises(TimeoutError()),
    )
    events: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, /, **fields: Any) -> None:
        events.append((event, fields))

    monkeypatch.setattr(journal_module, "write", _capture)
    asker = make_cli_user_asker(timeout=0.1)
    await asker("Pick", ["a", "b"], "a")
    timeout_events = [e for e in events if e[0] == "user_question_timeout"]
    assert len(timeout_events) == 1
    assert timeout_events[0][1]["timeout_sec"] == 0.1
    assert timeout_events[0][1]["kind"] == "choice"


async def test_free_text_timeout_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(
        "aura.cli.user_question._read_free_text",
        _stub_raises(TimeoutError()),
    )
    asker = make_cli_user_asker(timeout=0.1)
    result = await asker("name?", None, None)
    assert result == ""


async def test_free_text_timeout_writes_journal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    from aura.core.persistence import journal as journal_module

    monkeypatch.setattr(
        "aura.cli.user_question._read_free_text",
        _stub_raises(TimeoutError()),
    )
    events: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, /, **fields: Any) -> None:
        events.append((event, fields))

    monkeypatch.setattr(journal_module, "write", _capture)
    asker = make_cli_user_asker(timeout=5.0)
    await asker("name?", None, None)
    timeout_events = [e for e in events if e[0] == "user_question_timeout"]
    assert len(timeout_events) == 1
    assert timeout_events[0][1]["timeout_sec"] == 5.0
    assert timeout_events[0][1]["kind"] == "free_text"


async def test_timeout_none_preserves_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # timeout=None means "wait forever" — the stub still runs normally
    # and returns its preset value; no TimeoutError should ever fire.
    picker, captured = _stub_picker("green")
    monkeypatch.setattr("aura.cli.user_question._pick_choice_interactive", picker)
    asker = make_cli_user_asker(timeout=None)
    result = await asker("Pick", ["red", "green"], None)
    assert result == "green"
    assert captured["timeout"] is None


async def test_timeout_plumbed_through_to_picker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    picker, captured = _stub_picker("a")
    monkeypatch.setattr("aura.cli.user_question._pick_choice_interactive", picker)
    asker = make_cli_user_asker(timeout=42.0)
    await asker("q", ["a", "b"], None)
    assert captured["timeout"] == 42.0


async def test_timeout_plumbed_through_to_free_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, captured = _stub_free_text("ok")
    monkeypatch.setattr("aura.cli.user_question._read_free_text", reader)
    asker = make_cli_user_asker(timeout=7.5)
    await asker("q", None, None)
    assert captured["timeout"] == 7.5
