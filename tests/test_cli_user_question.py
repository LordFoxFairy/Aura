"""Tests for aura.cli.user_question — CLI-side asker for ask_user_question.

Mirrors test_cli_permission's stub pattern: monkeypatch prompt_toolkit's
dialog factories and assert the asker hands through the right kwargs.
"""

from __future__ import annotations

from typing import Any

import pytest


class _StubApp:
    """Stand-in for prompt_toolkit Application returned by ``*_dialog``."""

    def __init__(self, ret: Any) -> None:
        self._ret = ret

    def run_async(self) -> Any:
        async def _coro() -> Any:
            return self._ret

        return _coro()


def _stub_dialog(ret: Any) -> tuple[Any, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> _StubApp:
        captured.update(kwargs)
        return _StubApp(ret)

    return factory, captured


async def test_asker_with_options_delegates_to_radiolist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.cli.user_question import make_cli_user_asker

    factory, captured = _stub_dialog("b")
    monkeypatch.setattr("aura.cli.user_question.radiolist_dialog", factory)
    asker = make_cli_user_asker()
    answer = await asker("Pick one", ["a", "b", "c"], "b")
    assert answer == "b"
    assert captured["values"] == [("a", "a"), ("b", "b"), ("c", "c")]
    assert captured["default"] == "b"
    assert "Pick one" in str(captured.get("text", ""))


async def test_asker_with_options_falls_back_to_first_when_default_not_in_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: default already validated at schema layer, but the CLI
    should still survive being handed a stray default."""
    from aura.cli.user_question import make_cli_user_asker

    factory, captured = _stub_dialog("a")
    monkeypatch.setattr("aura.cli.user_question.radiolist_dialog", factory)
    asker = make_cli_user_asker()
    answer = await asker("Pick", ["a", "b"], "z")
    assert answer == "a"
    assert captured["default"] == "a"


async def test_asker_without_options_delegates_to_input_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.cli.user_question import make_cli_user_asker

    factory, captured = _stub_dialog("freeform answer")
    monkeypatch.setattr("aura.cli.user_question.input_dialog", factory)
    asker = make_cli_user_asker()
    answer = await asker("What?", None, None)
    assert answer == "freeform answer"
    assert "What?" in str(captured.get("text", ""))


async def test_asker_without_options_preserves_default_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.cli.user_question import make_cli_user_asker

    factory, captured = _stub_dialog("ok")
    monkeypatch.setattr("aura.cli.user_question.input_dialog", factory)
    asker = make_cli_user_asker()
    await asker("name?", None, "alice")
    assert captured["default"] == "alice"


async def test_cancelled_radiolist_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.cli.user_question import make_cli_user_asker

    factory, _ = _stub_dialog(None)
    monkeypatch.setattr("aura.cli.user_question.radiolist_dialog", factory)
    asker = make_cli_user_asker()
    answer = await asker("Pick", ["a", "b"], None)
    assert answer == ""


async def test_cancelled_input_dialog_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.cli.user_question import make_cli_user_asker

    factory, _ = _stub_dialog(None)
    monkeypatch.setattr("aura.cli.user_question.input_dialog", factory)
    asker = make_cli_user_asker()
    answer = await asker("Q?", None, None)
    assert answer == ""
