"""Tests for aura.cli.user_question — inline numbered / free-text prompt.

Matches the shape of test_cli_permission: monkeypatch the module-level
``_read_line`` helper so tests don't need a live terminal.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from rich.console import Console

from aura.cli.user_question import (
    _parse_multi_choice,
    _render_multi_choice,
    make_cli_user_asker,
)


def _capture_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    return console, buf


def _stub_read(responses: list[str]) -> Any:
    iterator = iter(responses)

    async def _fake(prompt: str) -> str:  # noqa: ARG001
        try:
            return next(iterator)
        except StopIteration as exc:
            raise EOFError("test responses exhausted") from exc

    return _fake


def _stub_read_raises(exc: BaseException) -> Any:
    async def _fake(prompt: str) -> str:  # noqa: ARG001
        raise exc

    return _fake


# ---------------------------------------------------------------------------
# _parse_multi_choice
# ---------------------------------------------------------------------------
def test_multi_choice_empty_returns_default() -> None:
    assert _parse_multi_choice("", options=["a", "b"], default_idx=2) == "b"


def test_multi_choice_numeric_index() -> None:
    assert _parse_multi_choice("1", options=["a", "b"], default_idx=1) == "a"
    assert _parse_multi_choice("2", options=["a", "b"], default_idx=1) == "b"


def test_multi_choice_out_of_range_index_returns_none() -> None:
    assert _parse_multi_choice("3", options=["a", "b"], default_idx=1) is None
    assert _parse_multi_choice("0", options=["a", "b"], default_idx=1) is None


def test_multi_choice_accepts_exact_option_text_case_insensitive() -> None:
    assert _parse_multi_choice("YES", options=["yes", "no"], default_idx=1) == "yes"
    assert _parse_multi_choice("no", options=["yes", "no"], default_idx=1) == "no"


def test_multi_choice_unknown_text_returns_none() -> None:
    assert _parse_multi_choice("maybe", options=["yes", "no"], default_idx=1) is None


# ---------------------------------------------------------------------------
# _render_multi_choice
# ---------------------------------------------------------------------------
def test_render_multi_choice_prints_question_and_numbered_options() -> None:
    console, buf = _capture_console()
    default_idx = _render_multi_choice(
        console, "Pick a color", ["red", "green", "blue"], default="green",
    )
    out = buf.getvalue()
    assert "Pick a color" in out
    assert "1. red" in out
    assert "2. green" in out
    assert "3. blue" in out
    assert default_idx == 2


def test_render_multi_choice_without_default_still_returns_one() -> None:
    console, _ = _capture_console()
    idx = _render_multi_choice(
        console, "Pick", ["a", "b"], default=None,
    )
    assert idx == 1


def test_render_multi_choice_no_box_border_chars() -> None:
    console, buf = _capture_console()
    _render_multi_choice(console, "q?", ["a", "b"], default="a")
    out = buf.getvalue()
    for glyph in ("╭", "╮", "╰", "╯", "│", "─"):
        assert glyph not in out


# ---------------------------------------------------------------------------
# make_cli_user_asker — options path
# ---------------------------------------------------------------------------
async def test_asker_with_options_returns_numeric_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._read_line", _stub_read(["2"]),
    )
    console, _ = _capture_console()
    asker = make_cli_user_asker(console=console)
    result = await asker("Pick a color", ["red", "green", "blue"], "red")
    assert result == "green"


async def test_asker_with_options_empty_returns_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._read_line", _stub_read([""]),
    )
    console, _ = _capture_console()
    asker = make_cli_user_asker(console=console)
    result = await asker("Pick", ["a", "b", "c"], "b")
    assert result == "b"


async def test_asker_with_options_accepts_exact_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._read_line", _stub_read(["blue"]),
    )
    console, _ = _capture_console()
    asker = make_cli_user_asker(console=console)
    result = await asker("Pick", ["red", "green", "blue"], "red")
    assert result == "blue"


async def test_asker_with_options_reprompts_on_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._read_line",
        _stub_read(["99", "nope", "2"]),
    )
    console, buf = _capture_console()
    asker = make_cli_user_asker(console=console)
    result = await asker("Pick", ["a", "b"], "a")
    assert result == "b"
    assert "Please enter" in buf.getvalue()


async def test_asker_with_options_ctrl_c_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._read_line", _stub_read_raises(KeyboardInterrupt()),
    )
    console, _ = _capture_console()
    asker = make_cli_user_asker(console=console)
    result = await asker("Pick", ["a", "b"], "a")
    assert result == ""


# ---------------------------------------------------------------------------
# make_cli_user_asker — free-text path
# ---------------------------------------------------------------------------
async def test_asker_free_text_returns_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._read_line", _stub_read(["deploy to prod"]),
    )
    console, _ = _capture_console()
    asker = make_cli_user_asker(console=console)
    result = await asker("What next?", None, None)
    assert result == "deploy to prod"


async def test_asker_free_text_empty_returns_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._read_line", _stub_read([""]),
    )
    console, _ = _capture_console()
    asker = make_cli_user_asker(console=console)
    result = await asker("name?", None, "anon")
    assert result == "anon"


async def test_asker_free_text_empty_no_default_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._read_line", _stub_read([""]),
    )
    console, _ = _capture_console()
    asker = make_cli_user_asker(console=console)
    result = await asker("name?", None, None)
    assert result == ""


async def test_asker_free_text_ctrl_c_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._read_line", _stub_read_raises(KeyboardInterrupt()),
    )
    console, _ = _capture_console()
    asker = make_cli_user_asker(console=console)
    result = await asker("name?", None, None)
    assert result == ""


async def test_asker_free_text_eof_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aura.cli.user_question._read_line", _stub_read_raises(EOFError()),
    )
    console, _ = _capture_console()
    asker = make_cli_user_asker(console=console)
    result = await asker("name?", None, None)
    assert result == ""
