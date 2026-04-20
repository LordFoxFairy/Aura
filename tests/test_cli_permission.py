"""Tests for aura.cli.permission — list-select prompt (Task 8).

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §8.1–§8.5.

The asker has ONE job: present the choice, capture the answer. It does NOT
decide, persist, or emit domain events — those are the hook's and store's
jobs. These tests pin that boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from rich.console import Console

from aura.cli.permission import make_cli_asker, print_bypass_banner
from aura.core import journal as journal_module
from aura.core.hooks.permission import AskerResponse
from aura.core.permissions.matchers import exact_match_on
from aura.core.permissions.rule import Rule
from aura.tools.base import build_tool


class _P(BaseModel):
    command: str = ""
    msg: str = ""


def _noop(**_: Any) -> dict[str, Any]:
    return {}


def _bash_like(
    *,
    is_destructive: bool = False,
    is_read_only: bool = False,
    with_matcher: bool = True,
    with_preview: bool = True,
) -> BaseTool:
    return build_tool(
        name="bash",
        description="bash-like tool",
        args_schema=_P,
        func=_noop,
        is_destructive=is_destructive,
        is_read_only=is_read_only,
        rule_matcher=exact_match_on("command") if with_matcher else None,
        args_preview=(
            (lambda args: f"command: {args.get('command', '')}")
            if with_preview
            else None
        ),
    )


_HINT = Rule(tool="bash", content=None)  # hint arg is unused now; kept for signature


class _StubApp:
    """Minimal stand-in for prompt_toolkit's ``Application`` returned by
    ``radiolist_dialog``. Exposes ``run_async()`` returning an awaitable.
    """

    def __init__(self, ret: Any) -> None:
        self._ret = ret

    def run_async(self) -> Any:
        async def _coro() -> Any:
            return self._ret

        return _coro()


def _stub_dialog(ret: Any) -> tuple[Any, dict[str, Any]]:
    """Return (factory, captured_kwargs_dict). Installing the factory over
    ``aura.cli.permission.radiolist_dialog`` captures the call kwargs for
    assertion on shape/default/text without touching a terminal.
    """
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> _StubApp:
        captured.update(kwargs)
        return _StubApp(ret)

    return factory, captured


@pytest.fixture
def _journal_capture(tmp_path: Path) -> Any:
    """Redirect journal to a tmp file; yield path so tests can read events."""
    log = tmp_path / "journal.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        yield log
    finally:
        journal_module.reset()


def _events(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def _text_of(captured: dict[str, Any]) -> str:
    """Return a plain-string view of whatever ``text=`` the dialog received.

    prompt_toolkit accepts ``str`` | ``HTML`` | tuple-list formatted text;
    we coerce to str so tests can do substring checks without caring which.
    """
    raw = captured.get("text", "")
    if isinstance(raw, str):
        return raw
    # HTML has a .value attribute; FormattedText is a list of (style, text) tuples.
    if hasattr(raw, "value"):
        return str(raw.value)
    return str(raw)


async def test_accept_returns_accept(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    factory, _ = _stub_dialog(1)
    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", factory)
    asker = make_cli_asker()
    resp = await asker(
        tool=_bash_like(), args={"command": "npm test"}, rule_hint=_HINT,
    )
    assert isinstance(resp, AskerResponse)
    assert resp.choice == "accept"
    assert resp.rule is None
    events = {e["event"] for e in _events(_journal_capture)}
    assert "permission_asked" in events
    assert "permission_answered" in events


async def test_deny_returns_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    factory, _ = _stub_dialog(3)
    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", factory)
    asker = make_cli_asker()
    resp = await asker(
        tool=_bash_like(), args={"command": "rm -rf /"}, rule_hint=_HINT,
    )
    assert resp.choice == "deny"


async def test_esc_none_returns_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    factory, _ = _stub_dialog(None)
    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", factory)
    asker = make_cli_asker()
    resp = await asker(
        tool=_bash_like(), args={"command": "x"}, rule_hint=_HINT,
    )
    assert resp.choice == "deny"


async def test_always_with_precise_rule_is_project_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, _ = _stub_dialog(2)
    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", factory)
    asker = make_cli_asker()
    resp = await asker(
        tool=_bash_like(with_matcher=True),
        args={"command": "npm test"},
        rule_hint=_HINT,
    )
    assert resp.choice == "always"
    assert resp.scope == "project"
    assert resp.rule == Rule(tool="bash", content="npm test")


async def test_always_fallback_to_session_when_no_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, _ = _stub_dialog(2)
    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", factory)
    asker = make_cli_asker()
    tool = _bash_like(with_matcher=False)
    resp = await asker(tool=tool, args={"command": "anything"}, rule_hint=_HINT)
    assert resp.choice == "always"
    assert resp.scope == "session"
    assert resp.rule == Rule(tool="bash", content=None)


async def test_destructive_default_cursor_is_option_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, captured = _stub_dialog(3)
    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", factory)
    asker = make_cli_asker()
    await asker(
        tool=_bash_like(is_destructive=True),
        args={"command": "rm"},
        rule_hint=_HINT,
    )
    assert captured["default"] == 3


async def test_nondestructive_default_cursor_is_option_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, captured = _stub_dialog(1)
    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", factory)
    asker = make_cli_asker()
    await asker(
        tool=_bash_like(is_destructive=False),
        args={"command": "ls"},
        rule_hint=_HINT,
    )
    assert captured["default"] == 1


async def test_title_and_preview_in_dialog_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, captured = _stub_dialog(1)
    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", factory)
    asker = make_cli_asker()
    await asker(
        tool=_bash_like(),
        args={"command": "npm test"},
        rule_hint=_HINT,
    )
    # The title kwarg holds the tool name; the text kwarg holds the preview.
    title_str = str(captured.get("title", ""))
    text_str = _text_of(captured)
    combined = title_str + "\n" + text_str
    assert "bash" in combined
    assert "command: npm test" in combined


async def test_no_tty_raises_exception_returns_deny(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    def _boom(**_: Any) -> _StubApp:
        raise RuntimeError("no tty")

    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", _boom)
    asker = make_cli_asker()
    resp = await asker(tool=_bash_like(), args={"command": "x"}, rule_hint=_HINT)
    assert resp.choice == "deny"
    events = {e["event"] for e in _events(_journal_capture)}
    assert "permission_prompt_unavailable" in events


async def test_keyboard_interrupt_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_: Any) -> _StubApp:
        raise KeyboardInterrupt()

    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", _boom)
    asker = make_cli_asker()
    with pytest.raises(KeyboardInterrupt):
        await asker(tool=_bash_like(), args={"command": "x"}, rule_hint=_HINT)


async def test_missing_args_preview_falls_back_to_tool_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, captured = _stub_dialog(1)
    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", factory)
    asker = make_cli_asker()
    tool = _bash_like(with_preview=False)
    await asker(tool=tool, args={"command": "x"}, rule_hint=_HINT)
    combined = str(captured.get("title", "")) + "\n" + _text_of(captured)
    # Must include the tool name — no crash, no garbage.
    assert "bash" in combined


async def test_destructive_tag_in_text(monkeypatch: pytest.MonkeyPatch) -> None:
    factory, captured = _stub_dialog(3)
    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", factory)
    asker = make_cli_asker()
    await asker(
        tool=_bash_like(is_destructive=True),
        args={"command": "rm"},
        rule_hint=_HINT,
    )
    combined = str(captured.get("title", "")) + "\n" + _text_of(captured)
    assert "destructive" in combined


async def test_read_only_tag_in_text(monkeypatch: pytest.MonkeyPatch) -> None:
    factory, captured = _stub_dialog(1)
    monkeypatch.setattr("aura.cli.permission.radiolist_dialog", factory)
    asker = make_cli_asker()
    await asker(
        tool=_bash_like(is_read_only=True),
        args={"command": "cat x"},
        rule_hint=_HINT,
    )
    combined = str(captured.get("title", "")) + "\n" + _text_of(captured)
    assert "read-only" in combined


def test_print_bypass_banner_writes_warning() -> None:
    import io

    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=120)
    print_bypass_banner(console)
    out = buffer.getvalue()
    assert "PERMISSION CHECKS DISABLED" in out
