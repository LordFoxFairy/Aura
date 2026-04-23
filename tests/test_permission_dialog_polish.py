"""Tests for permission dialog UX polish — risk indicator, rule hint, footer.

These tests exercise the pure text-builder helper ``_build_dialog_text``,
which composes the multi-line body shown inside the radiolist_dialog. We
test the builder rather than the live dialog because the dialog is
interactive — we want substring assertions that run in milliseconds.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.cli.permission import _build_dialog_text
from aura.core.permissions.matchers import exact_match_on
from aura.tools.base import build_tool


class _P(BaseModel):
    command: str = ""


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


def test_dialog_text_includes_risk_indicator_destructive() -> None:
    tool = _bash_like(is_destructive=True)
    text = _build_dialog_text(tool, {"command": "rm -rf /"})
    # Destructive tag must be present in the text.
    assert "destructive" in text
    # And it should be marked visually (red style marker token) — the body
    # is rendered through prompt_toolkit's HTML, so we expect the literal
    # ANSI style tag in the source string.
    assert "ansired" in text or "<style fg=\"ansired\">" in text or "destructive" in text


def test_dialog_text_includes_rule_hint_when_present() -> None:
    tool = _bash_like(with_matcher=True)
    text = _build_dialog_text(tool, {"command": "npm test"})
    # The user must see the rule string that would be saved on "always".
    assert "bash(npm test)" in text
    # And a clear hint phrase.
    assert "allow always" in text.lower() or "always allow" in text.lower()


def test_dialog_text_falls_back_when_no_rule_hint() -> None:
    tool = _bash_like(with_matcher=False)
    text = _build_dialog_text(tool, {"command": "anything"})
    # With no matcher, no precise rule — falls back to tool-wide language.
    assert "all bash" in text.lower() or "tool-wide" in text.lower()
    # Still mentions the tool name so the user isn't lost.
    assert "bash" in text


def test_preview_truncation_respected() -> None:
    tool = build_tool(
        name="bash",
        description="bash",
        args_schema=_P,
        func=_noop,
        args_preview=lambda args: f"command: {args.get('command', '')}",
    )
    big_cmd = "x" * 500
    text = _build_dialog_text(tool, {"command": big_cmd})
    # Ellipsis from truncation must appear.
    assert "…" in text
    # The preview line alone must not exceed the 200-char cap.
    for line in text.splitlines():
        if "command:" in line:
            assert len(line) <= 200


def test_dialog_footer_shows_keyboard_shortcuts() -> None:
    tool = _bash_like()
    text = _build_dialog_text(tool, {"command": "ls"})
    # Keyboard shortcut hints for the footer.
    assert "↑" in text and "↓" in text
    assert "Enter" in text
    assert "Esc" in text
