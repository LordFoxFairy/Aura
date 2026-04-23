"""Tests for aura.cli.permission_write — write/edit-specialized widget.

Covers:
* dispatch routing (write_file / edit_file → write widget)
* edit_file diff fragments contain - / + markers
* write_file preview shows size + first N lines
* full 4-option round-trip via driven pipe input
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from pydantic import BaseModel
from rich.console import Console

from aura.cli import permission_write
from aura.cli.permission_write import build_diff_fragments, build_write_preview
from aura.core.permissions.rule import Rule
from aura.tools.base import build_tool


class _WriteP(BaseModel):
    path: str = ""
    content: str = ""


class _EditP(BaseModel):
    path: str = ""
    old_str: str = ""
    new_str: str = ""


def _noop(**_: Any) -> dict[str, Any]:
    return {}


def _write_tool() -> Any:
    return build_tool(
        name="write_file",
        description="Write a file",
        args_schema=_WriteP,
        func=_noop,
        is_destructive=True,
    )


def _edit_tool() -> Any:
    return build_tool(
        name="edit_file",
        description="Edit a file in place",
        args_schema=_EditP,
        func=_noop,
        is_destructive=True,
    )


def _fragment_text(frags: list[tuple[str, str]]) -> str:
    return "".join(text for _style, text in frags)


# ---------------------------------------------------------------------------
# build_diff_fragments — unified-diff rendering
# ---------------------------------------------------------------------------
def test_diff_has_minus_and_plus_markers() -> None:
    frags = build_diff_fragments("a\nb\nc\n", "a\nB\nc\n", "foo.py")
    text = _fragment_text(frags)
    assert "-b" in text
    assert "+B" in text


def test_diff_colours_minus_and_plus_lines() -> None:
    # Minus lines carry the ansired style; plus lines ansigreen.
    frags = build_diff_fragments("a\nb\n", "a\nB\n", "foo.py")
    styles_with_minus = [
        s for s, t in frags
        if t.lstrip().startswith("-") and not t.lstrip().startswith("---")
    ]
    styles_with_plus = [
        s for s, t in frags
        if t.lstrip().startswith("+") and not t.lstrip().startswith("+++")
    ]
    assert any("ansired" in s for s in styles_with_minus)
    assert any("ansigreen" in s for s in styles_with_plus)


def test_diff_identical_strings_show_no_change() -> None:
    frags = build_diff_fragments("same\n", "same\n", "foo.py")
    text = _fragment_text(frags)
    assert "(no change)" in text


def test_diff_truncates_huge_diff() -> None:
    old = "\n".join(f"old line {i}" for i in range(200))
    new = "\n".join(f"new line {i}" for i in range(200))
    frags = build_diff_fragments(old, new, "big.py")
    text = _fragment_text(frags)
    assert "more diff lines" in text


# ---------------------------------------------------------------------------
# build_write_preview — size + head rendering
# ---------------------------------------------------------------------------
def test_write_preview_shows_byte_count() -> None:
    content = "hello\nworld\n"
    frags = build_write_preview(content)
    text = _fragment_text(frags)
    assert f"{len(content.encode('utf-8'))} bytes" in text


def test_write_preview_shows_first_lines() -> None:
    content = "line1\nline2\nline3\n"
    frags = build_write_preview(content)
    text = _fragment_text(frags)
    assert "line1" in text
    assert "line2" in text
    assert "line3" in text


def test_write_preview_truncates_head_after_five_lines() -> None:
    content = "\n".join(f"L{i}" for i in range(20)) + "\n"
    frags = build_write_preview(content)
    text = _fragment_text(frags)
    # First 5 lines shown.
    for i in range(5):
        assert f"L{i}" in text
    # Tail elided footer.
    assert "more lines" in text
    # Lines past the window aren't shown.
    assert "L19" not in text


def test_write_preview_line_count_reported() -> None:
    content = "a\nb\nc\nd\ne\nf\n"
    frags = build_write_preview(content)
    text = _fragment_text(frags)
    assert "6 lines" in text


def test_write_preview_wraps_single_long_line() -> None:
    # A pathological 500-char single line should be truncated, not
    # wrapped across 50 rows.
    content = "x" * 500
    frags = build_write_preview(content)
    text = _fragment_text(frags)
    assert "…" in text


# ---------------------------------------------------------------------------
# Dispatch routing — write_file / edit_file reach the write widget
# ---------------------------------------------------------------------------
async def test_dispatch_routes_write_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, int] = {"bash": 0, "generic": 0, "write": 0}

    async def fake_bash(**_kw: Any) -> tuple[int | None, str]:
        calls["bash"] += 1
        return 1, ""

    async def fake_generic(**_kw: Any) -> tuple[int | None, str]:
        calls["generic"] += 1
        return 1, ""

    async def fake_write(**_kw: Any) -> tuple[int | None, str]:
        calls["write"] += 1
        return 1, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "run_bash_permission", fake_bash)
    monkeypatch.setattr(perm_mod, "run_generic_permission", fake_generic)
    monkeypatch.setattr(perm_mod, "run_write_permission", fake_write)

    console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
    asker = perm_mod.make_cli_asker(console=console)
    await asker(
        tool=_write_tool(),
        args={"path": "foo.py", "content": "hello\n"},
        rule_hint=Rule(tool="write_file", content=None),
    )
    assert calls == {"bash": 0, "generic": 0, "write": 1}


async def test_dispatch_routes_edit_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, int] = {"bash": 0, "generic": 0, "write": 0}

    async def fake_bash(**_kw: Any) -> tuple[int | None, str]:
        calls["bash"] += 1
        return 1, ""

    async def fake_generic(**_kw: Any) -> tuple[int | None, str]:
        calls["generic"] += 1
        return 1, ""

    async def fake_write(**_kw: Any) -> tuple[int | None, str]:
        calls["write"] += 1
        return 1, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "run_bash_permission", fake_bash)
    monkeypatch.setattr(perm_mod, "run_generic_permission", fake_generic)
    monkeypatch.setattr(perm_mod, "run_write_permission", fake_write)

    console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
    asker = perm_mod.make_cli_asker(console=console)
    await asker(
        tool=_edit_tool(),
        args={"path": "foo.py", "old_str": "a", "new_str": "b"},
        rule_hint=Rule(tool="edit_file", content=None),
    )
    assert calls == {"bash": 0, "generic": 0, "write": 1}


# ---------------------------------------------------------------------------
# End-to-end via pipe input — full round-trip with real key bindings
# ---------------------------------------------------------------------------
async def _drive_write(
    tool: Any, args: dict[str, Any], keys: str,
) -> tuple[int | None, str]:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as inp:
        inp.send_text(keys)
        with create_app_session(input=inp, output=DummyOutput()):
            return await permission_write.run_write_permission(
                tool=tool,
                args=args,
                tag="destructive",
                option_two_label="Yes, always this session",
                default_choice=3,
            )


async def test_write_widget_yes_commits() -> None:
    # Cursor starts on 3 for destructive; press "1" to commit Yes.
    choice, feedback = await _drive_write(
        _write_tool(), {"path": "foo.py", "content": "x"}, "1",
    )
    assert choice == 1
    assert feedback == ""


async def test_write_widget_no_commits() -> None:
    # Default = 3 (destructive) → Enter commits No.
    choice, feedback = await _drive_write(
        _write_tool(), {"path": "foo.py", "content": "x"}, "\r",
    )
    assert choice == 3
    assert feedback == ""


async def test_write_widget_always_commits() -> None:
    choice, feedback = await _drive_write(
        _write_tool(), {"path": "foo.py", "content": "x"}, "2",
    )
    assert choice == 2


async def test_write_widget_esc_cancels() -> None:
    choice, feedback = await _drive_write(
        _write_tool(), {"path": "foo.py", "content": "x"}, "\x1b",
    )
    assert choice is None


async def test_write_widget_tab_feedback_roundtrip() -> None:
    # Tab → feedback mode; type "oops"; Enter → commit default (3) with fb.
    choice, feedback = await _drive_write(
        _write_tool(), {"path": "foo.py", "content": "x"}, "\toops\r",
    )
    assert choice == 3
    assert feedback == "oops"


async def test_edit_widget_drives_with_diff_present() -> None:
    # edit_file path — renders a diff, then commits via "1".
    choice, feedback = await _drive_write(
        _edit_tool(),
        {"path": "foo.py", "old_str": "alpha", "new_str": "beta"},
        "1",
    )
    assert choice == 1
    assert feedback == ""
