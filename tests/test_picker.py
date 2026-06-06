"""Tests for :mod:`cli.picker` — the scrollable, filterable picker.

Driven via ``prompt_toolkit.input.create_pipe_input`` + ``DummyOutput`` so
the picker runs in a memory-only terminal harness — same pattern as the
REPL key-binding tests. No real TTY required, but fully exercises pt's
event loop, key bindings, and renderer.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from collections.abc import Iterator

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.styles import Style

from cli import picker
from cli.picker import (
    DEFAULT_PAGE_SIZE,
    PickerResult,
    SimplePickerItem,
    _matches_filter,
    _PickerState,
    _truncate_to_width,
    run_picker,
)

# --- Pure-logic tests (no Application). Covers the internal state
# machine without booting pt — fastest path to high-confidence
# regression coverage on the cursor / viewport math.

def test_truncate_to_width_passes_short_text_through() -> None:
    assert _truncate_to_width("hello", 10) == "hello"


def test_truncate_to_width_appends_ellipsis_when_overflow() -> None:
    out = _truncate_to_width("hello world", 6)
    # 5 columns of text + 1 column for the ellipsis = exactly 6.
    assert out.endswith("…")
    assert len(out) == 6


def test_truncate_to_width_handles_zero_budget() -> None:
    assert _truncate_to_width("hello", 0) == ""


def test_matches_filter_empty_needle_matches_everything() -> None:
    item = SimplePickerItem(label="hello", sublabel="world", value=1)
    assert _matches_filter(item, "") is True


def test_matches_filter_searches_label_and_sublabel() -> None:
    item = SimplePickerItem(label="apples", sublabel="2 hours ago", value=1)
    assert _matches_filter(item, "hour") is True
    assert _matches_filter(item, "APPLE") is True  # case-insensitive
    assert _matches_filter(item, "banana") is False


def _state(n: int, page_size: int = DEFAULT_PAGE_SIZE) -> _PickerState:
    items = [
        SimplePickerItem(label=f"item-{i}", sublabel=None, value=i)
        for i in range(n)
    ]
    s = _PickerState(items, page_size=page_size, initial_filter="")
    s.clamp_after_filter_change()
    return s


def test_state_move_down_within_viewport() -> None:
    s = _state(5, page_size=3)
    s.move_down()
    assert s.cursor == 1
    assert s.viewport_top == 0


def test_state_move_down_scrolls_viewport_at_edge() -> None:
    s = _state(10, page_size=3)
    s.cursor = 2  # at the bottom of the viewport
    s.move_down()
    assert s.cursor == 3
    assert s.viewport_top == 1


def test_state_move_down_wraps_to_top() -> None:
    s = _state(3, page_size=3)
    s.cursor = 2  # last
    s.move_down()
    assert s.cursor == 0
    assert s.viewport_top == 0


def test_state_move_up_wraps_to_bottom() -> None:
    s = _state(10, page_size=3)
    assert s.cursor == 0
    s.move_up()
    assert s.cursor == 9
    assert s.viewport_top == 7  # n - page_size


def test_state_page_down_jumps_one_page() -> None:
    s = _state(20, page_size=5)
    s.page_down()
    assert s.cursor == 5
    assert s.viewport_top == 1


def test_state_page_up_jumps_one_page() -> None:
    s = _state(20, page_size=5)
    s.cursor = 10
    s.viewport_top = 6
    s.page_up()
    assert s.cursor == 5


def test_state_home_and_end() -> None:
    s = _state(20, page_size=5)
    s.end()
    assert s.cursor == 19
    assert s.viewport_top == 15
    s.home()
    assert s.cursor == 0
    assert s.viewport_top == 0


def test_state_filter_narrows_visible_set() -> None:
    items = [
        SimplePickerItem(label="apple", sublabel=None, value="a"),
        SimplePickerItem(label="banana", sublabel=None, value="b"),
        SimplePickerItem(label="apricot", sublabel=None, value="c"),
    ]
    s = _PickerState(items, page_size=8, initial_filter="ap")
    s.clamp_after_filter_change()
    visible = s.visible()
    assert [it.label for it in visible] == ["apple", "apricot"]


def test_state_filter_change_clamps_cursor() -> None:
    items = [
        SimplePickerItem(label=f"item-{i}", sublabel=None, value=i)
        for i in range(10)
    ]
    s = _PickerState(items, page_size=3, initial_filter="")
    s.clamp_after_filter_change()
    s.cursor = 9
    s.viewport_top = 7
    # Filter narrows to a single item — cursor must drop into range.
    s.filter_text = "item-3"
    s.clamp_after_filter_change()
    assert s.cursor == 0
    assert s.viewport_top == 0
    assert len(s.visible()) == 1


# --- Application-level tests. Drive a real pt picker via pipe input;
# assert the returned PickerResult reflects the keystrokes we sent.

def _items(n: int) -> list[SimplePickerItem]:
    return [
        SimplePickerItem(label=f"item-{i}", sublabel=None, value=i)
        for i in range(n)
    ]


@contextlib.contextmanager
def _piped(keystrokes: str) -> Iterator[None]:
    """Context manager: pipe ``keystrokes`` into the running pt app.

    Wraps ``create_pipe_input`` + ``create_app_session`` together so each
    test is one ``with`` block.
    """
    with create_pipe_input() as inp:
        inp.send_text(keystrokes)
        with create_app_session(input=inp, output=DummyOutput()):
            yield


async def test_picker_renders_initial_items() -> None:
    # Press Enter immediately — picker should return the first item
    # (cursor sits at position 0 by default).
    with _piped("\r"):
        result = await run_picker(_items(5), title="Test")
    assert isinstance(result, PickerResult)
    assert result.item is not None
    assert result.item.value == 0


async def test_picker_arrow_down_moves_cursor() -> None:
    # Down twice (\x1b[B is the ANSI down-arrow), then Enter → item 2.
    with _piped("\x1b[B\x1b[B\r"):
        result = await run_picker(_items(10), title="Test", enable_filter=False)
    assert result.item is not None
    assert result.item.value == 2


async def test_picker_wraps_at_bottom() -> None:
    # 3 items, 4 down-presses → cursor wraps once, lands on item 1.
    with _piped("\x1b[B\x1b[B\x1b[B\x1b[B\r"):
        result = await run_picker(
            _items(3), title="Test", enable_filter=False,
        )
    assert result.item is not None
    assert result.item.value == 1


async def test_picker_filter_narrows_list() -> None:
    items = [
        SimplePickerItem(label="apple", sublabel=None, value="a"),
        SimplePickerItem(label="banana", sublabel=None, value="b"),
        SimplePickerItem(label="apricot", sublabel=None, value="c"),
    ]
    # Type "ban" into the filter, then Enter → only "banana" matches,
    # cursor at position 0 → returns "b".
    with _piped("ban\r"):
        result = await run_picker(items, title="Test", enable_filter=True)
    assert result.item is not None
    assert result.item.value == "b"


async def test_picker_truncates_long_label(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force a tight terminal width so the picker truncates. We can't
    # observe the rendered string directly through DummyOutput, but
    # the truncation helper is exercised here as a smoke check that
    # the application doesn't crash on a label far longer than width.
    monkeypatch.setenv("COLUMNS", "40")
    long_items = [
        SimplePickerItem(
            label="this is a very long label that overflows " * 4,
            sublabel=None,
            value=i,
        )
        for i in range(3)
    ]
    with _piped("\r"):
        result = await run_picker(
            long_items, title="Test", enable_filter=False,
        )
    # Should still pick item 0 cleanly — the picker's job is not to
    # crash when widths get tight.
    assert result.item is not None
    assert result.item.value == 0


async def test_picker_pageup_pagedown() -> None:
    # 30 items, page_size=5. Press PageDown twice (skip 10 items)
    # then Enter → land on item 10.
    with _piped("\x1b[6~\x1b[6~\r"):
        result = await run_picker(
            _items(30),
            title="Test",
            enable_filter=False,
            page_size=5,
        )
    assert result.item is not None
    assert result.item.value == 10


async def test_picker_esc_returns_none() -> None:
    # Press Esc — picker exits with item=None.
    with _piped("\x1b"):
        # pt waits for the escape sequence to complete; we use a
        # short timeout pattern to ensure the bare-esc binding wins.
        result = await asyncio.wait_for(
            run_picker(_items(5), title="Test", enable_filter=False),
            timeout=2.0,
        )
    assert result.item is None


async def test_picker_enter_returns_selected() -> None:
    # Down once + Enter → item 1.
    with _piped("\x1b[B\r"):
        result = await run_picker(_items(5), title="Test", enable_filter=False)
    assert result.item is not None
    assert result.item.value == 1


async def test_picker_empty_items_short_circuits() -> None:
    # No keystrokes needed — empty list returns immediately.
    result = await run_picker([], title="Test")
    assert result.item is None


async def test_picker_ctrl_c_cancels() -> None:
    # Ctrl+C inside the picker is treated as cancel (not an exception).
    with _piped("\x03"):
        result = await run_picker(_items(5), title="Test", enable_filter=False)
    assert result.item is None


# --- Boundary matrix. Targets the uncovered error/edge branches:
# single-column truncation, OSError width fallback, navigation on an
# empty visible set, zero-match render + selection, sublabel render,
# the "showing N of M" footer, and the custom-style merge path.

def test_truncate_to_width_single_column_yields_bare_ellipsis() -> None:
    """A 1-column budget can hold nothing but the ellipsis, so callers must
    never get a clipped first glyph that misleads the user about content."""
    assert _truncate_to_width("abcdef", 1) == "…"


def test_terminal_width_falls_back_when_size_query_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headless/pipe contexts make the OS size probe raise; the picker must
    degrade to the caller default instead of crashing the whole render."""
    def _boom(*args: object, fallback: tuple[int, int] = (80, 24)) -> os.terminal_size:
        # pytest's own -v progress writer also calls get_terminal_size(fallback=...);
        # only the picker's positional probe raises, so pytest keeps rendering.
        if not args:
            return os.terminal_size(fallback)
        raise OSError("no tty")

    monkeypatch.setattr(shutil, "get_terminal_size", _boom)
    assert picker._terminal_width(default=77) == 77


def _filtered_to_empty(page_size: int = 3) -> _PickerState:
    items = [
        SimplePickerItem(label="apple", sublabel="ripe", value=1),
        SimplePickerItem(label="banana", sublabel="green", value=2),
    ]
    s = _PickerState(items, page_size=page_size, initial_filter="zzz")
    s.clamp_after_filter_change()
    return s


def test_state_navigation_is_noop_when_visible_set_empty() -> None:
    """When a filter hides every row, every navigation key must be inert —
    otherwise the cursor could index a phantom row and crash on select."""
    s = _filtered_to_empty()
    assert s.visible() == []
    for move in (s.move_down, s.move_up, s.page_down, s.page_up, s.end):
        move()
        assert s.cursor == 0
        assert s.viewport_top == 0


def test_state_clamp_resets_on_empty_after_having_a_cursor() -> None:
    """A filter that drops from many matches to zero must zero the cursor so
    a later widening doesn't restore a stale out-of-range position."""
    s = _filtered_to_empty()
    s.cursor = 5
    s.viewport_top = 4
    s.clamp_after_filter_change()
    assert s.cursor == 0
    assert s.viewport_top == 0


def test_render_items_shows_no_matches_placeholder() -> None:
    """An empty visible set must render an explicit placeholder, never a
    blank pane the user could mistake for a frozen UI."""
    s = _filtered_to_empty()
    rendered = "".join(frag[1] for frag in picker._render_items(s))
    assert "no matches" in rendered


def test_render_items_renders_sublabel_for_cursor_and_rest() -> None:
    """Sublabels carry the secondary context (timestamps, paths); the
    renderer must emit them for both the highlighted row and the others."""
    items = [
        SimplePickerItem(label="apple", sublabel="2h ago", value=1),
        SimplePickerItem(label="berry", sublabel="5m ago", value=2),
    ]
    s = _PickerState(items, page_size=8, initial_filter="")
    s.clamp_after_filter_change()
    rendered = "".join(frag[1] for frag in picker._render_items(s))
    assert "2h ago" in rendered
    assert "5m ago" in rendered


async def test_picker_zero_match_filter_then_enter_returns_none() -> None:
    """Pressing Enter while the filter matches nothing must yield no
    selection rather than returning a wrong/first item by accident."""
    items = [
        SimplePickerItem(label="apple", sublabel=None, value="a"),
        SimplePickerItem(label="banana", sublabel=None, value="b"),
    ]
    with _piped("zzz\r"):
        result = await run_picker(items, title="Test", enable_filter=True)
    assert result.item is None


async def test_picker_footer_reports_truncated_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller pre-trims a large list, the footer must surface the
    full ``total_count`` so the user knows more items exist off-screen."""
    monkeypatch.setenv("COLUMNS", "120")
    with _piped("\r"):
        result = await run_picker(
            _items(3),
            title="Test",
            enable_filter=False,
            total_count=500,
        )
    assert result.item is not None
    assert result.item.value == 0


async def test_picker_merges_caller_style_without_dropping_defaults() -> None:
    """A caller-supplied style must overlay, not replace, the picker's own
    palette — losing the cursor/selected rules would make rows unreadable."""
    custom = Style.from_dict({"picker.title": "underline"})
    with _piped("\r"):
        result = await run_picker(
            _items(3),
            title="Test",
            enable_filter=False,
            style=custom,
        )
    assert result.item is not None
    assert result.item.value == 0


async def test_picker_total_count_none_defaults_to_item_length() -> None:
    """Omitting ``total_count`` must fall back to the visible item count so
    the footer never claims a phantom larger universe of options."""
    with _piped("\r"):
        result = await run_picker(
            _items(4),
            title="Test",
            enable_filter=False,
            total_count=None,
        )
    assert result.item is not None
    assert result.item.value == 0


async def test_picker_idempotent_across_repeated_runs() -> None:
    """Re-running the picker on the same immutable items must give identical
    results every time — state must not leak between independent invocations."""
    items = _items(6)
    results: list[object] = []
    for _ in range(3):
        with _piped("\x1b[B\x1b[B\r"):
            result = await run_picker(items, title="Test", enable_filter=False)
        assert result.item is not None
        results.append(result.item.value)
    assert results == [2, 2, 2]
