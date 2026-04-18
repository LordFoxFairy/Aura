"""Tests for aura.cli.spinner.ThinkingSpinner."""

from __future__ import annotations

import asyncio
import io

from rich.console import Console

from aura.cli.spinner import ThinkingSpinner


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=200, highlight=False)


async def test_spinner_starts_and_stops_cleanly() -> None:
    spinner = ThinkingSpinner(_console())
    spinner.start()
    await asyncio.sleep(0.1)
    await spinner.stop()


async def test_spinner_stop_cancels_tick_task() -> None:
    spinner = ThinkingSpinner(_console())
    spinner.start()
    assert spinner._task is not None
    await asyncio.sleep(0.05)
    await spinner.stop()
    assert spinner._task is None


async def test_spinner_picks_a_verb_from_pool() -> None:
    from aura.cli.spinner import _VERBS

    spinner = ThinkingSpinner(_console())
    spinner.start()
    assert spinner._verb in _VERBS
    await spinner.stop()


async def test_spinner_elapsed_seconds_grow() -> None:
    spinner = ThinkingSpinner(_console())
    spinner.start()
    r0 = spinner._render(0)
    await asyncio.sleep(1.1)
    r1 = spinner._render(0)
    await spinner.stop()
    assert r0 != r1


def test_glyph_color_cyan_under_warn_threshold() -> None:
    from aura.cli.spinner import _STALL_WARN_SEC, _glyph_color

    assert _glyph_color(0) == "cyan"
    assert _glyph_color(_STALL_WARN_SEC - 0.01) == "cyan"


def test_glyph_color_red_over_red_threshold() -> None:
    from aura.cli.spinner import _STALL_RED_SEC, _glyph_color

    assert _glyph_color(_STALL_RED_SEC) == "red"
    assert _glyph_color(_STALL_RED_SEC + 10) == "red"


def test_glyph_color_blends_in_warn_zone() -> None:
    from aura.cli.spinner import _glyph_color

    mid = _glyph_color(22.5)
    assert mid.startswith("rgb(")
    assert "127" in mid or "128" in mid
