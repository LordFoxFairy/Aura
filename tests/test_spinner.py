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
    assert spinner._tick_task is not None
    await asyncio.sleep(0.05)
    await spinner.stop()
    assert spinner._tick_task is None


async def test_spinner_picks_a_verb_from_pool() -> None:
    from aura.cli.spinner import _VERBS

    spinner = ThinkingSpinner(_console())
    spinner.start()
    assert spinner._verb in _VERBS
    await spinner.stop()


async def test_spinner_elapsed_seconds_grow() -> None:
    spinner = ThinkingSpinner(_console())
    spinner.start()
    r0 = spinner._render()
    await asyncio.sleep(1.1)
    r1 = spinner._render()
    await spinner.stop()
    assert r0 != r1


async def test_spinner_registered_frames_match_claude_code() -> None:
    from aura.cli.spinner import _SPINNERS

    frames = _SPINNERS["aura"]["frames"]
    assert isinstance(frames, list)
    assert frames[0] == "·"
    assert "✶" in frames
    assert len(frames) == 12
