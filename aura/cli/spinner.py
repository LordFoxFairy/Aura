"""Thinking spinner: animated glyph, random verb, elapsed seconds, stall coloring."""

from __future__ import annotations

import asyncio
import random
import time

from rich.console import Console
from rich.live import Live
from rich.text import Text

_GLYPHS = ["·", "✢", "✳", "✶", "✻", "✽"]
_FRAMES = _GLYPHS + list(reversed(_GLYPHS))
_FRAME_INTERVAL = 0.12
_STALL_WARN_SEC = 15.0
_STALL_RED_SEC = 30.0


_VERBS = [
    "Accomplishing",
    "Architecting",
    "Brewing",
    "Calculating",
    "Channeling",
    "Churning",
    "Cogitating",
    "Composing",
    "Computing",
    "Concocting",
    "Considering",
    "Contemplating",
    "Crafting",
    "Creating",
    "Crunching",
    "Crystallizing",
    "Deliberating",
    "Deciphering",
    "Elucidating",
    "Envisioning",
    "Fabricating",
    "Gestating",
    "Ideating",
    "Musing",
    "Orchestrating",
    "Pondering",
    "Processing",
    "Puzzling",
    "Ruminating",
    "Simmering",
    "Stewing",
    "Synthesizing",
    "Thinking",
    "Weaving",
]


def _glyph_color(elapsed: float) -> str:
    if elapsed <= _STALL_WARN_SEC:
        return "cyan"
    if elapsed >= _STALL_RED_SEC:
        return "red"
    t = (elapsed - _STALL_WARN_SEC) / (_STALL_RED_SEC - _STALL_WARN_SEC)
    r = int(255 * t)
    g = int(255 * (1 - t))
    b = int(255 * (1 - t))
    return f"rgb({r},{g},{b})"


class ThinkingSpinner:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._live: Live | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._verb: str = ""
        self._start: float = 0.0

    def start(self) -> None:
        self._verb = random.choice(_VERBS)
        self._start = time.monotonic()
        self._stop_event = asyncio.Event()
        self._live = Live(
            self._render(0),
            console=self._console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()
        self._task = asyncio.create_task(self._tick())

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _render(self, frame: int) -> Text:
        elapsed = time.monotonic() - self._start
        glyph = _FRAMES[frame % len(_FRAMES)]
        text = Text()
        text.append(f"{glyph} ", style=f"bold {_glyph_color(elapsed)}")
        text.append(f"{self._verb}…", style="bold cyan")
        text.append(f" ({int(elapsed)}s)", style="dim")
        return text

    async def _tick(self) -> None:
        assert self._stop_event is not None
        frame = 0
        while not self._stop_event.is_set():
            if self._live is not None:
                self._live.update(self._render(frame))
            frame += 1
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=_FRAME_INTERVAL,
                )
            except TimeoutError:
                continue
            if self._stop_event.is_set():
                return
