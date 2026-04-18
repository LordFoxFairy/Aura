"""Claude-Code-style thinking spinner with random verb + elapsed seconds."""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from typing import cast

import rich.spinner as _rich_spinner
from rich.console import Console
from rich.status import Status

_SPINNERS = cast(
    "dict[str, dict[str, list[str] | int]]",
    _rich_spinner.SPINNERS,  # type: ignore[attr-defined]
)
_GLYPHS = ["·", "✢", "✳", "✶", "✻", "✽"]
_SPINNERS["aura"] = {
    "interval": 120,
    "frames": _GLYPHS + list(reversed(_GLYPHS)),
}


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


class ThinkingSpinner:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._status: Status | None = None
        self._tick_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._verb: str = ""
        self._start: float = 0.0

    def start(self) -> None:
        self._verb = random.choice(_VERBS)
        self._start = time.monotonic()
        self._stop_event = asyncio.Event()
        self._status = self._console.status(self._render(), spinner="aura")
        self._status.start()
        self._tick_task = asyncio.create_task(self._tick())

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._tick_task is not None:
            await self._tick_task
            self._tick_task = None
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _render(self) -> str:
        elapsed = int(time.monotonic() - self._start)
        return f"[bold cyan]{self._verb}…[/bold cyan] [dim]({elapsed}s)[/dim]"

    async def _tick(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
            if self._stop_event.is_set():
                return
            if self._status is not None:
                self._status.update(status=self._render())
