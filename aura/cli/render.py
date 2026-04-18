"""Rich renderer for AgentEvent instances — live markdown + tool call markers."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from aura.core.events import (
    AgentEvent,
    AssistantDelta,
    Final,
    ToolCallCompleted,
    ToolCallStarted,
)


class Renderer:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._buffer = ""
        self._live: Live | None = None

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, AssistantDelta):
            self._append_assistant(event.text)
            return
        if isinstance(event, ToolCallStarted):
            self._close_live()
            self._console.print(
                f"[dim]◆ {event.name}({_short(event.input)})[/dim]",
            )
            return
        if isinstance(event, ToolCallCompleted):
            if event.error:
                self._console.print(f"[red]✗ {event.error}[/red]")
            else:
                self._console.print("[green]✓[/green]")
            return
        if isinstance(event, Final):
            self._close_live()

    def finish(self) -> None:
        self._close_live()
        self._console.print()

    def _append_assistant(self, text: str) -> None:
        if self._live is None:
            self._buffer = ""
            self._live = Live(
                Markdown(""),
                console=self._console,
                refresh_per_second=12,
                auto_refresh=True,
            )
            self._live.start()
        self._buffer += text
        self._live.update(Markdown(self._buffer))

    def _close_live(self) -> None:
        if self._live is not None:
            self._live.update(Markdown(self._buffer))
            self._live.stop()
            self._live = None
            self._buffer = ""


def _short(args: dict[str, Any], *, max_len: int = 80) -> str:
    rendered = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) <= max_len:
        return rendered
    return rendered[:max_len] + "…"
