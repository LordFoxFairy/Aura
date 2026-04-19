"""Rich renderer for AgentEvent instances."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
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

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, AssistantDelta):
            self._console.print(Markdown(event.text))
            return
        if isinstance(event, ToolCallStarted):
            self._console.print(
                f"[dim]◆ {event.name}({compact_args(event.input)})[/dim]",
            )
            return
        if isinstance(event, ToolCallCompleted):
            if event.error:
                self._console.print(f"[red]✗ {event.error}[/red]")
            else:
                self._console.print("[green]✓[/green]")
            return
        if isinstance(event, Final):
            return

    def finish(self) -> None:
        self._console.print()


def compact_args(args: dict[str, Any], *, max_len: int = 80) -> str:
    """Compact JSON preview of a params dict, truncated with ellipsis."""
    rendered = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) <= max_len:
        return rendered
    return rendered[:max_len] + "…"
