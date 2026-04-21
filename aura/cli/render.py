"""Rich renderer for AgentEvent instances."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as rich_escape

from aura.schemas.events import (
    AgentEvent,
    AssistantDelta,
    PermissionAudit,
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
            # Escape variable content — tool input / name may contain ``[...]``
            # which rich would otherwise interpret as inline markup.
            name = rich_escape(event.name)
            args = rich_escape(compact_args(event.input))
            self._console.print(f"[dim]◆ {name}({args})[/dim]")
            return
        if isinstance(event, PermissionAudit):
            # Spec §8.4: dim one-liner, 4-space indent, directly after the
            # started line. Audit text carries rule strings that may contain
            # literal ``[...]`` — escape before wrapping in [dim].
            self._console.print(f"    [dim]{rich_escape(event.text)}[/dim]")
            return
        if isinstance(event, ToolCallCompleted):
            if event.error:
                self._console.print(f"[red]✗ {rich_escape(event.error)}[/red]")
            else:
                self._console.print("[green]✓[/green]")
            return
        # Final events carry no new text: the body was already streamed via
        # AssistantDelta, so the renderer intentionally drops them.

    def finish(self) -> None:
        self._console.print()


def compact_args(args: dict[str, Any], *, max_len: int = 80) -> str:
    """Compact JSON preview of a params dict, truncated with ellipsis."""
    rendered = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) <= max_len:
        return rendered
    return rendered[:max_len] + "…"
