"""/tasks — list running / recent subagent tasks.

Minimal table: short-id, status, description. Sorted newest-first and
capped at 20 rows so a long session doesn't scroll the terminal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.core.commands.types import CommandResult, CommandSource

if TYPE_CHECKING:
    from aura.core.agent import Agent


class TasksCommand:
    name = "/tasks"
    description = "list subagent tasks (running / recent)"
    source: CommandSource = "builtin"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        records = agent._tasks_store.list()
        if not records:
            return CommandResult(handled=True, kind="print", text="(no tasks)")
        records = sorted(records, key=lambda r: -r.started_at)[:20]
        lines = [
            f"{r.id[:8]}  {r.status:>10}  {r.description}"
            for r in records
        ]
        return CommandResult(
            handled=True, kind="print", text="\n".join(lines)
        )
