"""/tasks, /task-get, /task-stop — subagent lifecycle slash commands.

Minimal human-facing surface over the same TasksStore the LLM polls via
``task_get`` / ``task_list`` / ``task_stop``. ``short_id`` (first 8 hex
chars) is what we show; the user can still paste the full id for disambig
if short-id collides, though in practice a single REPL session never
produces two tasks that collide in the first 8 chars of uuid4.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aura.core.commands.types import CommandResult, CommandSource
from aura.core.tasks.types import TaskRecord

if TYPE_CHECKING:
    from aura.core.agent import Agent

_SHORT_ID = 8
# Mirror task_stop's own cap: a subagent should unwind in a single
# event-loop tick. If it doesn't, we force-mark cancelled and move on
# rather than hang the REPL.
_STOP_TIMEOUT_SECONDS = 2.0


def _resolve(agent: Agent, arg: str) -> TaskRecord | None:
    """Resolve the ``arg`` (a prefix or full id) to a TaskRecord.

    Accepts both full uuid hex and the 8-char short-id we display.
    Returns ``None`` if nothing matches — the caller surfaces this as a
    print-kind error.
    """
    arg = arg.strip()
    if not arg:
        return None
    store = agent._tasks_store
    rec = store.get(arg)
    if rec is not None:
        return rec
    # Fall back to prefix match — typical case when the user copies the
    # short-id from /tasks output.
    matches = [r for r in store.list() if r.id.startswith(arg)]
    if len(matches) == 1:
        return matches[0]
    return None


class TasksCommand:
    name = "/tasks"
    description = "list subagent tasks (running / recent)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        records = agent._tasks_store.list()
        if not records:
            return CommandResult(handled=True, kind="print", text="(no tasks)")
        records = sorted(records, key=lambda r: -r.started_at)[:20]
        # Show kind + agent_type alongside status so the operator can tell a
        # "verify" subagent from a "general-purpose" one at a glance; matches
        # the LLM-facing task_list tool which also carries both fields.
        lines = []
        for r in records:
            flavor = r.agent_type or r.kind  # "subagent" kind → agent_type
            lines.append(
                f"{r.id[:_SHORT_ID]}  {r.status:>10}  "
                f"[{flavor:>15}]  {r.description}"
            )
        return CommandResult(
            handled=True, kind="print", text="\n".join(lines)
        )


class TaskGetCommand:
    name = "/task-get"
    description = "show full status of a subagent task by id"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "<id-prefix>"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        if not arg.strip():
            return CommandResult(
                handled=True, kind="print",
                text="usage: /task-get <id-prefix>",
            )
        rec = _resolve(agent, arg)
        if rec is None:
            return CommandResult(
                handled=True, kind="print",
                text=f"no task matches {arg!r}",
            )
        duration = (
            rec.finished_at - rec.started_at
            if rec.finished_at is not None else None
        )
        lines = [
            f"id          {rec.id[:_SHORT_ID]}",
            f"kind        {rec.kind}",
        ]
        if rec.agent_type is not None:
            lines.append(f"agent_type  {rec.agent_type}")
        lines.extend([
            f"status      {rec.status}",
            f"description {rec.description}",
        ])
        if duration is not None:
            lines.append(f"duration    {duration:.2f}s")
        if rec.progress.tool_count:
            lines.append(f"tool_count  {rec.progress.tool_count}")
            if rec.progress.recent_activities:
                lines.append(
                    "recent      "
                    + ", ".join(rec.progress.recent_activities)
                )
        if rec.final_result:
            lines.append(f"result      {rec.final_result}")
        if rec.error:
            lines.append(f"error       {rec.error}")
        return CommandResult(
            handled=True, kind="print", text="\n".join(lines),
        )


class TaskStopCommand:
    name = "/task-stop"
    description = "cancel a running subagent task by id"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "<id-prefix>"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        if not arg.strip():
            return CommandResult(
                handled=True, kind="print",
                text="usage: /task-stop <id-prefix>",
            )
        rec = _resolve(agent, arg)
        if rec is None:
            return CommandResult(
                handled=True, kind="print",
                text=f"no task matches {arg!r}",
            )
        if rec.status != "running":
            return CommandResult(
                handled=True, kind="print",
                text=(
                    f"task {rec.id[:_SHORT_ID]} is already {rec.status}; "
                    "nothing to stop"
                ),
            )
        handle = agent._running_tasks.get(rec.id)
        if handle is None or handle.done():
            agent._tasks_store.mark_cancelled(rec.id)
            return CommandResult(
                handled=True, kind="print",
                text=f"task {rec.id[:_SHORT_ID]} cancelled",
            )
        handle.cancel()
        try:
            await asyncio.wait_for(
                asyncio.shield(handle), timeout=_STOP_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            agent._tasks_store.mark_cancelled(rec.id)
        return CommandResult(
            handled=True, kind="print",
            text=f"task {rec.id[:_SHORT_ID]} cancelled",
        )
