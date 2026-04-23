"""task_create — spawn a subagent fire-and-forget.

Returns the ``task_id`` the instant the subagent is scheduled, without
awaiting its completion. Parent's loop continues immediately; the subagent
runs as a detached :class:`asyncio.Task` in the same event loop. Callers
poll via ``task_output`` and/or the ``/tasks`` command.

State is injected at Agent construction time — see
``Agent.__init__``'s stateful-tools wiring block.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.core.hooks import HookChain
from aura.core.tasks.agent_types import all_agent_types, get_agent_type
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from aura.schemas.tool import ToolError, tool_metadata


def _agent_type_field_description() -> str:
    """Human-readable catalogue of available agent types.

    Rendered once at module import and embedded in the ``agent_type``
    pydantic field description so the LLM sees the options + selection
    guidance inside the tool schema — no separate introspection call
    needed.
    """
    lines = [
        "Subagent flavor to dispatch. Each flavor has a different tool "
        "allowlist + system prompt. Pick the most restrictive type that "
        "still accomplishes the task. Options:",
    ]
    for type_def in all_agent_types():
        lines.append(f"- {type_def.name!r}: {type_def.description}")
    return "\n".join(lines)


class TaskCreateParams(BaseModel):
    description: str = Field(
        ..., min_length=1, max_length=100,
        description="Short name shown in /tasks list.",
    )
    prompt: str = Field(
        ..., min_length=1,
        description="The prompt the subagent should work on.",
    )
    agent_type: str = Field(
        default="general-purpose",
        description=_agent_type_field_description(),
    )


def _preview(args: dict[str, Any]) -> str:
    desc = args.get("description", "?")
    return f"subagent: {desc}"


class TaskCreate(BaseTool):
    """Spawn a subagent; return its task_id immediately."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "task_create"
    description: str = (
        "Spawn a subagent to work on a focused subtask. Returns a task_id "
        "immediately — use task_output(task_id) to fetch progress / result. "
        "The subagent runs in the background; your turn continues."
    )
    args_schema: type[BaseModel] = TaskCreateParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_destructive=False,
        # State-mutating (adds to TasksStore + creates an asyncio.Task on
        # the loop); can't be batched with siblings.
        is_concurrency_safe=False,
        max_result_size_chars=1000,
        args_preview=_preview,
    )
    # Stateful deps: store + factory + the running-tasks map (shared with
    # the owning Agent so Agent.close() can cancel). ``running`` lives as a
    # PrivateAttr because pydantic v2 eagerly deep-copies dict fields during
    # model validation, which would break the identity-sharing contract
    # with the owning Agent. Accepted as a regular kwarg via __init__ below.
    store: TasksStore
    factory: SubagentFactory
    _running: dict[str, asyncio.Task[None]] = PrivateAttr()
    # Optional parent hook chain — threaded through so ``run_task`` can
    # fire ``post_subagent`` on terminal transitions. Stored as a
    # PrivateAttr for the same pydantic-vs-bare-Callable reason
    # ``_running`` is; defaults to ``None`` so legacy tests that wire
    # task_create without a chain keep working (no hooks fired, same
    # behavior as before the slot was added).
    _parent_hooks: HookChain | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        store: TasksStore,
        factory: SubagentFactory,
        running: dict[str, asyncio.Task[None]],
        parent_hooks: HookChain | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(store=store, factory=factory, **kwargs)
        self._running = running
        self._parent_hooks = parent_hooks

    @property
    def running(self) -> dict[str, asyncio.Task[None]]:
        return self._running

    def _run(
        self,
        description: str,
        prompt: str,
        agent_type: str = "general-purpose",
    ) -> dict[str, Any]:
        raise NotImplementedError("task_create is async-only; use ainvoke")

    async def _arun(
        self,
        description: str,
        prompt: str,
        agent_type: str = "general-purpose",
    ) -> dict[str, Any]:
        # Validate the flavor BEFORE touching the store so a typo can't
        # leave an orphan record. ``get_agent_type`` raises ValueError with
        # the full list of valid names — we lift it into ToolError so the
        # LLM sees the message in the tool result and can self-correct.
        try:
            get_agent_type(agent_type)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        record = self.store.create(
            description=description,
            prompt=prompt,
            agent_type=agent_type,
        )
        # Fire-and-forget. The handle is kept in ``self._running`` so the
        # parent Agent can cancel it on close / Ctrl+C. We also attach a
        # done-callback to clean the map on natural completion so the
        # dict doesn't grow unbounded across a long session.
        task: asyncio.Task[None] = asyncio.create_task(
            run_task(
                self.store,
                self.factory,
                record.id,
                parent_hooks=self._parent_hooks,
            ),
            name=f"aura-subagent-{record.id[:8]}",
        )
        self._running[record.id] = task

        def _cleanup(t: asyncio.Task[None]) -> None:
            self._running.pop(record.id, None)
            # Swallow any residual exception so the event loop's default
            # "Task exception was never retrieved" warning doesn't fire —
            # run_task already wrote the failure into the record.
            if not t.cancelled():
                t.exception()

        task.add_done_callback(_cleanup)
        return {
            "task_id": record.id,
            "description": record.description,
            "status": "running",
            "agent_type": agent_type,
        }
