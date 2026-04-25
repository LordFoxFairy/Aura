"""task_create — spawn a subagent fire-and-forget.

Returns the ``task_id`` the instant the subagent is scheduled, without
awaiting its completion. Parent's loop continues immediately; the subagent
runs as a detached :class:`asyncio.Task` in the same event loop. Callers
poll via ``task_output`` and/or the ``/tasks`` command.

State is injected at Agent construction time — see
``Agent.__init__``'s stateful-tools wiring block.

Round 7R — per-spawn model override. ``model`` defaults to ``None``
(inherit parent spec); when set, the spec is validated synchronously
via :meth:`SubagentFactory.validate_model_spec` BEFORE the TaskRecord
is created. A typo / unknown alias / unknown provider surfaces as a
``ToolError`` with the resolver's diagnostic — no orphan record left
behind.

Round 4F — transcript persistence. Constructed with an optional
``transcript_storage`` (the parent's :class:`SessionStorage`) so each
child's full message list lands as JSONL under
``<storage_root>/subagents/subagent-<task_id>.jsonl``. ``None`` keeps
the legacy in-memory-only behavior — used by tests + SDK callers that
don't want disk side effects.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.core import llm
from aura.core.persistence.storage import SessionStorage
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
    # Round 7R — optional model override. ``None`` (default) inherits
    # the parent's spec. When set, the string runs through the same
    # router-alias / provider:model resolver the parent uses. Surfaced
    # to the LLM so it can pick a cheap tier for read-only scans
    # (``model="haiku"``) without operator intervention.
    model: str | None = Field(
        default=None,
        description=(
            "Per-spawn model override. Accepts a router alias "
            "(e.g. 'haiku') or 'provider:model' form. None inherits "
            "the parent's model."
        ),
    )
    # Round 7R / claude-code parity — opt-in flag for future async
    # detach semantics. Currently every task is fire-and-forget so
    # this is a no-op signal kept on the schema for forward compat;
    # round-trips through ``TaskCreateParams.model_validate`` for tests.
    run_in_background: bool = Field(
        default=False,
        description=(
            "Reserved — every task_create today is fire-and-forget. "
            "Set to True for forward-compat with future blocking "
            "semantics. The behavior does not change today."
        ),
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
    # Round 4F — optional parent storage for JSONL transcript flush. Threaded
    # through ``__init__`` and forwarded to ``run_task``. ``None`` is the
    # legacy "no on-disk transcript" path.
    _transcript_storage: SessionStorage | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        store: TasksStore,
        factory: SubagentFactory,
        running: dict[str, asyncio.Task[None]],
        transcript_storage: SessionStorage | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(store=store, factory=factory, **kwargs)
        self._running = running
        self._transcript_storage = transcript_storage

    @property
    def running(self) -> dict[str, asyncio.Task[None]]:
        return self._running

    def _run(
        self,
        description: str,
        prompt: str,
        agent_type: str = "general-purpose",
        model: str | None = None,
        run_in_background: bool = False,
    ) -> dict[str, Any]:
        raise NotImplementedError("task_create is async-only; use ainvoke")

    async def _arun(
        self,
        description: str,
        prompt: str,
        agent_type: str = "general-purpose",
        model: str | None = None,
        run_in_background: bool = False,
    ) -> dict[str, Any]:
        # Validate the flavor BEFORE touching the store so a typo can't
        # leave an orphan record. ``get_agent_type`` raises ValueError with
        # the full list of valid names — we lift it into ToolError so the
        # LLM sees the message in the tool result and can self-correct.
        try:
            get_agent_type(agent_type)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        # Round 7R — validate the per-spawn model spec BEFORE creating
        # the record. Same orphan-prevention rationale as agent_type.
        if model is not None:
            try:
                self.factory.validate_model_spec(model)
            except llm.UnknownModelSpecError as exc:
                # Surface "invalid model spec: <detail>" so the LLM can
                # self-correct (the test pins this exact prefix). The
                # resolver's diagnostic is preserved in ``detail``.
                raise ToolError(f"invalid model spec: {exc}") from exc
        # The TaskRecord remembers the spec the child WILL run on —
        # either the explicit override or the inherited parent spec.
        # Pinned at create time (not lazily on first activity) so
        # ``task_get`` always sees a concrete value.
        resolved_spec = (
            model if model is not None else self.factory.parent_model_spec
        )
        record = self.store.create(
            description=description,
            prompt=prompt,
            agent_type=agent_type,
            model_spec=resolved_spec,
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
                transcript_storage=self._transcript_storage,
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
            # Round 7R — surface the resolved spec in the create result
            # so the LLM knows which model the child will run on.
            "model_spec": resolved_spec,
        }
