"""task_create — spawn a subagent fire-and-forget."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.application.tasks.factory import SubagentFactory
from aura.application.tasks.run import run_task
from aura.application.tasks.store import TasksStore
from aura.domain.tool import ToolError, ToolMetadata
from aura.infrastructure import llm
from aura.infrastructure.agents import all_agent_defs, get_agent_def
from aura.infrastructure.persistence.storage import SessionStorage


def _agent_type_field_description() -> str:
    lines = [
        "Subagent flavor to dispatch. Each flavor has a different tool "
        "allowlist + system prompt. Pick the most restrictive type that "
        "still accomplishes the task. Options:",
    ]
    for type_def in all_agent_defs():
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
    model: str | None = Field(
        default=None,
        description=(
            "Per-spawn model override (router alias or 'provider:model'). "
            "None inherits the parent's model."
        ),
    )


def _preview(args: dict[str, Any]) -> str:
    desc = args.get("description", "?")
    return f"subagent: {desc}"


class TaskCreate(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "task_create"
    description: str = (
        "Spawn a subagent to work on a focused subtask. Returns a task_id "
        "immediately — use task_output(task_id) to fetch progress / result. "
        "The subagent runs in the background; your turn continues."
    )
    args_schema: type[BaseModel] = TaskCreateParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=False,
        rule_matcher=None,
        args_preview=_preview,
        timeout_sec=None,
    )
    store: TasksStore
    factory: SubagentFactory
    # PrivateAttr: pydantic v2 deep-copies regular dict fields, breaking identity-share with Agent.
    _running: dict[str, asyncio.Task[None]] = PrivateAttr()
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
    ) -> dict[str, Any]:
        raise NotImplementedError("task_create is async-only; use ainvoke")

    async def _arun(
        self,
        description: str,
        prompt: str,
        agent_type: str = "general-purpose",
        model: str | None = None,
    ) -> dict[str, Any]:
        # Validate before store insert so a bad agent_type can't leave an orphan record.
        try:
            get_agent_def(agent_type)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        if model is not None:
            try:
                self.factory.validate_model_spec(model)
            except llm.UnknownModelSpecError as exc:
                raise ToolError(f"invalid model spec: {exc}") from exc
        resolved_spec = (
            model if model is not None else self.factory.parent_model_spec
        )
        record = self.store.create(
            description=description,
            prompt=prompt,
            agent_type=agent_type,
            model_spec=resolved_spec,
        )
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
            if not t.cancelled():
                t.exception()

        task.add_done_callback(_cleanup)
        return {
            "task_id": record.id,
            "description": record.description,
            "status": "running",
            "agent_type": agent_type,
            "model_spec": resolved_spec,
        }
