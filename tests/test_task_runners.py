"""LocalAgentTask — the only runner topology task_create dispatches to.

Exercises spawn / abort / idempotent start via the
``start() / wait_for_terminal() / abort()`` contract.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aura.application.session import AgentSession
from aura.application.tasks.runners import LocalAgentTask
from aura.application.tasks.spawn import SpawnContext, SubagentSpawner
from aura.application.tasks.store import TasksStore
from aura.config.schema import AuraConfig
from aura.infrastructure.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _factory_with_reply(text: str) -> SubagentSpawner[AgentSession]:
    return SubagentSpawner(
        SpawnContext(
            parent_config=_cfg(),
            parent_model_spec="openai:gpt-4o-mini",
            build_child=AgentSession,
            model_factory=lambda: FakeChatModel(
                turns=[FakeTurn(AIMessage(content=text))],
            ),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )


@pytest.mark.asyncio
async def test_local_agent_task_runs_to_completion() -> None:
    store = TasksStore()
    factory = _factory_with_reply("child done")
    rec = store.create(description="probe", prompt="go")
    task = LocalAgentTask(store=store, factory=factory, task_id=rec.id)
    task.start()
    await task.wait_for_terminal()
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.final_result == "child done"


@pytest.mark.asyncio
async def test_local_agent_task_abort_cancels_task() -> None:
    class _Slow(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            # Long enough that the abort definitely lands before
            # natural completion. CancelledError will short-circuit it.
            await asyncio.sleep(5.0)
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="!"))],
            )

    factory = SubagentSpawner(
        SpawnContext(
            parent_config=_cfg(),
            parent_model_spec="openai:gpt-4o-mini",
            build_child=AgentSession,
            model_factory=lambda: _Slow(),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    store = TasksStore()
    rec = store.create(description="slow", prompt="hang")
    task = LocalAgentTask(store=store, factory=factory, task_id=rec.id)
    task.start()
    # Give the child a moment to enter ``_agenerate`` so the cancel
    # has something to race against.
    await asyncio.sleep(0.05)
    task.abort()
    await task.wait_for_terminal()
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "cancelled"


@pytest.mark.asyncio
async def test_local_agent_task_start_is_idempotent() -> None:
    factory = _factory_with_reply("ok")
    store = TasksStore()
    rec = store.create(description="probe", prompt="go")
    task = LocalAgentTask(store=store, factory=factory, task_id=rec.id)
    t1 = task.start()
    t2 = task.start()
    # Same handle returned — no double-scheduling.
    assert t1 is t2
    await task.wait_for_terminal()
