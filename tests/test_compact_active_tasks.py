"""F-0910-020 — SUBAGENT-STOP semantics on compact.

After the rebuild, the new history includes one ``<active-task>``
HumanMessage per still-relevant subagent task (status in {running,
completed}). Failed / cancelled tasks are excluded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _agent(tmp_path: Path) -> Agent:
    return Agent(
        config=_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="SUMMARY"))]),
        storage=SessionStorage(tmp_path / "aura.db"),
    )


def _seed_history(agent: Agent, *, pairs: int = 10) -> None:
    h: list[Any] = []
    for i in range(pairs):
        h.append(HumanMessage(content=f"u-{i}"))
        h.append(AIMessage(content=f"a-{i}"))
    agent._storage.save(agent.session_id, h)


@pytest.mark.asyncio
async def test_running_task_emitted_as_active_task_message(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    _seed_history(agent)
    rec = agent._tasks_store.create(
        description="audit subagent",
        prompt="audit the codebase",
    )
    assert rec.status == "running"

    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "<active-task" in blob
    assert rec.id in blob
    assert 'status="running"' in blob
    assert "audit subagent" in blob


@pytest.mark.asyncio
async def test_completed_task_still_surfaces(tmp_path: Path) -> None:
    """Completed tasks count as 'completed-not-retrieved' in our store
    semantics — they survive compact so the model can still reference
    them after a long gap."""
    agent = _agent(tmp_path)
    _seed_history(agent)
    rec = agent._tasks_store.create(description="done subagent", prompt="x")
    agent._tasks_store.mark_completed(rec.id, result="ok")

    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "done subagent" in blob
    assert 'status="completed"' in blob


@pytest.mark.asyncio
async def test_failed_and_cancelled_excluded(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_history(agent)
    failed = agent._tasks_store.create(description="failed sub", prompt="x")
    agent._tasks_store.mark_failed(failed.id, error="boom")
    cancelled = agent._tasks_store.create(description="cancelled sub", prompt="x")
    agent._tasks_store.mark_cancelled(cancelled.id)

    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "failed sub" not in blob
    assert "cancelled sub" not in blob


@pytest.mark.asyncio
async def test_no_tasks_no_active_task_messages(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_history(agent)

    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "<active-task" not in blob
