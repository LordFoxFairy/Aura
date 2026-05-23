"""F-0910-020 — SUBAGENT-STOP semantics on compact.

After the rebuild, the new history includes one ``<active-task>``
HumanMessage per still-relevant subagent task: running tasks and terminal
tasks whose result has not yet been observed through the task tools.
Observed terminal tasks are excluded so compact does not keep re-injecting
already-read results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.infrastructure.persistence.storage import SessionStorage
from aura.tools.task_get import TaskGet
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
    agent.storage.save(agent.session_id, h)


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

    history = agent.storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "<active-task" in blob
    assert rec.id in blob
    assert 'status="running"' in blob
    assert "audit subagent" in blob


@pytest.mark.asyncio
async def test_unobserved_terminal_tasks_still_surface(tmp_path: Path) -> None:
    """Unobserved terminal tasks survive compact until task_get/task_output
    marks them observed."""
    agent = _agent(tmp_path)
    _seed_history(agent)
    completed = agent._tasks_store.create(description="done subagent", prompt="x")
    agent._tasks_store.mark_completed(completed.id, result="ok")
    failed = agent._tasks_store.create(description="failed subagent", prompt="x")
    agent._tasks_store.mark_failed(failed.id, error="boom")
    cancelled = agent._tasks_store.create(description="cancelled subagent", prompt="x")
    agent._tasks_store.mark_cancelled(cancelled.id)

    await agent.compact(source="manual")

    history = agent.storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "done subagent" in blob
    assert 'status="completed"' in blob
    assert "failed subagent" in blob
    assert 'status="failed"' in blob
    assert "cancelled subagent" in blob
    assert 'status="cancelled"' in blob


@pytest.mark.asyncio
async def test_observed_terminal_tasks_are_excluded(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_history(agent)
    completed = agent._tasks_store.create(description="observed done", prompt="x")
    agent._tasks_store.mark_completed(completed.id, result="ok")
    failed = agent._tasks_store.create(description="observed failed", prompt="x")
    agent._tasks_store.mark_failed(failed.id, error="boom")
    cancelled = agent._tasks_store.create(description="observed cancelled", prompt="x")
    agent._tasks_store.mark_cancelled(cancelled.id)

    tool = TaskGet(store=agent._tasks_store)
    for rec in (completed, failed, cancelled):
        assert tool._run(rec.id)["observed_at"] is not None

    await agent.compact(source="manual")

    history = agent.storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "observed done" not in blob
    assert "observed failed" not in blob
    assert "observed cancelled" not in blob


@pytest.mark.asyncio
async def test_mixed_task_observation_controls_compact_injection(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    _seed_history(agent)
    running = agent._tasks_store.create(description="still running", prompt="x")
    completed = agent._tasks_store.create(description="pending success", prompt="x")
    agent._tasks_store.mark_completed(completed.id, result="ok")
    failed = agent._tasks_store.create(description="unseen failed", prompt="x")
    agent._tasks_store.mark_failed(failed.id, error="boom")
    observed = agent._tasks_store.create(description="retrieved result", prompt="x")
    agent._tasks_store.mark_completed(observed.id, result="seen")

    tool = TaskGet(store=agent._tasks_store)
    first_seen = tool._run(observed.id)["observed_at"]
    assert first_seen is not None
    assert tool._run(observed.id)["observed_at"] == first_seen

    await agent.compact(source="manual")

    history = agent.storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert running.id in blob
    assert "still running" in blob
    assert completed.id in blob
    assert "pending success" in blob
    assert failed.id in blob
    assert "unseen failed" in blob
    assert observed.id not in blob
    assert "retrieved result" not in blob


@pytest.mark.asyncio
async def test_no_tasks_no_active_task_messages(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_history(agent)

    await agent.compact(source="manual")

    history = agent.storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "<active-task" not in blob
