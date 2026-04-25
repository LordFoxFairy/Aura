"""Tests for subagent lifecycle journal events (V13-T1C).

Contract:
- ``subagent_start`` fires BEFORE factory.spawn, carrying task_id +
  agent_type + prompt_chars. Lets operators observe "run_task began"
  even if spawn raises immediately.
- Every terminal event (``subagent_completed`` / ``subagent_cancelled`` /
  ``subagent_timeout`` / ``subagent_failed``) carries a ``duration_sec``
  field measured from just-before-spawn to the terminal branch — no more
  "which subagent took 4 minutes?" mystery in the audit log.
- ``subagent_completed`` additionally carries ``final_text_chars`` so
  post-hoc audits can tell "this task returned 12k chars" from "returned
  nothing" without re-reading storage.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence import journal as journal_module
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from tests.conftest import FakeChatModel, FakeTurn


def _minimal_config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["read_file", "write_file", "bash"]},
    })


def _events(log: Path) -> list[dict[str, object]]:
    if not log.exists():
        return []
    return [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _storage(root: Path) -> SessionStorage:
    return SessionStorage(root / "aura.db")


class _CompletingFactory(SubagentFactory):
    """Spawns a child that finishes its astream immediately."""

    def __init__(self, tmp_path: Path) -> None:
        pass  # skip base __init__

    def spawn(
        self,
        prompt: str,
        allowed_tools: list[str] | None = None,
        *,
        agent_type: str = "general-purpose",
        task_id: str | None = None,
        model_spec: str | None = None,
    ) -> Agent:
        tmp_path = Path("/tmp/aura-subagent-lifecycle-tests")
        tmp_path.mkdir(parents=True, exist_ok=True)
        return Agent(
            config=_minimal_config(),
            model=FakeChatModel(
                turns=[FakeTurn(message=AIMessage(content="done-text"))],
            ),
            storage=_storage(tmp_path / f"run-{task_id or 'x'}"),
        )


class _FailingFactory(SubagentFactory):
    def __init__(self) -> None:
        pass

    def spawn(
        self,
        prompt: str,
        allowed_tools: list[str] | None = None,
        *,
        agent_type: str = "general-purpose",
        task_id: str | None = None,
        model_spec: str | None = None,
    ) -> Agent:
        raise RuntimeError("spawn boom")


@pytest.mark.asyncio
async def test_subagent_start_fires_with_prompt_chars(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        store = TasksStore()
        record = store.create(
            description="probe",
            prompt="hello world",
            agent_type="general-purpose",
        )
        await run_task(store, _CompletingFactory(tmp_path), record.id)

        events = _events(log)
        starts = [e for e in events if e["event"] == "subagent_start"]
        assert len(starts) == 1
        assert starts[0]["task_id"] == record.id
        assert starts[0]["agent_type"] == "general-purpose"
        assert starts[0]["prompt_chars"] == len("hello world")
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_subagent_completed_carries_duration_and_final_text_chars(
    tmp_path: Path,
) -> None:
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        store = TasksStore()
        record = store.create(
            description="probe", prompt="x", agent_type="general-purpose",
        )
        await run_task(store, _CompletingFactory(tmp_path), record.id)

        events = _events(log)
        completed = [e for e in events if e["event"] == "subagent_completed"]
        assert len(completed) == 1
        ev = completed[0]
        assert ev["task_id"] == record.id
        assert isinstance(ev["duration_sec"], float)
        assert ev["duration_sec"] >= 0.0
        assert ev["final_text_chars"] == len("done-text")
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_subagent_start_fires_even_when_spawn_raises(
    tmp_path: Path,
) -> None:
    """The start event is emitted BEFORE factory.spawn — if spawn raises,
    operators still see a lifecycle start / failed pair in the journal."""
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        store = TasksStore()
        record = store.create(
            description="probe", prompt="x", agent_type="general-purpose",
        )
        await run_task(store, _FailingFactory(), record.id)

        events = _events(log)
        starts = [e for e in events if e["event"] == "subagent_start"]
        failed = [e for e in events if e["event"] == "subagent_failed"]
        assert len(starts) == 1
        assert len(failed) == 1
        assert starts[0]["task_id"] == record.id
        assert failed[0]["task_id"] == record.id
        err = failed[0]["error"]
        assert isinstance(err, str)
        assert "spawn boom" in err
        assert isinstance(failed[0]["duration_sec"], float)
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_subagent_cancelled_carries_duration(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)

    class _SlowSpinFactory(SubagentFactory):
        def __init__(self) -> None:
            pass

        def spawn(
            self,
            prompt: str,
            allowed_tools: list[str] | None = None,
            *,
            agent_type: str = "general-purpose",
            task_id: str | None = None,
            model_spec: str | None = None,
        ) -> Agent:
            _spin_path = Path("/tmp/aura-subagent-lifecycle-cancel")
            _spin_path.mkdir(parents=True, exist_ok=True)
            agent = Agent(
                config=_minimal_config(),
                model=FakeChatModel(
                    turns=[FakeTurn(message=AIMessage(content="_never_"))],
                ),
                storage=_storage(_spin_path / f"run-{task_id or 'x'}"),
            )
            # Patch astream to sleep forever so outer cancellation fires.
            async def _forever(_: Any) -> Any:  # pragma: no cover
                await asyncio.sleep(100)
                if False:  # pragma: no cover - make this an async generator
                    yield
            agent.astream = _forever  # type: ignore[method-assign,assignment]
            return agent

    try:
        store = TasksStore()
        record = store.create(
            description="probe", prompt="x", agent_type="general-purpose",
        )
        task = asyncio.create_task(
            run_task(store, _SlowSpinFactory(), record.id),
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        events = _events(log)
        cancelled = [e for e in events if e["event"] == "subagent_cancelled"]
        assert len(cancelled) == 1
        assert cancelled[0]["task_id"] == record.id
        assert isinstance(cancelled[0]["duration_sec"], float)
        assert cancelled[0]["duration_sec"] >= 0.0
    finally:
        journal_module.reset()
