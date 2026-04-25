"""F-07-003 + F-07-008: parent observability bridge for subagents.

Two halves:

- ``<task-notification>`` push: when a subagent reaches a terminal state,
  the parent's TasksStore listener queues a synthetic notification that
  Context.build drains as a HumanMessage on the next prompt envelope.
- ``task_output(wait=True)`` blocking poll: the parent can park on a
  child's terminal event with an asyncio-aware timeout instead of
  burning loop turns polling.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult

from aura.config.schema import AuraConfig
from aura.core.abort import AbortController, current_abort_signal
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from aura.core.tasks.types import TaskNotification
from aura.tools.task_create import TaskCreate
from aura.tools.task_output import TaskOutput
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["task_create", "task_output"]},
    })


def _make_factory() -> SubagentFactory:
    return SubagentFactory(
        parent_config=AuraConfig.model_validate({
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="child-final"))],
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )


def _make_agent(tmp_path: Path) -> Agent:
    """Build a real parent Agent so the listener wiring is exercised."""
    cfg = _cfg()
    storage = SessionStorage(tmp_path / "parent.db")
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="parent"))]),
        storage=storage,
    )


# ---------------------------------------------------------------------------
# F-07-003 — parent push notifications on subagent terminal transitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_completion_pushes_notification_to_parent(
    tmp_path: Path,
) -> None:
    """A child's mark_completed must enqueue a TaskNotification on the parent."""
    agent = _make_agent(tmp_path)
    try:
        # Hand the agent's actual store to a factory so the listener
        # registered in Agent.__init__ fires when the child terminates.
        store = agent._tasks_store
        factory = _make_factory()
        rec = store.create(description="probe", prompt="hi")
        await run_task(store, factory, rec.id)

        notifications = agent.pending_notifications
        assert len(notifications) == 1
        n = notifications[0]
        assert isinstance(n, TaskNotification)
        assert n.task_id == rec.id
        assert n.status == "completed"
        assert n.description == "probe"
        assert n.summary == "child-final"
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_notification_appears_in_next_prompt_envelope(
    tmp_path: Path,
) -> None:
    """Context.build drains queued notifications into a HumanMessage."""
    agent = _make_agent(tmp_path)
    try:
        store = agent._tasks_store
        factory = _make_factory()
        rec = store.create(description="audit-x", prompt="go")
        await run_task(store, factory, rec.id)

        # Render the next prompt envelope — this is what the model sees
        # before the next ainvoke. The notification block sits AFTER
        # the static eager prefix (system + project-memory) and BEFORE
        # history.
        ctx = agent._context
        rendered = ctx.build([HumanMessage(content="next-user-turn")])
        notification_msgs = [
            m for m in rendered
            if isinstance(m, HumanMessage)
            and isinstance(m.content, str)
            and "<task-notification>" in m.content
        ]
        assert len(notification_msgs) == 1
        body = notification_msgs[0].content
        assert isinstance(body, str)
        assert rec.id[:8] in body
        assert "audit-x" in body
        assert "completed" in body
        assert "child-final" in body

        # Drain semantics: a second build returns no notification block.
        again = ctx.build([HumanMessage(content="another-turn")])
        again_msgs = [
            m for m in again
            if isinstance(m, HumanMessage)
            and isinstance(m.content, str)
            and "<task-notification>" in m.content
        ]
        assert again_msgs == []
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_multiple_notifications_aggregate_within_one_flush(
    tmp_path: Path,
) -> None:
    """Multiple terminals before the next build must batch into ONE message."""
    agent = _make_agent(tmp_path)
    try:
        store = agent._tasks_store
        factory = _make_factory()
        rec_a = store.create(description="a", prompt="p1")
        rec_b = store.create(description="b", prompt="p2")
        rec_c = store.create(description="c", prompt="p3")
        await run_task(store, factory, rec_a.id)
        await run_task(store, factory, rec_b.id)
        await run_task(store, factory, rec_c.id)

        rendered = agent._context.build([])
        notif = [
            m for m in rendered
            if isinstance(m, HumanMessage)
            and isinstance(m.content, str)
            and "<task-notification>" in m.content
        ]
        assert len(notif) == 1
        body = notif[0].content
        assert isinstance(body, str)
        # All three task_ids land in the single block.
        for r in (rec_a, rec_b, rec_c):
            assert r.id[:8] in body
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_notification_queue_caps_at_five(tmp_path: Path) -> None:
    """More than 5 notifications collapse the tail to ``(N more earlier)``."""
    agent = _make_agent(tmp_path)
    try:
        store = agent._tasks_store
        factory = _make_factory()
        rec_ids = []
        for i in range(8):
            rec = store.create(description=f"d{i}", prompt=f"p{i}")
            await run_task(store, factory, rec.id)
            rec_ids.append(rec.id)

        rendered = agent._context.build([])
        notif = [
            m for m in rendered
            if isinstance(m, HumanMessage)
            and isinstance(m.content, str)
            and "<task-notification>" in m.content
        ]
        assert len(notif) == 1
        body = notif[0].content
        assert isinstance(body, str)
        # First 5 should be present (FIFO order).
        for rid in rec_ids[:5]:
            assert rid[:8] in body
        # Overflow envelope.
        assert "(3 more earlier)" in body
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_notification_with_summary_when_subagent_returns_one(
    tmp_path: Path,
) -> None:
    """Completed-with-final-text ⇒ summary carries the final_result body."""
    agent = _make_agent(tmp_path)
    try:
        store = agent._tasks_store

        class _CustomFactory(SubagentFactory):
            def __init__(self) -> None:
                # Skip parent SubagentFactory.__init__; we only ever call
                # spawn here and don't need its dependencies.
                self._parent_config = AuraConfig.model_validate({
                    "providers": [{"name": "openai", "protocol": "openai"}],
                    "router": {"default": "openai:gpt-4o-mini"},
                    "tools": {"enabled": []},
                })
                self._parent_model_spec = "openai:gpt-4o-mini"
                self._parent_skills = None
                self._parent_read_records_provider = None
                self._parent_ruleset = None
                self._parent_deny_rules = None
                self._parent_ask_rules = None
                self._parent_safety = None
                self._parent_mode_provider = None
                self._parent_session = None
                self._model_factory = lambda: FakeChatModel(
                    turns=[FakeTurn(AIMessage(content="found 3 issues"))],
                )
                self._storage_factory = lambda: SessionStorage(
                    Path(":memory:")
                )

        rec = store.create(description="audit", prompt="go")
        await run_task(store, _CustomFactory(), rec.id)

        notifications = agent.pending_notifications
        assert len(notifications) == 1
        assert notifications[0].summary == "found 3 issues"
        assert notifications[0].status == "completed"
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_notification_for_failed_subagent_includes_error(
    tmp_path: Path,
) -> None:
    """A failed terminal carries the error string as the summary field."""
    agent = _make_agent(tmp_path)
    try:
        store = agent._tasks_store

        class _BoomFactory(SubagentFactory):
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
                raise RuntimeError("explode")

        rec = store.create(description="boom", prompt="go")
        await run_task(store, _BoomFactory(), rec.id)

        notifications = agent.pending_notifications
        assert len(notifications) == 1
        n = notifications[0]
        assert n.status == "failed"
        assert n.summary is not None
        assert "explode" in n.summary
    finally:
        await agent.aclose()


# ---------------------------------------------------------------------------
# F-07-008 — task_output(wait=True) blocking poll
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_output_wait_true_blocks_until_terminal(
    tmp_path: Path,
) -> None:
    store = TasksStore()
    rec = store.create(description="slow", prompt="go")
    tool = TaskOutput(store=store)

    # Park a wait that should resolve when we mark the record completed.
    waiter = asyncio.create_task(
        tool.ainvoke({
            "task_id": rec.id, "wait": True, "timeout": 2.0,
        })
    )
    # Yield once so the waiter parks on the event.
    await asyncio.sleep(0.01)
    assert not waiter.done()

    store.mark_completed(rec.id, "child done")
    result = await asyncio.wait_for(waiter, timeout=1.0)
    assert result["terminal"] is True
    assert result["status"] == "completed"
    assert result["final_result"] == "child done"


@pytest.mark.asyncio
async def test_task_output_wait_true_times_out(tmp_path: Path) -> None:
    store = TasksStore()
    rec = store.create(description="slow", prompt="go")
    tool = TaskOutput(store=store)

    result = await tool.ainvoke({
        "task_id": rec.id, "wait": True, "timeout": 0.1,
    })
    assert result["terminal"] is False
    # Status still running because we never marked terminal.
    assert result["status"] == "running"


@pytest.mark.asyncio
async def test_task_output_wait_true_short_circuits_on_parent_abort(
    tmp_path: Path,
) -> None:
    store = TasksStore()
    rec = store.create(description="slow", prompt="go")
    tool = TaskOutput(store=store)

    abort = AbortController()
    cv_token = current_abort_signal.set(abort)
    try:
        async def _run() -> dict[str, Any]:
            result = await tool.ainvoke({
                "task_id": rec.id, "wait": True, "timeout": 5.0,
            })
            assert isinstance(result, dict)
            return dict(result)

        waiter = asyncio.create_task(_run())
        await asyncio.sleep(0.01)
        # Fire the parent abort while the wait is parked.
        abort.abort("parent_cancelled")
        result = await asyncio.wait_for(waiter, timeout=1.0)
        assert result["terminal"] is False
        assert result["error"] == "parent_aborted"
    finally:
        current_abort_signal.reset(cv_token)


@pytest.mark.asyncio
async def test_task_output_wait_false_keeps_existing_behavior(
    tmp_path: Path,
) -> None:
    store = TasksStore()
    rec = store.create(description="probe", prompt="hi")
    tool = TaskOutput(store=store)

    # Fire-and-forget snapshot — must return immediately even though the
    # task is still running, with terminal=False.
    result = await tool.ainvoke({"task_id": rec.id})
    assert result["status"] == "running"
    assert result["terminal"] is False
    assert result["final_result"] is None


@pytest.mark.asyncio
async def test_task_output_wait_true_returns_immediately_for_terminal_task(
    tmp_path: Path,
) -> None:
    """Tasks that are already terminal at wait time short-circuit."""
    store = TasksStore()
    rec = store.create(description="done", prompt="hi")
    store.mark_completed(rec.id, "result")
    tool = TaskOutput(store=store)
    result = await tool.ainvoke({
        "task_id": rec.id, "wait": True, "timeout": 5.0,
    })
    # No timeout was waited because we were already terminal.
    assert result["terminal"] is True
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_task_output_unknown_id_still_raises(tmp_path: Path) -> None:
    store = TasksStore()
    tool = TaskOutput(store=store)
    from aura.schemas.tool import ToolError
    with pytest.raises(ToolError, match="unknown task_id"):
        await tool.ainvoke({"task_id": "no-such", "wait": True, "timeout": 0.1})


@pytest.mark.asyncio
async def test_full_flow_task_create_then_wait(tmp_path: Path) -> None:
    """End-to-end: spawn via task_create, wait via task_output(wait=True)."""

    class _SlowFake(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            await asyncio.sleep(0.1)
            return await super()._agenerate(messages, stop, run_manager, **_)

    def _slow_fake_factory() -> _SlowFake:
        # ``turns`` lives on FakeChatModel.__init__ but mypy resolves
        # the subclass init via pydantic and loses the kwarg signature.
        # Build the instance and seed its turn dict directly, matching
        # how FakeChatModel's __init__ does it.
        m = _SlowFake()
        m.__dict__["_turns"] = [FakeTurn(AIMessage(content="late-final"))]
        return m

    store = TasksStore()
    factory = SubagentFactory(
        parent_config=AuraConfig.model_validate({
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=_slow_fake_factory,
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    create_tool = TaskCreate(
        store=store, factory=factory, running={},
    )
    out = await create_tool.ainvoke({"description": "d", "prompt": "p"})
    task_id = out["task_id"]

    output_tool = TaskOutput(store=store)
    result = await output_tool.ainvoke({
        "task_id": task_id, "wait": True, "timeout": 3.0,
    })
    assert result["terminal"] is True
    assert result["status"] == "completed"
    assert result["final_result"] == "late-final"
