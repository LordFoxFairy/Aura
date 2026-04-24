"""Integration: parent agent → task_create → real subagent → result back.

Unit-level task tests already exist under ``tests/test_task_tools.py``; they
drive the ``task_create`` tool directly and assert the ``TasksStore`` gets
the right record. This tier runs the full LLM-driven path: the parent
model issues a ``task_create`` tool call, the loop dispatches the
subagent Agent, the subagent runs its own ``astream`` → writes a final
result back into the store, the parent model's next turn observes the
result via ``task_get`` / ``task_list``.

If any of the plumbing across that boundary breaks (subagent's Agent
never spawns, final_result never lands, parent never gets the follow-up
turn), this test catches it — whereas the unit tests would keep
passing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aura.core.tasks.store import TasksStore
from tests.conftest import FakeChatModel, FakeTurn
from tests.integration.conftest import build_integration_agent, drain

# ---------------------------------------------------------------------------
# Test 1 — single subagent roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_subagent_roundtrip_parent_sees_result(tmp_path: Path) -> None:
    # Parent's scripted script:
    #   turn 1 → task_create(explore) returning task_id
    #   turn 2 → task_get(task_id) to read the child's final result
    #   turn 3 → final text
    parent_turns = [
        FakeTurn(
            message=AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_create",
                        "name": "task_create",
                        "args": {
                            "description": "listing",
                            "prompt": "list files",
                            "agent_type": "explore",
                        },
                    }
                ],
            )
        ),
        # The second turn must reference the task_id returned by turn 1. We
        # can't bake it in statically, so use a lazy model that pulls the id
        # from the shared store at call time. See _LazyTaskGetParent below.
        FakeTurn(message=AIMessage(content="summary: listing complete")),
    ]

    # Capture the factory hook so we can substitute the subagent's model.
    # The subagent's ``astream`` only needs to emit one Final event.
    agent, parent_model = build_integration_agent(tmp_path, parent_turns)

    # Swap the Subagent factory's model_factory to hand the child a scripted
    # FakeChatModel that finishes in one turn.
    child_final_text = "found 3 files: a.py, b.py, c.py"
    agent._subagent_factory._model_factory = lambda: FakeChatModel(
        turns=[FakeTurn(message=AIMessage(content=child_final_text))]
    )

    # Monkey-patch the second turn message so it calls task_get with the
    # actually-issued task_id. The parent_model pops turns in order; by the
    # time turn 2 is requested, the store has exactly one record.
    store: TasksStore = agent._tasks_store

    class _LazyTaskGetInjector(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            self.__dict__["ainvoke_calls"] += 1
            call_n = self.__dict__["ainvoke_calls"]
            if call_n == 1:
                return ChatResult(
                    generations=[
                        ChatGeneration(message=parent_turns[0].message)
                    ]
                )
            if call_n == 2:
                # Now that the child has been scheduled, grab its id and
                # ask for its result.
                recs = store.list()
                task_id = recs[0].id
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "id": "tc_get",
                                        "name": "task_get",
                                        "args": {"task_id": task_id},
                                    }
                                ],
                            )
                        )
                    ]
                )
            return ChatResult(
                generations=[ChatGeneration(message=parent_turns[1].message)]
            )

    # Swap the parent model wholesale (the build helper pre-set one).
    agent._model = _LazyTaskGetInjector()
    agent._loop = agent._build_loop()

    try:
        events = await drain(agent, "please list the files")
        # Give the detached subagent time to finish — it was scheduled during
        # turn 1's tool dispatch but the parent's turn 2 may observe it while
        # still running. We poll the store until the record is terminal.
        for _ in range(200):
            rec = store.list()[0]
            if rec.status in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("subagent never reached a terminal state")
    finally:
        await agent.aclose()

    # The child ran and wrote its final result.
    rec = store.list()[0]
    assert rec.status == "completed", f"child status={rec.status} err={rec.error}"
    assert rec.final_result == child_final_text

    # Parent saw the result. Find the ToolCallCompleted for task_get and
    # verify the child's final result appears in its output payload.
    from aura.schemas.events import ToolCallCompleted

    task_get_events = [
        e
        for e in events
        if isinstance(e, ToolCallCompleted) and e.name == "task_get"
    ]
    assert len(task_get_events) == 1
    payload = task_get_events[0].output
    assert isinstance(payload, dict)
    assert payload["final_result"] == child_final_text


# ---------------------------------------------------------------------------
# Test 2 — 3-subagent fan-out, parent reads all 3 via task_list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_three_subagents_parent_reads_all(tmp_path: Path) -> None:
    # Parent turn 1: three task_create calls in one AIMessage.
    parent_turn1 = FakeTurn(
        message=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_a",
                    "name": "task_create",
                    "args": {
                        "description": "scan",
                        "prompt": "scan repo",
                        "agent_type": "explore",
                    },
                },
                {
                    "id": "tc_b",
                    "name": "task_create",
                    "args": {
                        "description": "verify",
                        "prompt": "check invariant",
                        "agent_type": "verify",
                    },
                },
                {
                    "id": "tc_c",
                    "name": "task_create",
                    "args": {
                        "description": "plan",
                        "prompt": "draft plan",
                        "agent_type": "plan",
                    },
                },
            ],
        )
    )
    # Turn 2: task_list() to enumerate children.
    parent_turn2 = FakeTurn(
        message=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_list",
                    "name": "task_list",
                    "args": {"status": "all", "kind": "all", "limit": 20},
                }
            ],
        )
    )
    parent_turn3 = FakeTurn(message=AIMessage(content="fleet summary ready"))

    agent, parent_model = build_integration_agent(
        tmp_path, [parent_turn1, parent_turn2, parent_turn3]
    )
    store = agent._tasks_store

    # Child model factory returns a different reply per agent_type by
    # looking at the child Agent's system_prompt at spawn time.
    _responses = {
        "Explore": "scan complete",
        "Verify": "VERDICT: PASS all checks green",
        "Plan": "plan: do X",
    }

    def _model_factory_for_child() -> FakeChatModel:
        # The subagent factory calls this every spawn. We need to pick the
        # reply based on agent_type, but we don't have that info here. Use a
        # custom model that inspects the system message it receives on
        # first call and chooses the reply accordingly.
        class _TypeAwareFake(FakeChatModel):
            def __init__(self) -> None:
                super().__init__(turns=[])

            async def _agenerate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: AsyncCallbackManagerForLLMRun | None = None,
                **_: Any,
            ) -> ChatResult:
                self.__dict__["ainvoke_calls"] += 1
                # Pick the content based on which Subagent context marker
                # appears in the system prompt. The suffix contains exactly
                # one of "Explore" / "Verify" / "Plan".
                sys_text = ""
                for m in messages:
                    if m.type == "system":
                        content = m.content
                        if isinstance(content, str):
                            sys_text += content
                reply = "subagent-default"
                for marker, text in _responses.items():
                    if marker in sys_text:
                        reply = text
                        break
                return ChatResult(
                    generations=[
                        ChatGeneration(message=AIMessage(content=reply))
                    ]
                )

        return _TypeAwareFake()

    agent._subagent_factory._model_factory = _model_factory_for_child

    try:
        events = await drain(agent, "fan out the work")
        # Poll until all 3 children terminal. Each child is a single turn
        # so this usually finishes in a handful of event-loop ticks.
        for _ in range(300):
            all_done = all(
                r.status in ("completed", "failed", "cancelled")
                for r in store.list()
            )
            if all_done and len(store.list()) == 3:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail(
                f"not all children terminal: "
                f"{[(r.agent_type, r.status) for r in store.list()]}"
            )
    finally:
        await agent.aclose()

    recs = store.list()
    assert len(recs) == 3
    # Each type is represented exactly once.
    by_type = {r.agent_type: r for r in recs}
    assert set(by_type) == {"explore", "verify", "plan"}
    # Scripted reply survived to final_result.
    assert by_type["explore"].final_result == "scan complete"
    assert by_type["verify"].final_result == "VERDICT: PASS all checks green"
    assert by_type["plan"].final_result == "plan: do X"

    # The parent's task_list call returned the fleet.
    from aura.schemas.events import ToolCallCompleted

    list_events = [
        e for e in events
        if isinstance(e, ToolCallCompleted) and e.name == "task_list"
    ]
    assert len(list_events) == 1
    out = list_events[0].output
    assert isinstance(out, dict)
    assert out["counts"]["completed"] == 3


# ---------------------------------------------------------------------------
# Test 3 — task_stop cancels a running child mid-flight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_stop_cancels_running_subagent(tmp_path: Path) -> None:
    # Parent: turn 1 spawns a child; turn 2 calls task_stop with the id.
    parent_turn1 = FakeTurn(
        message=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_spawn",
                    "name": "task_create",
                    "args": {
                        "description": "long",
                        "prompt": "long work",
                        "agent_type": "explore",
                    },
                }
            ],
        )
    )
    # The second turn's tool_call must reference the actual task_id.
    agent, _ = build_integration_agent(tmp_path, [parent_turn1])
    store = agent._tasks_store

    # Child model that sleeps "forever".
    class _SleepyChild(FakeChatModel):
        def __init__(self) -> None:
            super().__init__(turns=[])

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            self.__dict__["ainvoke_calls"] += 1
            await asyncio.sleep(10.0)  # much longer than the test's timeout
            raise RuntimeError("child should have been cancelled")

    agent._subagent_factory._model_factory = lambda: _SleepyChild()

    # Swap the parent model with one that emits turn 1, then turn 2
    # using the live task_id.
    class _ParentThatStops(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            self.__dict__["ainvoke_calls"] += 1
            call_n = self.__dict__["ainvoke_calls"]
            if call_n == 1:
                return ChatResult(
                    generations=[
                        ChatGeneration(message=parent_turn1.message)
                    ]
                )
            if call_n == 2:
                task_id = store.list()[0].id
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "id": "tc_stop",
                                        "name": "task_stop",
                                        "args": {"task_id": task_id},
                                    }
                                ],
                            )
                        )
                    ]
                )
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content="cancelled"))
                ]
            )

    agent._model = _ParentThatStops()
    agent._loop = agent._build_loop()

    try:
        # The whole loop should finish within 1s — task_stop awaits child
        # unwind with a ~2s internal budget. We assert completion within 3s.
        await asyncio.wait_for(drain(agent, "spawn then stop"), timeout=3.0)
    finally:
        await agent.aclose()

    recs = store.list()
    assert len(recs) == 1
    assert recs[0].status == "cancelled"
    # The underlying asyncio.Task must be done (cancelled).
    handle = agent._running_tasks.get(recs[0].id)
    # The done-callback usually pops the entry; if it's still there, assert
    # it's terminal. Either state is acceptable — "gone or done".
    assert handle is None or handle.done()
