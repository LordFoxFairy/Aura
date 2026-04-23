"""Tests for aura.core.hooks.HookChain."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks import HookChain
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool


class _P(BaseModel):
    x: int = 0


def _noop(x: int = 0) -> dict[str, Any]:
    return {}


_stub_tool: BaseTool = build_tool(
    name="stub",
    description="stub",
    args_schema=_P,
    func=_noop,
)


@pytest.mark.asyncio
async def test_hookchain_empty_is_noop() -> None:
    chain = HookChain()
    history: list[BaseMessage] = []
    ai_msg = AIMessage(content="hi")
    args: dict[str, Any] = {}
    result = ToolResult(ok=True, output={})
    state = LoopState()

    await chain.run_pre_model(history=history, state=state)
    await chain.run_post_model(ai_message=ai_msg, history=history, state=state)
    decision = await chain.run_pre_tool(tool=_stub_tool, args=args, state=state)
    final = await chain.run_post_tool(
        tool=_stub_tool, args=args, result=result, state=state
    )

    assert history == []
    assert decision is None
    assert final is result


@pytest.mark.asyncio
async def test_pre_model_sees_history_and_can_mutate() -> None:
    async def inject(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        history.append(SystemMessage(content="injected"))

    chain = HookChain(pre_model=[inject])
    history: list[BaseMessage] = []
    await chain.run_pre_model(history=history, state=LoopState())

    assert len(history) == 1
    assert isinstance(history[0], SystemMessage)
    assert history[0].content == "injected"


@pytest.mark.asyncio
async def test_post_model_sees_ai_message() -> None:
    captured: list[AIMessage] = []

    async def capture(
        *, ai_message: AIMessage, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        captured.append(ai_message)

    chain = HookChain(post_model=[capture])
    ai_msg = AIMessage(content="test")
    await chain.run_post_model(ai_message=ai_msg, history=[], state=LoopState())

    assert len(captured) == 1
    assert captured[0] is ai_msg


@pytest.mark.asyncio
async def test_pre_tool_short_circuits_with_tool_result() -> None:
    denied = ToolResult(ok=False, error="denied")

    async def deny(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> ToolResult | None:
        return denied

    chain = HookChain(pre_tool=[deny])
    result = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert result is denied


@pytest.mark.asyncio
async def test_pre_tool_first_short_circuit_wins() -> None:
    call_log: list[str] = []
    decision = ToolResult(ok=False, error="first")

    async def first(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> ToolResult | None:
        call_log.append("first")
        return decision

    async def second(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> ToolResult | None:
        call_log.append("second")
        return ToolResult(ok=False, error="second")

    chain = HookChain(pre_tool=[first, second])
    result = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert result is decision
    assert call_log == ["first"]


@pytest.mark.asyncio
async def test_post_tool_chains_in_order() -> None:
    async def append_a(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult, state: LoopState,
        **_: object,
    ) -> ToolResult:
        out = list(result.output) if isinstance(result.output, list) else []
        out.append("a")
        return ToolResult(ok=True, output=out)

    async def append_b(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult, state: LoopState,
        **_: object,
    ) -> ToolResult:
        out = list(result.output) if isinstance(result.output, list) else []
        out.append("b")
        return ToolResult(ok=True, output=out)

    chain = HookChain(post_tool=[append_a, append_b])
    final = await chain.run_post_tool(
        tool=_stub_tool, args={}, result=ToolResult(ok=True, output=[]), state=LoopState()
    )

    assert final.output == ["a", "b"]


@pytest.mark.asyncio
async def test_multiple_hooks_of_same_type_run_in_registration_order() -> None:
    call_log: list[str] = []

    async def first(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        call_log.append("first")

    async def second(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        call_log.append("second")

    chain = HookChain(pre_model=[first, second])
    await chain.run_pre_model(history=[], state=LoopState())

    assert call_log == ["first", "second"]


@pytest.mark.asyncio
async def test_pre_tool_chain_returns_none_when_all_return_none() -> None:
    async def pass_through(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> ToolResult | None:
        return None

    chain = HookChain(pre_tool=[pass_through, pass_through])
    result = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert result is None


@pytest.mark.asyncio
async def test_hooks_receive_state_kwarg() -> None:
    received: list[LoopState] = []

    async def capture(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        received.append(state)

    hooks = HookChain(pre_model=[capture])
    s = LoopState()
    await hooks.run_pre_model(history=[], state=s)

    assert received == [s]


def test_pre_model_hook_protocol_accepts_correct_signature() -> None:
    async def ok_hook(*, history: list[BaseMessage], state: LoopState, **_: object) -> None:
        return None

    chain = HookChain(pre_model=[ok_hook])
    assert len(chain.pre_model) == 1


def test_post_tool_hook_protocol_accepts_correct_signature() -> None:
    async def ok_hook(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult, state: LoopState,
        **_: object,
    ) -> ToolResult:
        return result

    chain = HookChain(post_tool=[ok_hook])
    assert len(chain.post_tool) == 1


# ---------------------------------------------------------------------------
# Session-cycle hooks — pre_session / post_session / post_subagent /
# pre_compact / pre_user_prompt. Defaults stay empty; run_X iterates in
# registration order; merge concatenates all 9 slots.
# ---------------------------------------------------------------------------


def test_hookchain_defaults_include_all_nine_slots() -> None:
    # Regression guard: a new hook slot must NOT be introduced as a
    # non-field (e.g. module-level list) — defaults come through the
    # dataclass field(default_factory=list) machinery so every consumer
    # sees the same empty-list contract.
    chain = HookChain()
    assert chain.pre_model == []
    assert chain.post_model == []
    assert chain.pre_tool == []
    assert chain.post_tool == []
    assert chain.pre_session == []
    assert chain.post_session == []
    assert chain.post_subagent == []
    assert chain.pre_compact == []
    assert chain.pre_user_prompt == []


@pytest.mark.asyncio
async def test_pre_session_fires_with_session_id_and_cwd() -> None:
    captured: list[dict[str, Any]] = []

    async def hook(*, session_id: str, cwd: object, **_: object) -> None:
        captured.append({"session_id": session_id, "cwd": cwd})

    chain = HookChain(pre_session=[hook])
    await chain.run_pre_session(session_id="sess-1", cwd="/tmp/proj")

    assert captured == [{"session_id": "sess-1", "cwd": "/tmp/proj"}]


@pytest.mark.asyncio
async def test_post_session_fires_with_session_id_and_cwd() -> None:
    captured: list[dict[str, Any]] = []

    async def hook(*, session_id: str, cwd: object, **_: object) -> None:
        captured.append({"session_id": session_id, "cwd": cwd})

    chain = HookChain(post_session=[hook])
    await chain.run_post_session(session_id="sess-2", cwd="/var/x")

    assert captured == [{"session_id": "sess-2", "cwd": "/var/x"}]


@pytest.mark.asyncio
async def test_post_subagent_fires_with_all_kwargs() -> None:
    captured: list[dict[str, Any]] = []

    async def hook(
        *,
        task_id: str,
        status: str,
        final_text: str,
        error: str | None,
        **_: object,
    ) -> None:
        captured.append(
            {
                "task_id": task_id,
                "status": status,
                "final_text": final_text,
                "error": error,
            }
        )

    chain = HookChain(post_subagent=[hook])
    await chain.run_post_subagent(
        task_id="t-123", status="completed",
        final_text="done.", error=None,
    )

    assert captured == [
        {
            "task_id": "t-123",
            "status": "completed",
            "final_text": "done.",
            "error": None,
        }
    ]


@pytest.mark.asyncio
async def test_pre_compact_fires_with_state_and_trigger() -> None:
    captured: list[dict[str, Any]] = []

    async def hook(*, state: LoopState, trigger: str, **_: object) -> None:
        captured.append({"state": state, "trigger": trigger})

    chain = HookChain(pre_compact=[hook])
    s = LoopState()
    await chain.run_pre_compact(state=s, trigger="auto")

    assert len(captured) == 1
    assert captured[0]["state"] is s
    assert captured[0]["trigger"] == "auto"


@pytest.mark.asyncio
async def test_pre_user_prompt_fires_with_prompt_and_state() -> None:
    captured: list[dict[str, Any]] = []

    async def hook(*, prompt: str, state: LoopState, **_: object) -> None:
        captured.append({"prompt": prompt, "state": state})

    chain = HookChain(pre_user_prompt=[hook])
    s = LoopState()
    await chain.run_pre_user_prompt(prompt="hello world", state=s)

    assert len(captured) == 1
    assert captured[0]["prompt"] == "hello world"
    assert captured[0]["state"] is s


@pytest.mark.asyncio
async def test_session_cycle_hooks_fire_in_registration_order() -> None:
    # One test covers ordering for all 5 new hook types — their machinery
    # is identical (iterate the list, await each), so a single spy per
    # type would be overkill.
    log: list[str] = []

    async def s1(**_: object) -> None:
        log.append("s1")

    async def s2(**_: object) -> None:
        log.append("s2")

    chain = HookChain(
        pre_session=[s1, s2],
        post_session=[s1, s2],
        post_subagent=[s1, s2],
        pre_compact=[s1, s2],
        pre_user_prompt=[s1, s2],
    )
    await chain.run_pre_session(session_id="x", cwd="/")
    await chain.run_post_session(session_id="x", cwd="/")
    await chain.run_post_subagent(
        task_id="t", status="completed", final_text="", error=None,
    )
    await chain.run_pre_compact(state=LoopState(), trigger="manual")
    await chain.run_pre_user_prompt(prompt="p", state=LoopState())

    # 5 hook types * 2 hooks each = 10 entries, ordered
    assert log == ["s1", "s2"] * 5


def test_merge_concatenates_all_nine_slots() -> None:
    async def _noop(**_: object) -> None:
        return None

    async def _pre_tool(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState,
        **_: object,
    ) -> ToolResult | None:
        return None

    async def _post_tool(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult,
        state: LoopState, **_: object,
    ) -> ToolResult:
        return result

    left = HookChain(
        pre_model=[_noop],
        post_model=[_noop],
        pre_tool=[_pre_tool],
        post_tool=[_post_tool],
        pre_session=[_noop],
        post_session=[_noop],
        post_subagent=[_noop],
        pre_compact=[_noop],
        pre_user_prompt=[_noop],
    )
    right = HookChain(
        pre_model=[_noop],
        post_model=[_noop],
        pre_tool=[_pre_tool],
        post_tool=[_post_tool],
        pre_session=[_noop],
        post_session=[_noop],
        post_subagent=[_noop],
        pre_compact=[_noop],
        pre_user_prompt=[_noop],
    )
    merged = left.merge(right)
    # Every slot should carry 2 hooks after merge — no slot dropped.
    assert len(merged.pre_model) == 2
    assert len(merged.post_model) == 2
    assert len(merged.pre_tool) == 2
    assert len(merged.post_tool) == 2
    assert len(merged.pre_session) == 2
    assert len(merged.post_session) == 2
    assert len(merged.post_subagent) == 2
    assert len(merged.pre_compact) == 2
    assert len(merged.pre_user_prompt) == 2
    # Non-destructive — originals untouched.
    assert len(left.pre_session) == 1
    assert len(right.post_subagent) == 1


@pytest.mark.asyncio
async def test_empty_session_cycle_hooks_are_noop() -> None:
    # All 5 new run_X methods with empty slots must be callable without
    # raising — matches the existing empty-pre_model etc. contract so
    # consumers don't have to null-check before firing.
    chain = HookChain()
    await chain.run_pre_session(session_id="s", cwd="/")
    await chain.run_post_session(session_id="s", cwd="/")
    await chain.run_post_subagent(
        task_id="t", status="completed", final_text="", error=None,
    )
    await chain.run_pre_compact(state=LoopState(), trigger="manual")
    await chain.run_pre_user_prompt(prompt="p", state=LoopState())


# ---------------------------------------------------------------------------
# Lifecycle integration — each hook fires at its documented call site.
# These tests pair each run_X machinery with the code path that owns the
# fire: Agent.bootstrap/shutdown, AgentLoop.run_turn, run_compact, run_task.
# ---------------------------------------------------------------------------


from pathlib import Path  # noqa: E402

from aura.config.schema import AuraConfig  # noqa: E402
from aura.core.agent import Agent  # noqa: E402
from aura.core.persistence.storage import SessionStorage  # noqa: E402
from tests.conftest import FakeChatModel, FakeTurn  # noqa: E402


def _minimal_config(enabled: list[str] | None = None) -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": enabled if enabled is not None else []},
    })


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db")


@pytest.mark.asyncio
async def test_agent_bootstrap_fires_pre_session(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    async def hook(*, session_id: str, cwd: object, **_: object) -> None:
        captured.append({"session_id": session_id, "cwd": cwd})

    chain = HookChain(pre_session=[hook])
    agent = Agent(
        config=_minimal_config(),
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
        hooks=chain,
        session_id="sess-boot",
    )
    await agent.bootstrap()
    agent.close()

    assert len(captured) == 1
    assert captured[0]["session_id"] == "sess-boot"


@pytest.mark.asyncio
async def test_agent_shutdown_fires_post_session(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    async def hook(*, session_id: str, cwd: object, **_: object) -> None:
        captured.append({"session_id": session_id, "cwd": cwd})

    chain = HookChain(post_session=[hook])
    agent = Agent(
        config=_minimal_config(),
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
        hooks=chain,
        session_id="sess-shut",
    )
    await agent.shutdown()  # close is invoked inside shutdown

    assert len(captured) == 1
    assert captured[0]["session_id"] == "sess-shut"


@pytest.mark.asyncio
async def test_run_turn_fires_pre_user_prompt(tmp_path: Path) -> None:
    captured: list[str] = []

    async def hook(*, prompt: str, state: LoopState, **_: object) -> None:
        captured.append(prompt)

    chain = HookChain(pre_user_prompt=[hook])
    # A single-turn conversation with a natural-stop AI message (no tool
    # calls) keeps the run_turn loop tidy.
    agent = Agent(
        config=_minimal_config(),
        model=FakeChatModel(turns=[FakeTurn(message=AIMessage(content="hi"))]),
        storage=_storage(tmp_path),
        hooks=chain,
    )
    async for _ev in agent.astream("hello"):
        pass
    agent.close()

    assert captured == ["hello"]


@pytest.mark.asyncio
async def test_compact_fires_pre_compact_before_summary(tmp_path: Path) -> None:
    # We can't easily stub the summary LLM call in a compact run, so
    # instead we route through a history that's too short to summarize —
    # pre_compact STILL fires (it's at entry) and run_compact no-ops.
    captured: list[dict[str, Any]] = []

    async def hook(*, state: LoopState, trigger: str, **_: object) -> None:
        captured.append({"trigger": trigger})

    chain = HookChain(pre_compact=[hook])
    agent = Agent(
        config=_minimal_config(),
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
        hooks=chain,
    )
    await agent.compact(source="manual")
    agent.close()

    assert captured == [{"trigger": "manual"}]


@pytest.mark.asyncio
async def test_run_task_fires_post_subagent_on_completion(
    tmp_path: Path,
) -> None:
    # Dispatch a subagent via run_task directly so we don't need to
    # spin up the whole task_create fire-and-forget path. The subagent
    # streams one natural-stop turn, mark_completed fires, then the hook.
    from aura.core.tasks.factory import SubagentFactory
    from aura.core.tasks.run import run_task
    from aura.core.tasks.store import TasksStore

    captured: list[dict[str, Any]] = []

    async def hook(
        *,
        task_id: str,
        status: str,
        final_text: str,
        error: str | None,
        **_: object,
    ) -> None:
        captured.append(
            {
                "task_id": task_id,
                "status": status,
                "final_text": final_text,
                "error": error,
            }
        )

    chain = HookChain(post_subagent=[hook])
    store = TasksStore()
    record = store.create(
        description="sub", prompt="x", agent_type="general-purpose",
    )

    class _FakeFactory(SubagentFactory):
        def __init__(self) -> None:
            pass  # skip base __init__

        def spawn(
            self, prompt: str, allowed_tools: list[str] | None = None,
            *, agent_type: str = "general-purpose",
        ) -> Agent:
            return Agent(
                config=_minimal_config(),
                model=FakeChatModel(
                    turns=[FakeTurn(message=AIMessage(content="done-text"))],
                ),
                storage=_storage(tmp_path / f"sub-{record.id[:6]}"),
            )

    await run_task(store, _FakeFactory(), record.id, parent_hooks=chain)

    assert len(captured) == 1
    assert captured[0]["task_id"] == record.id
    assert captured[0]["status"] == "completed"
    assert captured[0]["final_text"] == "done-text"
    assert captured[0]["error"] is None


@pytest.mark.asyncio
async def test_run_task_fires_post_subagent_on_failure(tmp_path: Path) -> None:
    # Force ``factory.spawn`` to raise; run_task should catch, mark
    # failed, fire the hook with status="failed" + the error string.
    from aura.core.tasks.factory import SubagentFactory
    from aura.core.tasks.run import run_task
    from aura.core.tasks.store import TasksStore

    captured: list[dict[str, Any]] = []

    async def hook(
        *, task_id: str, status: str, final_text: str, error: str | None,
        **_: object,
    ) -> None:
        captured.append(
            {"task_id": task_id, "status": status, "error": error},
        )

    chain = HookChain(post_subagent=[hook])
    store = TasksStore()
    record = store.create(
        description="sub", prompt="x", agent_type="general-purpose",
    )

    class _FailingFactory(SubagentFactory):
        def __init__(self) -> None:
            pass

        def spawn(
            self, prompt: str, allowed_tools: list[str] | None = None,
            *, agent_type: str = "general-purpose",
        ) -> Agent:
            raise RuntimeError("spawn boom")

    await run_task(
        store, _FailingFactory(), record.id, parent_hooks=chain,
    )

    assert len(captured) == 1
    assert captured[0]["status"] == "failed"
    assert "spawn boom" in (captured[0]["error"] or "")
