"""Integration tests: HookChain wired into AgentLoop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel

from aura.core.budget import make_size_budget_hook
from aura.core.events import AgentEvent, ToolCallCompleted
from aura.core.hooks import HookChain
from aura.core.loop import AgentLoop
from aura.core.registry import ToolRegistry
from aura.core.state import LoopState
from aura.tools.base import AuraTool, ToolResult, build_tool
from tests.conftest import FakeChatModel, FakeTurn


class _EchoParams(BaseModel):
    msg: str


_acall_counter = 0


async def _echo_call(params: BaseModel) -> ToolResult:
    global _acall_counter
    _acall_counter += 1
    assert isinstance(params, _EchoParams)
    return ToolResult(ok=True, output={"echoed": params.msg})


_echo_tool: AuraTool = build_tool(
    name="echo",
    description="echoes input",
    input_model=_EchoParams,
    call=_echo_call,
    is_read_only=True,
    is_concurrency_safe=True,
)


def _tool_turn(msg: str = "hi") -> FakeTurn:
    return FakeTurn(message=AIMessage(
        content="",
        tool_calls=[{"name": "echo", "args": {"msg": msg}, "id": "tc_1"}],
    ))


def _final_turn(text: str = "done") -> FakeTurn:
    return FakeTurn(message=AIMessage(content=text))


@pytest.mark.asyncio
async def test_pre_model_fires_before_each_ainvoke() -> None:
    counter = {"n": 0}

    async def count(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        counter["n"] += 1

    model = FakeChatModel(turns=[_tool_turn(), _final_turn()])
    registry = ToolRegistry([_echo_tool])
    hooks = HookChain(pre_model=[count])
    loop = AgentLoop(model=model, registry=registry, hooks=hooks)

    async for _ in loop.run_turn(user_prompt="go", history=[]):
        pass

    assert counter["n"] == 2


@pytest.mark.asyncio
async def test_post_model_fires_after_each_ainvoke_with_ai_message() -> None:
    seen: list[AIMessage] = []

    async def capture(
        *, ai_message: AIMessage, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        seen.append(ai_message)

    model = FakeChatModel(turns=[_tool_turn(), _final_turn()])
    registry = ToolRegistry([_echo_tool])
    hooks = HookChain(post_model=[capture])
    loop = AgentLoop(model=model, registry=registry, hooks=hooks)

    async for _ in loop.run_turn(user_prompt="go", history=[]):
        pass

    assert len(seen) == 2
    assert seen[0] is not seen[1]


@pytest.mark.asyncio
async def test_pre_tool_fires_before_acall_and_can_deny() -> None:
    global _acall_counter
    _acall_counter = 0
    denied = ToolResult(ok=False, error="denied")

    async def deny(
        *, tool: AuraTool, params: BaseModel, state: LoopState, **_: object
    ) -> ToolResult | None:
        return denied

    model = FakeChatModel(turns=[_tool_turn(), _final_turn()])
    registry = ToolRegistry([_echo_tool])
    hooks = HookChain(pre_tool=[deny])
    loop = AgentLoop(model=model, registry=registry, hooks=hooks)

    events: list[AgentEvent] = []
    async for ev in loop.run_turn(user_prompt="go", history=[]):
        events.append(ev)

    assert _acall_counter == 0

    completed = next(e for e in events if isinstance(e, ToolCallCompleted))
    assert completed.error == "denied"


@pytest.mark.asyncio
async def test_post_tool_fires_after_acall_and_can_rewrite_output() -> None:
    async def truncate(
        *, tool: AuraTool, params: BaseModel, result: ToolResult, state: LoopState, **_: object
    ) -> ToolResult:
        return ToolResult(ok=True, output={"truncated": True})

    model = FakeChatModel(turns=[_tool_turn(), _final_turn()])
    registry = ToolRegistry([_echo_tool])
    hooks = HookChain(post_tool=[truncate])
    loop = AgentLoop(model=model, registry=registry, hooks=hooks)

    history: list[BaseMessage] = []
    async for _ in loop.run_turn(user_prompt="go", history=history):
        pass

    import json
    raw = history[2].content
    assert isinstance(raw, str)
    tool_msg_content = json.loads(raw)
    assert tool_msg_content == {"truncated": True}


@pytest.mark.asyncio
async def test_hooks_all_fire_in_order_for_tool_turn() -> None:
    event_log: list[str] = []

    async def pre_model(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        event_log.append("pre_model")

    async def post_model(
        *, ai_message: AIMessage, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        event_log.append("post_model")

    async def pre_tool(
        *, tool: AuraTool, params: BaseModel, state: LoopState, **_: object
    ) -> ToolResult | None:
        event_log.append("pre_tool")
        return None

    async def post_tool(
        *, tool: AuraTool, params: BaseModel, result: ToolResult, state: LoopState, **_: object
    ) -> ToolResult:
        event_log.append("post_tool")
        return result

    model = FakeChatModel(turns=[_tool_turn(), _final_turn()])
    registry = ToolRegistry([_echo_tool])
    hooks = HookChain(
        pre_model=[pre_model],
        post_model=[post_model],
        pre_tool=[pre_tool],
        post_tool=[post_tool],
    )
    loop = AgentLoop(model=model, registry=registry, hooks=hooks)

    async for _ in loop.run_turn(user_prompt="go", history=[]):
        pass

    assert event_log[:4] == ["pre_model", "post_model", "pre_tool", "post_tool"]


@pytest.mark.asyncio
async def test_hooks_see_monotonic_turn_count() -> None:
    """Pre_model hooks see turn_count incrementing across turns."""
    observed: list[int] = []

    async def record(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        observed.append(state.turn_count)

    hooks = HookChain(pre_model=[record])
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"msg": "x"}, "id": "tc_1"}
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    registry = ToolRegistry([_echo_tool])
    loop = AgentLoop(model=model, registry=registry, hooks=hooks)

    history: list[BaseMessage] = []
    async for _ in loop.run_turn(user_prompt="go", history=history):
        pass

    assert observed == [1, 2]


@pytest.mark.asyncio
async def test_budget_hook_truncates_large_tool_output(tmp_path: Path) -> None:
    async def _big_call(params: BaseModel) -> ToolResult:
        return ToolResult(ok=True, output={"content": "x" * 20_000})

    class _P(BaseModel):
        pass

    big_tool: AuraTool = build_tool(
        name="big",
        description="emits large output",
        input_model=_P,
        call=_big_call,
        is_read_only=True,
    )

    hooks = HookChain(post_tool=[make_size_budget_hook(max_chars=500, spill_dir=tmp_path)])

    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "big", "args": {}, "id": "tc_1"},
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model,
        registry=ToolRegistry([big_tool]),
        hooks=hooks,
    )

    history: list[BaseMessage] = []
    async for _ in loop.run_turn(user_prompt="go", history=history):
        pass

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    payload = json.loads(str(tool_msgs[0].content))
    assert payload["truncated"] is True
    assert payload["total_chars"] > 500
    assert "spill_path" in payload
    assert Path(payload["spill_path"]).exists()
