"""F-01-003 — ``max_turns`` is per-user-turn, not per-Agent-lifetime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from aura.schemas.events import Final
from tests.conftest import FakeChatModel, FakeTurn


def _minimal_config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _make_tool_call_turn(call_id: str, tool_name: str = "echo") -> FakeTurn:
    return FakeTurn(message=AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {"msg": "hi"}, "id": call_id}],
    ))


@pytest.mark.asyncio
async def test_max_turns_resets_per_astream_call(tmp_path: Path) -> None:
    """Two independent astream calls each get a fresh max_turns budget.

    Pre-fix: ``LoopState.turn_count`` accumulated across astream calls,
    so after enough cumulative rounds the second user prompt would
    trip ``max_turns_reached`` immediately. Now Agent.astream zeroes
    the counter at entry, mirroring claude-code's local
    ``turnCount = 1`` initialisation at every ``query`` entry.
    """
    # Two single-round astream calls (model emits content, no tool_calls)
    # back-to-back — both must finish naturally with no max_turns trip.
    turns = [
        FakeTurn(AIMessage(content="reply-1")),
        FakeTurn(AIMessage(content="reply-2")),
    ]
    model = FakeChatModel(turns=turns)
    storage = SessionStorage(tmp_path / "s.db")
    agent = Agent(
        config=_minimal_config(), model=model, storage=storage,
    )

    finals_first: list[Any] = []
    async for ev in agent.astream("first"):
        if isinstance(ev, Final):
            finals_first.append(ev)
    assert finals_first
    assert finals_first[-1].reason == "natural"
    assert agent.state.turn_count == 1  # one model round this astream

    finals_second: list[Any] = []
    async for ev in agent.astream("second"):
        if isinstance(ev, Final):
            finals_second.append(ev)
    assert finals_second
    assert finals_second[-1].reason == "natural"
    # Counter reset — the second astream did one round, not two.
    assert agent.state.turn_count == 1
    await agent.aclose()


@pytest.mark.asyncio
async def test_max_turns_does_not_trip_after_long_prior_session(
    tmp_path: Path,
) -> None:
    """A long prior session does NOT shorten the next astream's budget.

    Drive 5 user turns, each one model round. With a per-Agent-lifetime
    counter, turn_count would be 5 after the 5th astream and a
    max_turns=3 cap would trip on the next astream's FIRST round. With
    the per-astream reset, the next astream still gets a full 3-round
    budget.
    """
    # Each astream call is a single-round natural reply; fixture cap
    # is 3 (constructor default 50, but we'll override below to test
    # the boundary).
    turns = [FakeTurn(AIMessage(content=f"r{i}")) for i in range(6)]
    model = FakeChatModel(turns=turns)
    storage = SessionStorage(tmp_path / "s.db")
    agent = Agent(
        config=_minimal_config(), model=model, storage=storage,
    )
    # Squeeze max_turns down to 3 so the test boundary is observable.
    agent._loop._max_turns = 3

    # Five back-to-back single-round user turns — all must succeed.
    for i in range(5):
        finals: list[Any] = []
        async for ev in agent.astream(f"prompt-{i}"):
            if isinstance(ev, Final):
                finals.append(ev)
        assert finals[-1].reason == "natural", (
            f"prompt-{i} unexpectedly hit max_turns "
            f"(turn_count={agent.state.turn_count})"
        )
    await agent.aclose()


@pytest.mark.asyncio
async def test_max_turns_still_caps_within_a_single_user_turn(
    tmp_path: Path,
) -> None:
    """Inside a single astream call, the cap STILL fires.

    Drive 10 tool-call rounds with cap=3 — must stop at 3 with
    ``Final.reason == "max_turns"``. Symmetric guard for the reset
    fix: zeroing on entry must NOT also disable the cap.
    """
    # 10 rounds of tool_calls that the loop will cap at 3.
    # Build via a tool name that's a no-op echo.
    from pydantic import BaseModel

    from aura.tools.base import build_tool

    class _EchoArgs(BaseModel):
        msg: str = ""

    def _echo(msg: str = "") -> dict[str, str]:
        return {"echoed": msg}

    echo_tool = build_tool(
        name="echo",
        description="echo",
        args_schema=_EchoArgs,
        func=_echo,
        is_read_only=True,
        is_concurrency_safe=True,
    )

    turns = [_make_tool_call_turn(f"tc_{i}") for i in range(10)]
    model = FakeChatModel(turns=turns)
    storage = SessionStorage(tmp_path / "s.db")
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["echo"]},
    })
    agent = Agent(
        config=cfg, model=model, storage=storage,
        available_tools={"echo": echo_tool},
    )
    agent._loop._max_turns = 3

    finals: list[Any] = []
    async for ev in agent.astream("infinite"):
        if isinstance(ev, Final):
            finals.append(ev)
    assert finals
    assert finals[-1].reason == "max_turns"
    # After hitting the cap, turn_count == max_turns (3).
    assert agent.state.turn_count == 3
    await agent.aclose()
