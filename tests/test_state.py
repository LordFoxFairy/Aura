"""Tests for aura.core.state.LoopState."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from aura.core.loop import AgentLoop
from aura.core.registry import ToolRegistry
from aura.core.state import LoopState
from tests.conftest import FakeChatModel, FakeTurn


def test_loop_state_defaults() -> None:
    s = LoopState()
    assert s.turn_count == 0
    assert s.total_tokens_used == 0
    assert s.custom == {}


def test_loop_state_turn_count_is_mutable() -> None:
    s = LoopState()
    s.turn_count = 5
    assert s.turn_count == 5


def test_loop_state_custom_dict_isolated_per_instance() -> None:
    a = LoopState()
    b = LoopState()
    a.custom["key"] = "value"
    assert "key" not in b.custom


def test_agent_loop_exposes_state_via_property() -> None:
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="hi"))])
    loop = AgentLoop(model=model, registry=ToolRegistry(()))
    assert loop.state is loop.state
    assert isinstance(loop.state, LoopState)
