"""Tests for aura.core.state.LoopState."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from aura.capabilities.tools.registry import ToolRegistry
from aura.core.loop import AgentLoop
from aura.schemas.state import LoopSlots, LoopState
from tests.conftest import FakeChatModel, FakeTurn, make_minimal_context


def test_loop_state_defaults() -> None:
    s = LoopState()
    assert s.turn_count == 0
    assert s.total_tokens_used == 0


def test_loop_state_turn_count_is_mutable() -> None:
    s = LoopState()
    s.turn_count = 5
    assert s.turn_count == 5


def test_agent_loop_exposes_state_via_property() -> None:
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="hi"))])
    loop = AgentLoop(model=model, registry=ToolRegistry(()), context=make_minimal_context())
    assert loop.state is loop.state
    assert isinstance(loop.state, LoopState)


def test_loop_state_reset_zeroes_counters() -> None:
    state = LoopState()
    state.turn_count = 42
    state.total_tokens_used = 1000

    state.reset()

    assert state.turn_count == 0
    assert state.total_tokens_used == 0


def test_loop_state_reset_preserves_instance_identity() -> None:
    """Reset mutates in place so holders of the reference see the change."""
    state = LoopState()
    original = state
    state.turn_count = 5
    state.reset()
    assert state is original
    assert state.turn_count == 0


def test_loop_state_has_default_slots() -> None:
    """LoopState() exposes a fresh LoopSlots instance on .slots."""
    state = LoopState()
    assert isinstance(state.slots, LoopSlots)
    assert state.slots == LoopSlots()


def test_loop_state_slots_isolated_per_instance() -> None:
    """Each LoopState gets its own LoopSlots — no shared default."""
    a = LoopState()
    b = LoopState()
    assert a.slots is not b.slots
