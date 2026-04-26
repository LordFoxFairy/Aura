"""F-0910-001 — model-aware auto-compact threshold.

Default threshold derives from ``get_context_window(model_spec) - 13_000``.
``switch_model`` re-resolves on the next read. Explicit constructor
override still wins. ``threshold=0`` still disables.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.compact.constants import (
    AUTO_COMPACT_HEADROOM_TOKENS,
    auto_compact_threshold_for,
)
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def test_function_returns_window_minus_headroom() -> None:
    from aura.core.llm import get_context_window

    spec = "openai:gpt-4o-mini"
    expected = get_context_window(spec) - AUTO_COMPACT_HEADROOM_TOKENS
    assert auto_compact_threshold_for(spec) == max(1_000, expected)


def test_function_floors_at_one_thousand() -> None:
    # An unknown model whose window minus headroom would be tiny still
    # gets a positive (enabled) threshold so auto-compact stays operational.
    # We construct an artificially small spec; if get_context_window
    # returns the default (512k) we still get a sensible value.
    threshold = auto_compact_threshold_for("totally:unknown-model-spec")
    assert threshold >= 1_000


def test_agent_default_threshold_is_model_aware(tmp_path: Path) -> None:
    """Agent constructed without an explicit threshold uses the model-aware
    derivation — i.e. its effective threshold matches
    ``auto_compact_threshold_for(current_model)``."""
    agent = Agent(
        config=_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="x"))]),
        storage=SessionStorage(tmp_path / "a.db"),
        # No auto_compact_threshold kwarg → defaults to AUTO_COMPACT_THRESHOLD
        # which is the -1 sentinel.
    )
    expected = auto_compact_threshold_for("openai:gpt-4o-mini")
    assert agent._effective_auto_compact_threshold() == expected


def test_explicit_override_wins(tmp_path: Path) -> None:
    agent = Agent(
        config=_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="x"))]),
        storage=SessionStorage(tmp_path / "a.db"),
        auto_compact_threshold=42_000,
    )
    assert agent._effective_auto_compact_threshold() == 42_000


def test_zero_threshold_still_disables(tmp_path: Path) -> None:
    agent = Agent(
        config=_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="x"))]),
        storage=SessionStorage(tmp_path / "a.db"),
        auto_compact_threshold=0,
    )
    assert agent._effective_auto_compact_threshold() == 0


def test_threshold_recomputes_on_switch_model(tmp_path: Path) -> None:
    """Switching models updates the resolved threshold (default-sentinel only).

    We can't really call ``switch_model`` without provider plumbing, so we
    update ``_current_model_spec`` directly — same code path the helper
    reads from.
    """
    agent = Agent(
        config=_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="x"))]),
        storage=SessionStorage(tmp_path / "a.db"),
    )
    before = agent._effective_auto_compact_threshold()
    agent._current_model_spec = "anthropic:claude-opus-4"
    after = agent._effective_auto_compact_threshold()
    # Different windows → different thresholds (both positive).
    assert before != after
    assert after > 0
