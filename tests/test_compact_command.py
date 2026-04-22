"""Tests for the ``/compact`` slash command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from aura.cli.commands import build_default_registry, dispatch
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.compact import CompactResult
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _agent(tmp_path: Path) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="SUM"))]),
        storage=SessionStorage(tmp_path / "db"),
    )


def test_compact_command_registered_in_default_registry() -> None:
    r = build_default_registry()
    names = {c.name for c in r.list()}
    assert "/compact" in names


@pytest.mark.asyncio
async def test_compact_command_invokes_agent_compact() -> None:
    mock_agent = MagicMock(spec=Agent)
    mock_agent.compact = AsyncMock(return_value=CompactResult(
        before_tokens=1000,
        after_tokens=1100,
        source="manual",
    ))
    r = build_default_registry()
    result = await dispatch("/compact", mock_agent, r)
    mock_agent.compact.assert_awaited_once_with(source="manual")
    assert result.handled is True
    assert result.kind == "print"


@pytest.mark.asyncio
async def test_compact_command_prints_before_and_after_tokens() -> None:
    mock_agent = MagicMock(spec=Agent)
    mock_agent.compact = AsyncMock(return_value=CompactResult(
        before_tokens=12345,
        after_tokens=6789,
        source="manual",
    ))
    r = build_default_registry()
    result = await dispatch("/compact", mock_agent, r)
    assert "12345" in result.text
    assert "6789" in result.text


@pytest.mark.asyncio
async def test_compact_command_end_to_end_with_real_agent(tmp_path: Path) -> None:
    # Seed enough history to trigger actual compaction and verify history
    # shrinks end-to-end through the command.
    agent = _agent(tmp_path)
    history: list[BaseMessage] = []
    for i in range(10):
        history.append(HumanMessage(content=f"u-{i}"))
        history.append(AIMessage(content=f"a-{i}"))
    agent._storage.save(agent.session_id, history)

    r = build_default_registry()
    result = await dispatch("/compact", agent, r)
    assert result.handled and result.kind == "print"

    after = agent._storage.load(agent.session_id)
    # Tail (last 6) + summary (1) = 7 messages.
    assert len(after) == 7
    agent.close()
