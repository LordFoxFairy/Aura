"""F-0910-002 — auto-compact consecutive-failure circuit breaker.

Three consecutive failed auto-compact attempts must disable subsequent auto
firings on this Agent. Manual /compact bypasses the breaker. A successful
auto-compact resets the counter to 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.compact import CompactResult
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _agent(tmp_path: Path, threshold: int = 10) -> Agent:
    return Agent(
        config=_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="x"))] * 20),
        storage=SessionStorage(tmp_path / "aura.db"),
        auto_compact_threshold=threshold,
    )


def test_breaker_counter_seeded_at_session_start(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    assert agent._state.custom["consecutive_compact_failures"] == 0


@pytest.mark.asyncio
async def test_breaker_blocks_after_three_failures(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent._state.total_tokens_used = 100

    calls: list[str] = []

    async def _fail(self: Agent, *, source: str = "manual") -> CompactResult:
        calls.append(source)
        raise RuntimeError("simulated compact failure")

    with patch.object(Agent, "compact", _fail):
        for _ in range(3):
            with pytest.raises(RuntimeError):
                async for _ev in agent.astream("hi"):
                    pass

    assert agent._state.custom["consecutive_compact_failures"] == 3
    # Fourth attempt — same conditions, but breaker tripped → no more calls.
    pre = len(calls)
    with patch.object(Agent, "compact", _fail):
        async for _ev in agent.astream("hi"):
            pass
    assert len(calls) == pre, "breaker did not block 4th auto-compact"
    await agent.aclose()


@pytest.mark.asyncio
async def test_breaker_resets_on_success(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent._state.total_tokens_used = 100
    agent._state.custom["consecutive_compact_failures"] = 2

    async def _ok(self: Agent, *, source: str = "manual") -> CompactResult:
        return CompactResult(
            before_tokens=0, after_tokens=0,
            source=source,  # type: ignore[arg-type]
        )

    with patch.object(Agent, "compact", _ok):
        async for _ev in agent.astream("hi"):
            pass

    assert agent._state.custom["consecutive_compact_failures"] == 0
    await agent.aclose()


@pytest.mark.asyncio
async def test_manual_compact_bypasses_breaker(tmp_path: Path) -> None:
    """Manual ``Agent.compact()`` runs the heavy lifting via ``run_compact``,
    not the auto-compact branch in ``astream``, so the breaker never gates
    it. Tripping the counter to 99 must not stop a manual call from at
    least *attempting* to run."""
    agent = _agent(tmp_path)
    agent._state.custom["consecutive_compact_failures"] = 99

    # Short history → run_compact short-circuits to a noop, returning a
    # CompactResult without invoking the model. That's enough to confirm
    # the manual path never consulted the breaker.
    result = await agent.compact(source="manual")
    assert result is not None
    assert result.source == "manual"
    await agent.aclose()


def _read_journal_lines(log: Path) -> list[dict[str, Any]]:
    import json

    if not log.exists():
        return []
    return [
        json.loads(line)
        for line in log.read_text().splitlines()
        if line.strip()
    ]


@pytest.mark.asyncio
async def test_breaker_emits_skip_journal_event(tmp_path: Path) -> None:
    from aura.core.persistence import journal

    log = tmp_path / "audit.jsonl"
    journal.configure(log)
    try:
        agent = _agent(tmp_path)
        agent._state.total_tokens_used = 100
        agent._state.custom["consecutive_compact_failures"] = 5

        async def _ok(self: Agent, *, source: str = "manual") -> CompactResult:
            return CompactResult(
                before_tokens=0, after_tokens=0,
                source=source,  # type: ignore[arg-type]
            )

        with patch.object(Agent, "compact", _ok):
            async for _ev in agent.astream("hi"):
                pass

        events = _read_journal_lines(log)
        skips = [e for e in events if e["event"] == "auto_compact_skipped_circuit_breaker"]
        assert len(skips) == 1
        assert skips[0]["consecutive_failures"] == 5
        await agent.aclose()
    finally:
        journal.reset()
