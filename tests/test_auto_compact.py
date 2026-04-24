"""Auto-compact — Agent.astream calls self.compact(source='auto') when
``state.total_tokens_used`` crosses ``auto_compact_threshold`` after the
current turn completes.

The invariant we test:

- Trigger is post-turn only — never mid-stream, never on cancellation.
- Zero threshold disables the feature entirely.
- A journal event ``auto_compact_triggered`` is emitted right before the
  compact call so an audit trail reconstructs the decision.

We patch ``Agent.compact`` to a spy in the first four tests so we can
assert call behavior without paying the runtime + FakeChatModel-scripting
cost of actually running compaction. The journal-event test uses the real
compact path against a short history — the short-history branch inside
``run_compact`` no-ops after at most one summary invocation, which keeps
the test fast.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.compact import CompactResult
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _minimal_config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db")


def _agent(
    tmp_path: Path, *, threshold: int, turns: list[FakeTurn] | None = None,
) -> Agent:
    model = FakeChatModel(
        turns=turns or [FakeTurn(AIMessage(content="done"))],
    )
    return Agent(
        config=_minimal_config(),
        model=model,
        storage=_storage(tmp_path),
        auto_compact_threshold=threshold,
    )


@pytest.mark.asyncio
async def test_auto_compact_fires_when_threshold_crossed(tmp_path: Path) -> None:
    agent = _agent(tmp_path, threshold=50)
    agent._state.total_tokens_used = 100

    compact_calls: list[str] = []

    async def _spy(self: Agent, *, source: str = "manual") -> CompactResult:
        compact_calls.append(source)
        return CompactResult(
            before_tokens=0, after_tokens=0,
            source=source,  # type: ignore[arg-type]
        )

    with patch.object(Agent, "compact", _spy):
        async for _ in agent.astream("hi"):
            pass

    assert compact_calls == ["auto"]
    await agent.aclose()


@pytest.mark.asyncio
async def test_auto_compact_not_fired_below_threshold(tmp_path: Path) -> None:
    agent = _agent(tmp_path, threshold=100)
    agent._state.total_tokens_used = 50

    compact_calls: list[str] = []

    async def _spy(self: Agent, *, source: str = "manual") -> CompactResult:
        compact_calls.append(source)
        return CompactResult(
            before_tokens=0, after_tokens=0,
            source=source,  # type: ignore[arg-type]
        )

    with patch.object(Agent, "compact", _spy):
        async for _ in agent.astream("hi"):
            pass

    assert compact_calls == []
    await agent.aclose()


@pytest.mark.asyncio
async def test_auto_compact_disabled_when_threshold_zero(tmp_path: Path) -> None:
    agent = _agent(tmp_path, threshold=0)
    # Even a massive token count must not trigger auto-compact when disabled.
    agent._state.total_tokens_used = 10_000_000

    compact_calls: list[str] = []

    async def _spy(self: Agent, *, source: str = "manual") -> CompactResult:
        compact_calls.append(source)
        return CompactResult(
            before_tokens=0, after_tokens=0,
            source=source,  # type: ignore[arg-type]
        )

    with patch.object(Agent, "compact", _spy):
        async for _ in agent.astream("hi"):
            pass

    assert compact_calls == []
    await agent.aclose()


@pytest.mark.asyncio
async def test_auto_compact_journal_event(tmp_path: Path) -> None:
    # Real journal wired to a temp file so we can inspect the emitted event.
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)
    try:
        agent = _agent(tmp_path, threshold=10)
        agent._state.total_tokens_used = 42

        compact_calls: list[str] = []

        async def _spy(self: Agent, *, source: str = "manual") -> CompactResult:
            compact_calls.append(source)
            return CompactResult(
                before_tokens=0, after_tokens=0,
                source=source,  # type: ignore[arg-type]
            )

        with patch.object(Agent, "compact", _spy):
            async for _ in agent.astream("hi"):
                pass

        events = [
            json.loads(line)
            for line in log_path.read_text().strip().split("\n")
            if line
        ]
        triggered = [e for e in events if e["event"] == "auto_compact_triggered"]
        assert len(triggered) == 1
        ev = triggered[0]
        assert ev["tokens"] == 42
        assert ev["threshold"] == 10
        assert ev["session"] == "default"
        assert compact_calls == ["auto"]
        await agent.aclose()
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_auto_compact_skipped_on_cancel(tmp_path: Path) -> None:
    # A cancelled turn must skip the else-branch entirely — including the
    # auto-compact check. Otherwise a cancel-while-over-threshold would run
    # compact against a half-torn-down state, which is exactly the footgun
    # we promise to avoid.
    class _SlowFake(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **_: Any,
        ) -> Any:
            await asyncio.sleep(10)
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **_,
            )

    agent = Agent(
        config=_minimal_config(),
        model=_SlowFake(turns=[FakeTurn(AIMessage(content="never"))]),  # type: ignore[call-arg]
        storage=_storage(tmp_path),
        auto_compact_threshold=10,
    )
    # Over the threshold BEFORE the turn even starts.
    agent._state.total_tokens_used = 100

    compact_calls: list[str] = []

    async def _spy(self: Agent, *, source: str = "manual") -> CompactResult:
        compact_calls.append(source)
        return CompactResult(
            before_tokens=0, after_tokens=0,
            source=source,  # type: ignore[arg-type]
        )

    with patch.object(Agent, "compact", _spy):
        async def _run() -> None:
            async for _ in agent.astream("slow"):
                pass

        task = asyncio.ensure_future(_run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert compact_calls == []
    await agent.aclose()
