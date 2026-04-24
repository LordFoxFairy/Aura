"""Reactive recompact — when the model raises a context-overflow error,
``Agent.astream`` catches it, runs ``compact(source='reactive')``, and retries
the turn ONCE. Non-overflow errors pass through unchanged. A hard guard
prevents infinite retry loops.

All tests drive behavior through a custom FakeChatModel subclass that fails
on the first call and (optionally) succeeds on the second. That keeps the
test surface tight — the only moving piece is Agent.astream's error branch.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

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


class _RaisingModel(FakeChatModel):
    """FakeChatModel variant that raises scripted exceptions per call.

    ``errors`` is an iterable of (Exception | None). None means "fall through
    to the FakeTurn scripted message". Consumed left-to-right, same as turns.
    """

    def __init__(
        self,
        *,
        errors: Sequence[BaseException | None],
        turns: list[FakeTurn] | None = None,
    ) -> None:
        super().__init__(turns=turns or [])
        self.__dict__["_errors"] = list(errors)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["ainvoke_calls"] += 1
        errs: list[BaseException | None] = self.__dict__["_errors"]
        err = errs.pop(0) if errs else None
        if err is not None:
            raise err
        turn = self._pop_turn()
        return ChatResult(generations=[ChatGeneration(message=turn.message)])


@pytest.mark.asyncio
async def test_reactive_compact_on_context_length_error(tmp_path: Path) -> None:
    """First call raises context-overflow; compact runs; retry succeeds."""
    model = _RaisingModel(
        errors=[
            RuntimeError("400 — context_length_exceeded for model gpt-4o"),
            None,  # summary turn (triggered by compact)
            None,  # retry turn — success
        ],
        turns=[
            FakeTurn(AIMessage(content="SUMMARY")),
            FakeTurn(AIMessage(content="OK after retry")),
        ],
    )
    agent = Agent(
        config=_minimal_config(),
        model=model,
        storage=_storage(tmp_path),
        auto_compact_threshold=0,  # disable auto-compact; we test reactive only
    )
    # Seed enough history that compact is non-trivial (triggers summary turn).
    from langchain_core.messages import HumanMessage
    h: list[Any] = []
    for i in range(10):
        h.append(HumanMessage(content=f"u-{i}"))
        h.append(AIMessage(content=f"a-{i}"))
    agent._storage.save(agent.session_id, h)

    compact_calls: list[str] = []
    orig_compact = Agent.compact

    async def _spy(self: Agent, *, source: str = "manual") -> CompactResult:
        compact_calls.append(source)
        return await orig_compact(self, source=source)  # type: ignore[arg-type]

    with patch.object(Agent, "compact", _spy):
        finals = []
        async for ev in agent.astream("hi"):
            finals.append(ev)

    assert compact_calls == ["reactive"]
    # Second attempt completed — at least one event yielded after retry.
    assert finals, "expected the retry turn to yield events"
    await agent.aclose()


@pytest.mark.asyncio
async def test_reactive_compact_only_retries_once(tmp_path: Path) -> None:
    """Two successive context-overflow errors → second one propagates."""
    err1 = RuntimeError("prompt is too long: 123456 tokens")
    err2 = RuntimeError("prompt is too long: still too long")
    model = _RaisingModel(
        errors=[err1, None, err2],  # initial fail, summary, retry fails again
        turns=[FakeTurn(AIMessage(content="SUMMARY"))],
    )
    agent = Agent(
        config=_minimal_config(),
        model=model,
        storage=_storage(tmp_path),
        auto_compact_threshold=0,
    )
    from langchain_core.messages import HumanMessage
    h: list[Any] = []
    for i in range(10):
        h.append(HumanMessage(content=f"u-{i}"))
        h.append(AIMessage(content=f"a-{i}"))
    agent._storage.save(agent.session_id, h)

    with pytest.raises(RuntimeError, match="still too long"):
        async for _ in agent.astream("hi"):
            pass
    await agent.aclose()


@pytest.mark.asyncio
async def test_reactive_compact_other_error_passthrough(tmp_path: Path) -> None:
    """Unrelated errors are NOT caught — they propagate unchanged."""
    err = ValueError("unrelated programming error")
    model = _RaisingModel(errors=[err], turns=[])
    agent = Agent(
        config=_minimal_config(),
        model=model,
        storage=_storage(tmp_path),
        auto_compact_threshold=0,
    )

    compact_calls: list[str] = []

    async def _spy(self: Agent, *, source: str = "manual") -> CompactResult:
        compact_calls.append(source)
        return CompactResult(
            before_tokens=0, after_tokens=0,
            source=source,  # type: ignore[arg-type]
        )

    with (
        patch.object(Agent, "compact", _spy),
        pytest.raises(ValueError, match="unrelated"),
    ):
        async for _ in agent.astream("hi"):
            pass

    assert compact_calls == []
    await agent.aclose()


@pytest.mark.asyncio
async def test_reactive_compact_journal_event(tmp_path: Path) -> None:
    """``reactive_compact_triggered`` event is emitted with the error string."""
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)
    try:
        err = RuntimeError("context_length_exceeded — prompt too long")
        model = _RaisingModel(
            errors=[err, None, None],
            turns=[
                FakeTurn(AIMessage(content="SUMMARY")),
                FakeTurn(AIMessage(content="recovered")),
            ],
        )
        agent = Agent(
            config=_minimal_config(),
            model=model,
            storage=_storage(tmp_path),
            auto_compact_threshold=0,
        )
        from langchain_core.messages import HumanMessage
        h: list[Any] = []
        for i in range(10):
            h.append(HumanMessage(content=f"u-{i}"))
            h.append(AIMessage(content=f"a-{i}"))
        agent._storage.save(agent.session_id, h)

        async for _ in agent.astream("hi"):
            pass

        events = [
            json.loads(line)
            for line in log_path.read_text().strip().split("\n")
            if line
        ]
        triggered = [
            e for e in events if e["event"] == "reactive_compact_triggered"
        ]
        assert len(triggered) == 1
        ev = triggered[0]
        assert "context_length_exceeded" in ev["error"]
        assert ev["session"] == "default"
        await agent.aclose()
    finally:
        journal.reset()
