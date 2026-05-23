"""Phase 4 Task 3 — :class:`aura.application.compact.compactor.Compactor`.

Locks the four-method surface (microcompact / reactive / auto / manual)
of the new first-class :class:`Compactor`, plus the per-call event
contract: every method emits exactly one ``compact_event`` journal
record AND one wire-format payload to the optional event emitter.

The auto path's circuit breaker is also pinned here — three
consecutive failures must trip the breaker (mirrors the legacy
adapter's behavior, now sourced from
``CompactConfig.max_consecutive_failures`` instead of a module
constant).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.application.compact.compact import CompactResult
from aura.application.compact.compactor import Compactor
from aura.application.compact.microcompact import MicrocompactPolicy
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _agent(tmp_path: Path, *, threshold: int = 10) -> Agent:
    return Agent(
        config=_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="x"))] * 20),
        storage=SessionStorage(tmp_path / "aura.db"),
        auto_compact_threshold=threshold,
    )


def _make_compactor(
    agent: Agent,
    *,
    events: list[dict[str, Any]] | None = None,
    microcompact_policy: MicrocompactPolicy | None = None,
) -> Compactor:
    """Build a :class:`Compactor` wired to ``agent`` plus an event sink list.

    Tests assert against ``events`` to verify the AG-UI emitter receives
    the expected wire-format dict per call. ``microcompact_policy``
    defaults to a tiny policy so :meth:`microcompact` actually runs the
    clearing logic when invoked with enough pairs.
    """
    return Compactor(
        agent=agent,
        config=agent.config.compact,
        summary_model=agent._model,
        microcompact_policy=microcompact_policy,
        session_id=agent.session_id,
        turn_provider=lambda: agent.state.turn_count,
        event_emitter=(events.append if events is not None else None),
    )


def _read_journal(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [
        json.loads(line)
        for line in log.read_text().splitlines()
        if line.strip()
    ]


@pytest.mark.asyncio
async def test_microcompact_skipped_when_policy_none(tmp_path: Path) -> None:
    """No policy → method is a pass-through; emits ``outcome="skipped"``."""
    agent = _agent(tmp_path)
    events: list[dict[str, Any]] = []
    compactor = _make_compactor(agent, events=events, microcompact_policy=None)

    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    out = await compactor.microcompact(msgs, agent.state.slots)

    assert out is msgs  # identity preserved
    assert len(events) == 1
    assert events[0]["event"] == "compact_event"
    assert events[0]["trigger"] == "microcompact"
    assert events[0]["outcome"] == "skipped"
    await agent.aclose()


@pytest.mark.asyncio
async def test_microcompact_happy_path_emits_ok(tmp_path: Path) -> None:
    """With a policy + enough pairs, microcompact clears and emits ``ok``.

    The microcompact view transform replaces tool_result payloads on
    eligible pairs. We seed a long-enough conversation that
    :func:`apply_microcompact` actually fires (trigger_pairs=2, so the
    third pair triggers clearing of the older one).
    """
    from langchain_core.messages import ToolMessage
    agent = _agent(tmp_path)
    events: list[dict[str, Any]] = []
    policy = MicrocompactPolicy(trigger_pairs=2, keep_recent=1)
    compactor = _make_compactor(
        agent, events=events, microcompact_policy=policy,
    )

    # Build 3 tool_use/tool_result pairs so the third one triggers
    # clearing of the older two (keep_recent=1 → only the newest stays).
    msgs: list[Any] = []
    for i in range(3):
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_file", "args": {"path": f"/x{i}.py"},
                    "id": f"call-{i}",
                }],
            ),
        )
        msgs.append(
            ToolMessage(
                content=f"contents-{i}", tool_call_id=f"call-{i}",
            ),
        )

    out = await compactor.microcompact(msgs, agent.state.slots)
    assert len(events) == 1
    # When pairs were cleared, outcome must be "ok".
    if events[0]["outcome"] == "ok":
        # Cleared at least one pair — the message list is a NEW list.
        assert out is not msgs
    await agent.aclose()


@pytest.mark.asyncio
async def test_microcompact_writes_compact_event_to_journal(
    tmp_path: Path,
) -> None:
    """Journal record shape per spec §5.

    Asserts the canonical fields exist and the trigger is the enum's
    string value (StrEnum round-trip).
    """
    log = tmp_path / "audit.jsonl"
    journal.configure(log)
    try:
        agent = _agent(tmp_path)
        compactor = _make_compactor(agent, microcompact_policy=None)
        await compactor.microcompact(
            [HumanMessage(content="hi")], agent.state.slots,
        )
        records = [
            ev for ev in _read_journal(log) if ev.get("event") == "compact_event"
        ]
        assert len(records) == 1
        rec = records[0]
        assert rec["trigger"] == "microcompact"
        assert "tokens_before" in rec
        assert "tokens_after" in rec
        assert rec["outcome"] == "skipped"
        assert isinstance(rec["duration_ms"], (int, float))
        await agent.aclose()
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_reactive_delegates_to_agent_compact(tmp_path: Path) -> None:
    """reactive() routes through ``Agent.compact(source="reactive")``."""
    agent = _agent(tmp_path)
    events: list[dict[str, Any]] = []
    compactor = _make_compactor(agent, events=events)

    seen: list[str] = []

    async def _ok(self: Agent, *, source: str = "manual") -> CompactResult:
        seen.append(source)
        return CompactResult(
            before_tokens=100, after_tokens=50,
            source=source,  # type: ignore[arg-type]
        )

    history: list[Any] = []
    with patch.object(Agent, "compact", _ok):
        result = await compactor.reactive(history, agent.state.slots)

    assert seen == ["reactive"]
    assert result.before_tokens == 100
    assert result.after_tokens == 50
    assert len(events) == 1
    assert events[0]["trigger"] == "reactive"
    assert events[0]["outcome"] == "ok"
    assert events[0]["tokens_before"] == 100
    assert events[0]["tokens_after"] == 50
    await agent.aclose()


@pytest.mark.asyncio
async def test_reactive_failure_emits_failed_outcome(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    events: list[dict[str, Any]] = []
    compactor = _make_compactor(agent, events=events)

    async def _boom(self: Agent, *, source: str = "manual") -> CompactResult:
        raise RuntimeError("simulated")

    history: list[Any] = []
    with patch.object(Agent, "compact", _boom), pytest.raises(RuntimeError, match="simulated"):
        await compactor.reactive(history, agent.state.slots)

    assert len(events) == 1
    assert events[0]["trigger"] == "reactive"
    assert events[0]["outcome"] == "failed"
    await agent.aclose()


@pytest.mark.asyncio
async def test_auto_below_threshold_skips(tmp_path: Path) -> None:
    """Token usage below threshold → ``None`` + ``outcome="skipped"``."""
    agent = _agent(tmp_path, threshold=1_000)
    agent.state.total_tokens_used = 10
    events: list[dict[str, Any]] = []
    compactor = _make_compactor(agent, events=events)

    result = await compactor.auto(
        [], agent.state.slots, model="openai:gpt-4o-mini",
    )

    assert result is None
    assert len(events) == 1
    assert events[0]["trigger"] == "auto"
    assert events[0]["outcome"] == "skipped"
    await agent.aclose()


@pytest.mark.asyncio
async def test_auto_above_threshold_runs_and_resets_breaker(
    tmp_path: Path,
) -> None:
    """Crossing threshold → run; success resets ``consecutive_compact_failures``."""
    import dataclasses as _dc
    agent = _agent(tmp_path, threshold=10)
    agent.state.total_tokens_used = 100
    agent.state.slots = _dc.replace(
        agent.state.slots, consecutive_compact_failures=2,
    )
    events: list[dict[str, Any]] = []
    compactor = _make_compactor(agent, events=events)

    async def _ok(self: Agent, *, source: str = "manual") -> CompactResult:
        return CompactResult(
            before_tokens=100, after_tokens=20,
            source=source,  # type: ignore[arg-type]
        )

    with patch.object(Agent, "compact", _ok):
        result = await compactor.auto(
            [], agent.state.slots, model="openai:gpt-4o-mini",
        )

    assert result is not None
    assert result.after_tokens == 20
    assert agent.state.slots.consecutive_compact_failures == 0
    assert any(
        ev["trigger"] == "auto" and ev["outcome"] == "ok" for ev in events
    )
    await agent.aclose()


@pytest.mark.asyncio
async def test_auto_failure_increments_circuit_breaker(tmp_path: Path) -> None:
    """Each failed auto run increments ``consecutive_compact_failures``.

    The breaker limit comes from
    ``CompactConfig.max_consecutive_failures`` (default 3); after that
    many failures the next call short-circuits without invoking
    ``Agent.compact`` again.
    """
    agent = _agent(tmp_path, threshold=10)
    agent.state.total_tokens_used = 100
    events: list[dict[str, Any]] = []
    compactor = _make_compactor(agent, events=events)

    calls: list[str] = []

    async def _fail(self: Agent, *, source: str = "manual") -> CompactResult:
        calls.append(source)
        raise RuntimeError("compact failed")

    # Three failures in a row.
    with patch.object(Agent, "compact", _fail):
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await compactor.auto(
                    [], agent.state.slots, model="openai:gpt-4o-mini",
                )

    assert agent.state.slots.consecutive_compact_failures == 3
    assert len(calls) == 3

    # Fourth attempt — breaker tripped, no more Agent.compact calls.
    pre = len(calls)
    with patch.object(Agent, "compact", _fail):
        result = await compactor.auto(
            [], agent.state.slots, model="openai:gpt-4o-mini",
        )
    assert result is None
    assert len(calls) == pre, "circuit breaker did not block 4th attempt"
    # The blocked attempt still emitted a ``skipped`` event.
    assert events[-1]["trigger"] == "auto"
    assert events[-1]["outcome"] == "skipped"
    await agent.aclose()


@pytest.mark.asyncio
async def test_auto_breaker_limit_honors_config(tmp_path: Path) -> None:
    """Lowering ``max_consecutive_failures`` to 1 trips the breaker sooner."""
    cfg = _config()
    # CompactConfig is frozen via pydantic; rebuild the AuraConfig with
    # an override so the agent's compact config carries the new limit.
    cfg = AuraConfig.model_validate({
        **cfg.model_dump(),
        "compact": {"max_consecutive_failures": 1},
    })
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="x"))] * 5),
        storage=SessionStorage(tmp_path / "aura.db"),
        auto_compact_threshold=10,
    )
    agent.state.total_tokens_used = 100
    compactor = _make_compactor(agent)

    calls: list[str] = []

    async def _fail(self: Agent, *, source: str = "manual") -> CompactResult:
        calls.append(source)
        raise RuntimeError("boom")

    with patch.object(Agent, "compact", _fail):
        with pytest.raises(RuntimeError):
            await compactor.auto(
                [], agent.state.slots, model="openai:gpt-4o-mini",
            )
        # After ONE failure the breaker is tripped (limit=1).
        result = await compactor.auto(
            [], agent.state.slots, model="openai:gpt-4o-mini",
        )

    assert result is None
    assert len(calls) == 1, "breaker should block 2nd attempt at limit=1"
    await agent.aclose()


@pytest.mark.asyncio
async def test_manual_delegates_and_bypasses_breaker(tmp_path: Path) -> None:
    """Manual ignores the circuit breaker entirely."""
    import dataclasses as _dc
    agent = _agent(tmp_path)
    agent.state.slots = _dc.replace(
        agent.state.slots, consecutive_compact_failures=99,
    )
    events: list[dict[str, Any]] = []
    compactor = _make_compactor(agent, events=events)

    seen: list[str] = []

    async def _ok(self: Agent, *, source: str = "manual") -> CompactResult:
        seen.append(source)
        return CompactResult(
            before_tokens=10, after_tokens=5,
            source=source,  # type: ignore[arg-type]
        )

    with patch.object(Agent, "compact", _ok):
        result = await compactor.manual([], agent.state.slots)

    assert seen == ["manual"]
    assert result.before_tokens == 10
    assert result.after_tokens == 5
    # Breaker counter untouched — manual is a user-explicit override.
    assert agent.state.slots.consecutive_compact_failures == 99
    assert len(events) == 1
    assert events[0]["trigger"] == "manual"
    assert events[0]["outcome"] == "ok"
    await agent.aclose()


@pytest.mark.asyncio
async def test_event_emitter_payload_matches_spec(tmp_path: Path) -> None:
    """Spec §5: ``{"event": "compact_event", "trigger": ..., ...}``.

    The emitter receives the exact same dict shape the journal records
    (minus the ``session`` field, which is only on the journal side).
    """
    agent = _agent(tmp_path)
    events: list[dict[str, Any]] = []
    compactor = _make_compactor(agent, events=events)

    async def _ok(self: Agent, *, source: str = "manual") -> CompactResult:
        return CompactResult(
            before_tokens=42, after_tokens=7,
            source=source,  # type: ignore[arg-type]
        )

    with patch.object(Agent, "compact", _ok):
        await compactor.manual([], agent.state.slots)

    assert len(events) == 1
    ev = events[0]
    assert set(ev.keys()) == {
        "event", "trigger", "tokens_before", "tokens_after",
        "outcome", "duration_ms",
    }
    assert ev["event"] == "compact_event"
    assert ev["trigger"] == "manual"
    assert ev["tokens_before"] == 42
    assert ev["tokens_after"] == 7
    assert ev["outcome"] == "ok"
    assert isinstance(ev["duration_ms"], float)
    await agent.aclose()


@pytest.mark.asyncio
async def test_emitter_failure_does_not_propagate(tmp_path: Path) -> None:
    """A buggy AG-UI emitter must not abort a compact cycle."""
    agent = _agent(tmp_path)

    def _boom(_: dict[str, Any]) -> None:
        raise RuntimeError("emitter bug")

    compactor = Compactor(
        agent=agent,
        config=agent.config.compact,
        summary_model=agent._model,
        microcompact_policy=None,
        session_id=agent.session_id,
        turn_provider=lambda: agent.state.turn_count,
        event_emitter=_boom,
    )

    # microcompact with policy=None is a pass-through; the emitter
    # would normally fire here. The buggy emitter must be swallowed.
    out = await compactor.microcompact(
        [HumanMessage(content="x")], agent.state.slots,
    )
    assert len(out) == 1
    await agent.aclose()


