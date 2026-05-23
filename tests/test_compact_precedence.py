"""Phase 4 §6 — pin the precedence rules that govern how the four
compaction triggers (microcompact / reactive / auto / length-recovery)
interact within a single turn AND across turns.

These tests are *contracts*, not implementation tests:

1. ``microcompact`` runs FIRST every turn — view-only, before
   ``model.ainvoke``; storage stays raw.
2. ``reactive`` triggers ONLY on ``PromptTooLong`` mid-turn — a normal
   turn never invokes it.
3. ``auto`` triggers ONLY after a successful turn finishes — it never
   fires mid-turn.
4. ``length-recovery`` and ``reactive`` are disjoint — a length-truncated
   ``AIMessage`` never invokes ``reactive``; a ``PromptTooLong`` exception
   never triggers length-recovery.
5. The auto-compact circuit breaker disables the trigger after
   ``max_consecutive_failures`` consecutive failures and resets on any
   successful run.

Each test patches a tight surface (``Compactor.microcompact`` /
``Agent.compact`` / model error stream) and asserts on call ordering,
counts, or raised types — no full compaction round-trip needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aura.application.compact import CompactResult
from aura.application.compact.compactor import Compactor
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.infrastructure.persistence.storage import SessionStorage
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
    tmp_path: Path,
    *,
    threshold: int = 0,
    turns: list[FakeTurn] | None = None,
    model: FakeChatModel | None = None,
) -> Agent:
    return Agent(
        config=_minimal_config(),
        model=model or FakeChatModel(
            turns=turns or [FakeTurn(AIMessage(content="ok"))],
        ),
        storage=_storage(tmp_path),
        auto_compact_threshold=threshold,
    )


class _RaisingModel(FakeChatModel):
    """FakeChatModel whose i-th call raises ``errors[i]`` (None = scripted turn)."""

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
async def test_microcompact_runs_before_model_ainvoke(tmp_path: Path) -> None:
    """Spec §6 #1 — ``Compactor.microcompact`` must complete before the
    very first ``model.ainvoke`` of the turn. Storage is the canonical
    proof microcompact didn't mutate persisted history (it's view-only).
    """
    agent = _agent(tmp_path)

    # Pre-seed two compactable read_file pairs so microcompact has work
    # to do — the storage-invariant assertion below depends on the
    # ToolMessage content surviving regardless.
    raw_history: list[BaseMessage] = []
    for i in range(2):
        raw_history.append(HumanMessage(content=f"u-{i}"))
        raw_history.append(AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": f"/f{i}"},
                         "id": f"tc-{i}"}],
        ))
        raw_history.append(ToolMessage(
            content=f"payload-{i}",
            tool_call_id=f"tc-{i}",
            name="read_file",
            status="success",
        ))
    agent.storage.save(agent.session_id, raw_history)

    call_order: list[str] = []

    orig_micro = Compactor.microcompact

    async def _spy_micro(
        self: Compactor,
        messages: list[BaseMessage],
        slots: Any,
        trigger: Any = None,
    ) -> list[BaseMessage]:
        call_order.append("microcompact")
        if trigger is None:
            return await orig_micro(self, messages, slots)
        return await orig_micro(self, messages, slots, trigger)

    orig_agenerate = FakeChatModel._agenerate

    async def _spy_agenerate(
        self: FakeChatModel,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        call_order.append("ainvoke")
        return await orig_agenerate(
            self, messages, stop, run_manager, **kwargs,
        )

    with (
        patch.object(Compactor, "microcompact", _spy_micro),
        patch.object(FakeChatModel, "_agenerate", _spy_agenerate),
    ):
        async for _ in agent.astream("next"):
            pass

    # microcompact precedes the first ainvoke. Both events must be
    # present and microcompact's index must be smaller.
    assert "microcompact" in call_order
    assert "ainvoke" in call_order
    assert call_order.index("microcompact") < call_order.index("ainvoke"), (
        f"microcompact must run before ainvoke; got order={call_order!r}"
    )

    # Storage stayed raw — view-only contract. Reload to confirm both
    # ToolMessage payloads survived verbatim.
    reloaded = agent.storage.load(agent.session_id)
    tool_msgs = [m for m in reloaded if isinstance(m, ToolMessage)]
    assert [tm.content for tm in tool_msgs] == ["payload-0", "payload-1"]
    await agent.aclose()


@pytest.mark.asyncio
async def test_reactive_only_fires_on_prompt_too_long(tmp_path: Path) -> None:
    """Spec §6 #2 — a normal turn (no PTL exception) must NOT invoke
    ``Agent.compact(source="reactive")``. A PTL turn must.
    """
    agent_ok = _agent(tmp_path / "ok")
    reactive_calls_ok: list[str] = []
    auto_calls_ok: list[str] = []

    async def _spy_ok(self: Agent, *, source: str = "manual") -> CompactResult:
        if source == "reactive":
            reactive_calls_ok.append(source)
        elif source == "auto":
            auto_calls_ok.append(source)
        return CompactResult(
            before_tokens=0, after_tokens=0,
            source=source,  # type: ignore[arg-type]
        )

    with patch.object(Agent, "compact", _spy_ok):
        async for _ in agent_ok.astream("hi"):
            pass

    assert reactive_calls_ok == [], (
        "reactive must not fire on a normal turn"
    )
    await agent_ok.aclose()

    err = RuntimeError("400 — context_length_exceeded for model gpt-4o")
    model = _RaisingModel(
        errors=[err, None, None],  # PTL → summary turn → retry
        turns=[
            FakeTurn(AIMessage(content="SUMMARY")),
            FakeTurn(AIMessage(content="recovered")),
        ],
    )
    agent_ptl = _agent(tmp_path / "ptl", model=model)

    # Seed enough history that compact's tail-keep branch produces a
    # non-trivial summary turn (otherwise run_compact short-circuits).
    seeded: list[BaseMessage] = []
    for i in range(10):
        seeded.append(HumanMessage(content=f"u-{i}"))
        seeded.append(AIMessage(content=f"a-{i}"))
    agent_ptl._storage.save(agent_ptl.session_id, seeded)

    reactive_calls_ptl: list[str] = []
    orig_compact = Agent.compact

    async def _spy_ptl(self: Agent, *, source: str = "manual") -> CompactResult:
        reactive_calls_ptl.append(source)
        return await orig_compact(self, source=source)  # type: ignore[arg-type]

    with patch.object(Agent, "compact", _spy_ptl):
        async for _ in agent_ptl.astream("hi"):
            pass

    assert reactive_calls_ptl == ["reactive"], (
        f"PTL turn must invoke reactive exactly once; got {reactive_calls_ptl!r}"
    )
    await agent_ptl.aclose()


@pytest.mark.asyncio
async def test_auto_fires_post_turn_only(tmp_path: Path) -> None:
    """Spec §6 #3 — ``Compactor.auto`` is invoked AFTER ``astream_end``
    (post-turn). The test pins this by checking that no auto call lands
    while the turn's ``Final`` event is still mid-stream — i.e. the
    auto call is observable strictly after iteration finishes.
    """
    agent = _agent(tmp_path, threshold=50)
    agent.state.total_tokens_used = 100  # threshold crossed

    auto_call_indices: list[int] = []
    final_event_indices: list[int] = []
    event_count = 0

    async def _spy_auto(
        self: Agent, *, source: str = "manual",
    ) -> CompactResult:
        # Record the event-stream position at which the auto call lands.
        auto_call_indices.append(event_count)
        return CompactResult(
            before_tokens=0, after_tokens=0,
            source=source,  # type: ignore[arg-type]
        )

    from aura.schemas.events import Final

    with patch.object(Agent, "compact", _spy_auto):
        async for ev in agent.astream("hi"):
            event_count += 1
            if isinstance(ev, Final):
                final_event_indices.append(event_count)

    # auto fired exactly once.
    assert len(auto_call_indices) == 1
    # auto fired AFTER the Final event was yielded — astream's auto
    # branch lives after the per-turn save + astream_end journal write.
    assert final_event_indices, "Final must be yielded before auto runs"
    assert auto_call_indices[0] >= final_event_indices[0], (
        "auto-compact must not fire mid-turn — it lands post-Final"
    )
    await agent.aclose()


@pytest.mark.asyncio
async def test_length_recovery_does_not_invoke_reactive(tmp_path: Path) -> None:
    """Spec §6 #4 — a length-truncated reply (``finish_reason='length'``)
    is handled by the resume-prompt path INSIDE ``_invoke_model``. It
    must NOT invoke ``reactive`` (which is reserved for ``PromptTooLong``).
    """
    truncated = AIMessage(
        content="partial",
        response_metadata={"finish_reason": "length"},
    )
    final = AIMessage(
        content="complete",
        response_metadata={"finish_reason": "stop"},
    )
    model = FakeChatModel(turns=[FakeTurn(truncated), FakeTurn(final)])
    agent = _agent(tmp_path, model=model)

    reactive_calls: list[str] = []

    async def _spy(self: Agent, *, source: str = "manual") -> CompactResult:
        reactive_calls.append(source)
        return CompactResult(
            before_tokens=0, after_tokens=0,
            source=source,  # type: ignore[arg-type]
        )

    with patch.object(Agent, "compact", _spy):
        async for _ in agent.astream("hi"):
            pass

    # The model was invoked twice (initial truncated + resume).
    assert model.ainvoke_calls == 2
    # No reactive call — length-recovery handled it locally.
    assert "reactive" not in reactive_calls, (
        f"length-recovery must not invoke reactive; got {reactive_calls!r}"
    )
    await agent.aclose()


@pytest.mark.asyncio
async def test_circuit_breaker_disables_after_three_failures_and_resets_on_success(
    tmp_path: Path,
) -> None:
    """Spec §6 #5 — ``max_consecutive_failures=3`` is the configured
    threshold. After 3 consecutive failures the breaker disables the
    next auto attempt entirely (the spy is NOT re-entered for call #4).
    Any successful run resets ``consecutive_compact_failures`` to 0.
    """
    import dataclasses as _dc

    # 20 scripted turns covers 4 fail attempts + 1 success run with headroom.
    agent = _agent(
        tmp_path,
        threshold=10,
        turns=[FakeTurn(AIMessage(content="ok"))] * 20,
    )
    agent.state.total_tokens_used = 100  # always crosses threshold

    calls: list[str] = []

    async def _fail(self: Agent, *, source: str = "manual") -> CompactResult:
        calls.append(source)
        raise RuntimeError("simulated compact failure")

    # Three consecutive failing astream calls trip the breaker.
    with patch.object(Agent, "compact", _fail):
        for _ in range(3):
            with pytest.raises(RuntimeError):
                async for _ in agent.astream("hi"):
                    pass

    assert len(calls) == 3, (
        f"first three turns must each invoke compact once; got {calls!r}"
    )
    assert agent.state.slots.consecutive_compact_failures == 3

    # 4th turn — breaker is tripped → spy MUST NOT be re-entered.
    with patch.object(Agent, "compact", _fail):
        async for _ in agent.astream("hi"):
            pass

    assert len(calls) == 3, (
        "breaker did not block the 4th auto-compact attempt"
    )

    # Reset path — drop the counter back to a sub-threshold value so the
    # breaker re-arms, then run a successful turn. The success path
    # (Compactor.auto) zeros the counter on its way out.
    agent.state.slots = _dc.replace(
        agent.state.slots, consecutive_compact_failures=2,
    )

    async def _ok(self: Agent, *, source: str = "manual") -> CompactResult:
        calls.append(source)
        return CompactResult(
            before_tokens=0, after_tokens=0,
            source=source,  # type: ignore[arg-type]
        )

    with patch.object(Agent, "compact", _ok):
        async for _ in agent.astream("hi"):
            pass

    assert agent.state.slots.consecutive_compact_failures == 0, (
        "successful auto run must reset consecutive_compact_failures"
    )
    await agent.aclose()
