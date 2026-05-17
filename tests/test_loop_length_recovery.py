"""F-01-005 — partial-response (max_output_tokens) recovery.

When ``ai.response_metadata['finish_reason']`` is ``'length'`` (OpenAI) or
``ai.stop_reason`` / ``response_metadata['stop_reason']`` is ``'max_tokens'``
(Anthropic), :meth:`AgentLoop._invoke_model` must append a meta
HumanMessage and re-invoke up to ``_MAX_LENGTH_RETRY`` times.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.capabilities.tools.registry import ToolRegistry
from aura.core.hooks import HookChain
from aura.core.loop import _MAX_LENGTH_RETRY, AgentLoop
from aura.core.persistence import journal
from aura.schemas.events import AgentEvent, Final, ToolCallStarted
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel, FakeTurn, make_minimal_context


class _EchoParams(BaseModel):
    msg: str


def _echo(msg: str) -> dict[str, Any]:
    return {"echoed": msg}


_echo_tool: BaseTool = build_tool(
    name="echo",
    description="echoes input",
    args_schema=_EchoParams,
    func=_echo,
    is_read_only=True,
    is_concurrency_safe=True,
)


def _truncated(content: str) -> AIMessage:
    """OpenAI-style finish_reason='length' AIMessage."""
    return AIMessage(
        content=content,
        response_metadata={"finish_reason": "length"},
    )


def _truncated_anthropic(content: str) -> AIMessage:
    """Anthropic-style stop_reason='max_tokens' AIMessage."""
    return AIMessage(
        content=content,
        response_metadata={"stop_reason": "max_tokens"},
    )


def _final(content: str) -> AIMessage:
    return AIMessage(content=content, response_metadata={"finish_reason": "stop"})


@pytest.mark.asyncio
async def test_length_truncated_tool_calls_are_not_dispatched() -> None:
    """Tool calls on a length-truncated response are discarded before dispatch."""
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(
            content="partial tool decision",
            response_metadata={"finish_reason": "length"},
            tool_calls=[{"name": "echo", "args": {"msg": "hi"}, "id": "tc_1"}],
        )),
        FakeTurn(message=_final("recovered final")),
    ])
    history: list[BaseMessage] = [HumanMessage(content="long task")]
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_echo_tool]),
        context=make_minimal_context(), hooks=HookChain(),
    )

    events: list[AgentEvent] = []
    async for ev in loop.run_turn(history=history):
        events.append(ev)

    assert model.ainvoke_calls == 2
    assert not any(isinstance(ev, ToolCallStarted) for ev in events)
    assert not any(isinstance(msg, ToolMessage) for msg in history)

    ai_messages = [m for m in history if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1
    assert ai_messages[0].content == "recovered final"


@pytest.mark.asyncio
async def test_length_finish_reason_triggers_resume_one_round(
    tmp_path: Path,
) -> None:
    """First reply truncated, second reply complete → exactly one re-invoke."""
    journal.configure(tmp_path / "j.jsonl")
    try:
        model = FakeChatModel(turns=[
            FakeTurn(message=_truncated("part one ...")),
            FakeTurn(message=_final("part two done")),
        ])
        history: list[BaseMessage] = [HumanMessage(content="long task")]
        loop = AgentLoop(
            model=model, registry=ToolRegistry(()),
            context=make_minimal_context(), hooks=HookChain(),
        )
        events: list[AgentEvent] = []
        async for ev in loop.run_turn(history=history):
            events.append(ev)

        # Exactly two ainvoke calls: one truncated + one resume.
        assert model.ainvoke_calls == 2

        # Only the FINAL AIMessage is on history (not the truncated one).
        ai_messages = [m for m in history if isinstance(m, AIMessage)]
        assert len(ai_messages) == 1
        assert ai_messages[0].content == "part two done"

        # The resume HumanMessage was NOT appended to history — it's a
        # local "view" injection only.
        human_messages = [m for m in history if isinstance(m, HumanMessage)]
        assert len(human_messages) == 1
        assert human_messages[0].content == "long task"

        # Journal records exactly one length_recovery event.
        lines = (tmp_path / "j.jsonl").read_text().splitlines()
        recoveries = [
            json.loads(line) for line in lines if '"length_recovery"' in line
        ]
        assert len(recoveries) == 1
        assert recoveries[0]["attempt"] == 1
        assert recoveries[0]["max_attempts"] == _MAX_LENGTH_RETRY
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_anthropic_max_tokens_stop_reason_triggers_resume() -> None:
    """Anthropic's ``stop_reason='max_tokens'`` shape also triggers recovery."""
    model = FakeChatModel(turns=[
        FakeTurn(message=_truncated_anthropic("partial")),
        FakeTurn(message=_final("complete")),
    ])
    history: list[BaseMessage] = [HumanMessage(content="hi")]
    loop = AgentLoop(
        model=model, registry=ToolRegistry(()),
        context=make_minimal_context(), hooks=HookChain(),
    )
    async for _ in loop.run_turn(history=history):
        pass

    assert model.ainvoke_calls == 2


@pytest.mark.asyncio
async def test_length_recovery_caps_at_max_retry(tmp_path: Path) -> None:
    """Always-truncated provider stops after _MAX_LENGTH_RETRY extra calls."""
    journal.configure(tmp_path / "j.jsonl")
    try:
        # Need 1 (initial) + _MAX_LENGTH_RETRY scripted truncated turns =
        # 4 truncated total. After the budget is exhausted the loop accepts
        # whatever the last AIMessage is and carries on.
        turns = [
            FakeTurn(message=_truncated(f"chunk {i}"))
            for i in range(_MAX_LENGTH_RETRY + 1)
        ]
        model = FakeChatModel(turns=turns)
        history: list[BaseMessage] = [HumanMessage(content="never ends")]
        loop = AgentLoop(
            model=model, registry=ToolRegistry(()),
            context=make_minimal_context(), hooks=HookChain(),
        )
        async for _ in loop.run_turn(history=history):
            pass

        # Original + 3 retries = 4 ainvoke calls.
        assert model.ainvoke_calls == _MAX_LENGTH_RETRY + 1

        lines = (tmp_path / "j.jsonl").read_text().splitlines()
        recoveries = [
            json.loads(line) for line in lines if '"length_recovery"' in line
        ]
        assert len(recoveries) == _MAX_LENGTH_RETRY
        assert [r["attempt"] for r in recoveries] == list(
            range(1, _MAX_LENGTH_RETRY + 1),
        )
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_length_recovery_exhaustion_does_not_dispatch_truncated_tool_calls(
    tmp_path: Path,
) -> None:
    """Exhaustion must not dispatch the truncated turn's tool_calls AND must
    surface ``Final.reason == "length_recovery_exhausted"`` carrying the
    partial assistant text (Phase 1 Task 12)."""
    journal.configure(tmp_path / "j.jsonl")
    try:
        turns = [
            FakeTurn(message=AIMessage(
                content=f"truncated {i}",
                response_metadata={"finish_reason": "length"},
                tool_calls=[{
                    "name": "echo",
                    "args": {"msg": f"unsafe-{i}"},
                    "id": f"tc_{i}",
                }],
            ))
            for i in range(_MAX_LENGTH_RETRY + 1)
        ]
        model = FakeChatModel(turns=turns)
        history: list[BaseMessage] = [HumanMessage(content="never ends")]
        loop = AgentLoop(
            model=model, registry=ToolRegistry([_echo_tool]),
            context=make_minimal_context(), hooks=HookChain(),
        )

        events: list[AgentEvent] = []
        async for ev in loop.run_turn(history=history):
            events.append(ev)

        assert model.ainvoke_calls == _MAX_LENGTH_RETRY + 1
        assert not any(isinstance(ev, ToolCallStarted) for ev in events)
        assert not any(isinstance(msg, ToolMessage) for msg in history)
        finals = [ev for ev in events if isinstance(ev, Final)]
        assert finals
        assert finals[-1].reason == "length_recovery_exhausted"
        # Partial content from the LAST truncated turn is surfaced verbatim
        # — no synthetic "recovery exhausted" stub overwrites it.
        assert finals[-1].message == f"truncated {_MAX_LENGTH_RETRY}"
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_length_recovery_exhaustion_surfaces_partial_content_no_tool_calls(
    tmp_path: Path,
) -> None:
    """Pure-text exhaustion path: Final.message carries the partial body and
    Final.reason is the new sentinel."""
    journal.configure(tmp_path / "j.jsonl")
    try:
        turns = [
            FakeTurn(message=_truncated(f"chunk {i} of partial answer ..."))
            for i in range(_MAX_LENGTH_RETRY + 1)
        ]
        model = FakeChatModel(turns=turns)
        history: list[BaseMessage] = [HumanMessage(content="really long")]
        loop = AgentLoop(
            model=model, registry=ToolRegistry(()),
            context=make_minimal_context(), hooks=HookChain(),
        )
        events: list[AgentEvent] = []
        async for ev in loop.run_turn(history=history):
            events.append(ev)

        finals = [ev for ev in events if isinstance(ev, Final)]
        assert len(finals) == 1
        assert finals[0].reason == "length_recovery_exhausted"
        assert finals[0].message == f"chunk {_MAX_LENGTH_RETRY} of partial answer ..."
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_no_length_recovery_when_finish_reason_is_stop() -> None:
    """Normal finish_reason=stop response → zero re-invokes."""
    model = FakeChatModel(turns=[FakeTurn(message=_final("hello"))])
    history: list[BaseMessage] = [HumanMessage(content="hi")]
    loop = AgentLoop(
        model=model, registry=ToolRegistry(()),
        context=make_minimal_context(), hooks=HookChain(),
    )
    async for _ in loop.run_turn(history=history):
        pass

    assert model.ainvoke_calls == 1


@pytest.mark.asyncio
async def test_no_length_recovery_when_metadata_missing() -> None:
    """A bare AIMessage (no response_metadata at all) is treated as final."""
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="bare"))])
    history: list[BaseMessage] = [HumanMessage(content="hi")]
    loop = AgentLoop(
        model=model, registry=ToolRegistry(()),
        context=make_minimal_context(), hooks=HookChain(),
    )
    async for _ in loop.run_turn(history=history):
        pass

    assert model.ainvoke_calls == 1
