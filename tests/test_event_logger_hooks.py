"""Tests for aura.core.hooks.logging — make_event_logger_hooks + wrap_with_event_logger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from aura.core import journal as journal_module
from aura.core.hooks import HookChain
from aura.core.hooks.logging import make_event_logger_hooks, wrap_with_event_logger
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool


@pytest.fixture(autouse=True)
def _journal_to_tmp(tmp_path: Path) -> Any:
    journal_module.reset()
    journal_module.configure(tmp_path / "events.jsonl")
    yield tmp_path / "events.jsonl"
    journal_module.reset()


def _events(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text().splitlines()]


class _FakeParams(BaseModel):
    x: int = 0


def _fake_tool(*, destructive: bool = False) -> Any:
    def _run(x: int = 0) -> dict[str, int]:
        return {"got": x}
    return build_tool(
        name="fake",
        description="fake tool",
        args_schema=_FakeParams,
        func=_run,
        is_destructive=destructive,
    )


@pytest.mark.asyncio
async def test_pre_model_records_turn_and_history_len(_journal_to_tmp: Path) -> None:
    chain = make_event_logger_hooks()
    state = LoopState(turn_count=3)
    history = [HumanMessage(content="a"), AIMessage(content="b")]

    await chain.run_pre_model(history=history, state=state)

    [event] = _events(_journal_to_tmp)
    assert event["event"] == "pre_model"
    assert event["turn"] == 3
    assert event["history_len"] == 2


@pytest.mark.asyncio
async def test_post_model_records_content_preview_and_usage(
    _journal_to_tmp: Path,
) -> None:
    chain = make_event_logger_hooks()
    state = LoopState(turn_count=1, total_tokens_used=150)
    ai = AIMessage(
        content="hello",
        usage_metadata={"input_tokens": 10, "output_tokens": 32, "total_tokens": 42},
    )

    await chain.run_post_model(ai_message=ai, history=[], state=state)

    [event] = _events(_journal_to_tmp)
    assert event["event"] == "post_model"
    assert event["content_chars"] == 5
    assert event["content_preview"] == "hello"
    assert event["tool_calls"] == 0
    assert event["usage"]["total_tokens"] == 42
    assert event["total_tokens"] == 150


@pytest.mark.asyncio
async def test_post_model_trims_long_content(_journal_to_tmp: Path) -> None:
    chain = make_event_logger_hooks()
    long_text = "x" * 600
    ai = AIMessage(content=long_text)

    await chain.run_post_model(ai_message=ai, history=[], state=LoopState())

    [event] = _events(_journal_to_tmp)
    assert event["content_chars"] == 600
    assert event["content_preview"].endswith("\u2026")
    assert len(event["content_preview"]) == 501


@pytest.mark.asyncio
async def test_post_model_handles_missing_usage_metadata(
    _journal_to_tmp: Path,
) -> None:
    chain = make_event_logger_hooks()
    ai = AIMessage(content="")

    await chain.run_post_model(ai_message=ai, history=[], state=LoopState())

    [event] = _events(_journal_to_tmp)
    assert event["usage"] == {}
    assert event["content_chars"] == 0


@pytest.mark.asyncio
async def test_post_model_counts_tool_calls(_journal_to_tmp: Path) -> None:
    chain = make_event_logger_hooks()
    ai = AIMessage(
        content="",
        tool_calls=[
            {"id": "1", "name": "fake", "args": {"x": 1}},
            {"id": "2", "name": "fake", "args": {"x": 2}},
        ],
    )

    await chain.run_post_model(ai_message=ai, history=[], state=LoopState())

    [event] = _events(_journal_to_tmp)
    assert event["tool_calls"] == 2


@pytest.mark.asyncio
async def test_pre_tool_records_destructive_flag_and_args(
    _journal_to_tmp: Path,
) -> None:
    chain = make_event_logger_hooks()
    tool = _fake_tool(destructive=True)

    decision = await chain.run_pre_tool(
        tool=tool, args={"x": 5}, state=LoopState(turn_count=2),
    )

    assert decision is None
    [event] = _events(_journal_to_tmp)
    assert event["event"] == "pre_tool"
    assert event["tool"] == "fake"
    assert event["is_destructive"] is True
    assert "5" in event["args_preview"] and "x" in event["args_preview"]


@pytest.mark.asyncio
async def test_pre_tool_defaults_is_destructive_false(_journal_to_tmp: Path) -> None:
    chain = make_event_logger_hooks()
    tool = _fake_tool(destructive=False)

    await chain.run_pre_tool(tool=tool, args={}, state=LoopState())

    [event] = _events(_journal_to_tmp)
    assert event["is_destructive"] is False


@pytest.mark.asyncio
async def test_pre_tool_preview_survives_unserializable_args(
    _journal_to_tmp: Path,
) -> None:
    chain = make_event_logger_hooks()
    tool = _fake_tool()

    class _Weird:
        def __repr__(self) -> str:
            raise RuntimeError("boom")

    await chain.run_pre_tool(tool=tool, args={"w": _Weird()}, state=LoopState())

    [event] = _events(_journal_to_tmp)
    assert event["args_preview"] == "<unserializable>"


@pytest.mark.asyncio
async def test_post_tool_forwards_result_unchanged(_journal_to_tmp: Path) -> None:
    chain = make_event_logger_hooks()
    tool = _fake_tool()
    result = ToolResult(ok=True, output={"k": 1})

    returned = await chain.run_post_tool(
        tool=tool, args={"x": 1}, result=result, state=LoopState(turn_count=4),
    )

    assert returned is result
    [event] = _events(_journal_to_tmp)
    assert event["event"] == "post_tool"
    assert event["ok"] is True
    assert event["error"] is None
    assert event["output_chars"] > 0


@pytest.mark.asyncio
async def test_post_tool_records_error_without_output(_journal_to_tmp: Path) -> None:
    chain = make_event_logger_hooks()
    tool = _fake_tool()
    result = ToolResult(ok=False, error="permission denied")

    await chain.run_post_tool(
        tool=tool, args={"x": 1}, result=result, state=LoopState(),
    )

    [event] = _events(_journal_to_tmp)
    assert event["ok"] is False
    assert event["error"] == "permission denied"
    assert event["output_chars"] == 0


@pytest.mark.asyncio
async def test_post_tool_serializes_datetime_via_default_str(
    _journal_to_tmp: Path,
) -> None:
    import datetime as dt

    chain = make_event_logger_hooks()
    tool = _fake_tool()
    result = ToolResult(ok=True, output={"when": dt.datetime(2026, 4, 19)})

    await chain.run_post_tool(
        tool=tool, args={}, result=result, state=LoopState(),
    )

    [event] = _events(_journal_to_tmp)
    assert event["output_chars"] > 0


def test_wrap_with_event_logger_preserves_inner_order() -> None:
    inner_calls: list[str] = []

    async def _inner_pre_model(**_: Any) -> None:
        inner_calls.append("inner_pre_model")

    async def _inner_post_model(**_: Any) -> None:
        inner_calls.append("inner_post_model")

    async def _inner_pre_tool(**_: Any) -> None:
        inner_calls.append("inner_pre_tool")
        return None

    async def _inner_post_tool(**kw: Any) -> ToolResult:
        inner_calls.append("inner_post_tool")
        result: ToolResult = kw["result"]
        return result

    inner = HookChain(
        pre_model=[_inner_pre_model],
        post_model=[_inner_post_model],
        pre_tool=[_inner_pre_tool],
        post_tool=[_inner_post_tool],
    )
    wrapped = wrap_with_event_logger(inner)

    # Logger-first on write paths (pre_model / pre_tool); logger-last on
    # observation paths (post_model / post_tool) — so logging sees the
    # final result after inner transforms.
    assert len(wrapped.pre_model) == 2
    assert wrapped.pre_model[-1] is _inner_pre_model
    assert len(wrapped.post_model) == 2
    assert wrapped.post_model[0] is _inner_post_model
    assert len(wrapped.pre_tool) == 2
    assert wrapped.pre_tool[-1] is _inner_pre_tool
    assert len(wrapped.post_tool) == 2
    assert wrapped.post_tool[0] is _inner_post_tool


@pytest.mark.asyncio
async def test_wrap_with_event_logger_runs_end_to_end(
    _journal_to_tmp: Path,
) -> None:
    inner = HookChain()
    wrapped = wrap_with_event_logger(inner)
    tool = _fake_tool()

    await wrapped.run_pre_model(history=[], state=LoopState(turn_count=1))
    await wrapped.run_pre_tool(tool=tool, args={"x": 1}, state=LoopState())
    await wrapped.run_post_tool(
        tool=tool, args={"x": 1}, result=ToolResult(ok=True, output={}),
        state=LoopState(),
    )
    await wrapped.run_post_model(
        ai_message=AIMessage(content="done"), history=[], state=LoopState(),
    )

    events = [e["event"] for e in _events(_journal_to_tmp)]
    assert events == ["pre_model", "pre_tool", "post_tool", "post_model"]
