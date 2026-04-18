"""Tests for FakeChatModel fixture."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tests.conftest import FakeChatModel, FakeTurn


def test_fake_chat_model_records_bind_tools() -> None:
    model = FakeChatModel()
    schema = {"type": "function", "function": {"name": "x", "description": "", "parameters": {}}}
    model.bind_tools([schema])
    assert model.seen_bound_tools == [[schema]]


@pytest.mark.asyncio
async def test_fake_chat_model_ainvoke_yields_scripted_message() -> None:
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="hi"))])
    result = await model.ainvoke([HumanMessage(content="x")])
    assert isinstance(result, AIMessage)
    assert result.content == "hi"
    assert model.ainvoke_calls == 1


@pytest.mark.asyncio
async def test_fake_chat_model_ainvoke_preserves_tool_calls() -> None:
    msg = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "tc_1"}])
    model = FakeChatModel(turns=[FakeTurn(message=msg)])
    result = await model.ainvoke([HumanMessage(content="x")])
    assert isinstance(result, AIMessage)
    assert result.tool_calls[0]["id"] == "tc_1"


@pytest.mark.asyncio
async def test_fake_chat_model_runs_out_of_turns_raises() -> None:
    model = FakeChatModel(turns=[])
    with pytest.raises(RuntimeError, match="no scripted turns"):
        await model.ainvoke([HumanMessage(content="x")])
