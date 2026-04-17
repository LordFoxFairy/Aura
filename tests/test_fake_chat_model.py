"""Tests for FakeChatModel fixture and chunk helpers (Task 15)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage  # noqa: F401

from tests.conftest import (
    FakeChatModel,
    FakeTurn,
    parallel_tool_calls_chunk,
    text_chunk,
    tool_call_chunk,
)

# ---------------------------------------------------------------------------
# Chunk helper tests
# ---------------------------------------------------------------------------


def test_text_chunk_non_final() -> None:
    chunk = text_chunk("hi")
    assert chunk.content == "hi"
    # chunk_position should not be "last" for non-final chunks
    pos = getattr(chunk, "chunk_position", None)
    assert pos != "last"


def test_text_chunk_final_sets_chunk_position_last() -> None:
    chunk = text_chunk("done", final=True)
    assert chunk.chunk_position == "last"


def test_tool_call_chunk_default_final_consolidates_tool_calls() -> None:
    """Accumulate text + tool_call_chunk → AIMessage.tool_calls populated."""
    c1 = text_chunk("call: ")
    c2 = tool_call_chunk("t", '{"x":1}', "tc_1")
    acc: AIMessageChunk = c1 + c2
    ai = AIMessage(content=acc.content, tool_calls=acc.tool_calls or [])
    assert len(ai.tool_calls) == 1
    assert ai.tool_calls[0]["id"] == "tc_1"
    assert ai.tool_calls[0]["name"] == "t"


def test_tool_call_chunk_non_final_has_no_chunk_position_last() -> None:
    """With final=False, the chunk does NOT have chunk_position='last'.

    In this LangChain version tool_calls is always computed from tool_call_chunks
    regardless of chunk_position; what final=False guarantees is that the chunk
    does not carry the 'last' marker used to signal end-of-stream to callers.
    """
    c = tool_call_chunk("t", '{"x":1}', "tc_1", final=False)
    assert getattr(c, "chunk_position", None) != "last"
    # tool_call_chunks IS present (partial form)
    assert len(c.tool_call_chunks) == 1


def test_parallel_tool_calls_chunk_produces_two_tool_calls() -> None:
    chunk = parallel_tool_calls_chunk([
        {"name": "a", "args_json": "{}", "id": "tc_1"},
        {"name": "b", "args_json": "{}", "id": "tc_2"},
    ])
    ai = AIMessage(content=chunk.content, tool_calls=chunk.tool_calls or [])
    assert len(ai.tool_calls) == 2
    assert ai.tool_calls[0]["id"] == "tc_1"
    assert ai.tool_calls[1]["id"] == "tc_2"


# ---------------------------------------------------------------------------
# FakeChatModel tests
# ---------------------------------------------------------------------------


def test_fake_chat_model_records_bind_tools() -> None:
    model = FakeChatModel()
    schema = {"type": "function", "function": {"name": "x", "description": "", "parameters": {}}}
    model.bind_tools([schema])
    assert model.seen_bound_tools == [[schema]]


@pytest.mark.asyncio
async def test_fake_chat_model_astream_yields_scripted_chunks() -> None:
    model = FakeChatModel(turns=[
        FakeTurn(chunks=[text_chunk("hi ", final=False), text_chunk("world", final=True)])
    ])
    contents = []
    async for chunk in model.astream([HumanMessage(content="test")]):
        contents.append(chunk.content)
    assert contents == ["hi ", "world"]
    assert model.astream_calls == 1


@pytest.mark.asyncio
async def test_fake_chat_model_agenerate_returns_consolidated_message() -> None:
    model = FakeChatModel(turns=[
        FakeTurn(chunks=[text_chunk("hello", final=True)])
    ])
    result = await model.ainvoke([HumanMessage(content="x")])
    assert isinstance(result, AIMessage)
    assert result.content == "hello"
    assert model.agenerate_calls == 1
    assert model.astream_calls == 0


@pytest.mark.asyncio
async def test_fake_chat_model_ainvoke_agenerate_tool_calls_populated() -> None:
    model = FakeChatModel(turns=[
        FakeTurn(chunks=[tool_call_chunk("t", "{}", "tc_1")])
    ])
    result = await model.ainvoke([HumanMessage(content="x")])
    assert isinstance(result, AIMessage)
    assert len(result.tool_calls) >= 1
    assert result.tool_calls[0]["id"] == "tc_1"


@pytest.mark.asyncio
async def test_fake_chat_model_runs_out_of_turns_raises() -> None:
    model = FakeChatModel(turns=[])
    with pytest.raises(RuntimeError, match="no scripted turns"):
        async for _ in model.astream([HumanMessage(content="x")]):
            pass
