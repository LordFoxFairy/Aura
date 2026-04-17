"""Tests for aura.core.history — serialize/deserialize helpers and tool_schema_for."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from aura.core.history import deserialize_messages, serialize_messages, tool_schema_for
from aura.tools.base import ToolResult

# ---------------------------------------------------------------------------
# Minimal fake tool for schema tests
# ---------------------------------------------------------------------------


class _FakeParams(BaseModel):
    path: str  # required — no default
    count: int = 10  # has default


class _FakeTool:
    name = "fake_tool"
    description = "does fake things"
    input_model = _FakeParams
    is_read_only = True
    is_destructive = False
    is_concurrency_safe = True

    async def acall(self, params: BaseModel) -> ToolResult:  # pragma: no cover
        return ToolResult(ok=True)


# ---------------------------------------------------------------------------
# tool_schema_for
# ---------------------------------------------------------------------------


def test_tool_schema_shape() -> None:
    schema = tool_schema_for(_FakeTool())  # type: ignore[arg-type]
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "fake_tool"
    assert fn["description"] == "does fake things"
    assert isinstance(fn["parameters"], dict)
    assert "properties" in fn["parameters"]


def test_tool_schema_required_covers_non_default_fields() -> None:
    schema = tool_schema_for(_FakeTool())  # type: ignore[arg-type]
    required = schema["function"]["parameters"]["required"]
    assert "path" in required
    assert "count" not in required


# ---------------------------------------------------------------------------
# Round-trip: all four message types in one test
# ---------------------------------------------------------------------------


def test_serialize_messages_roundtrip_human_ai_tool_system() -> None:
    msgs = [
        HumanMessage(content="hello"),
        AIMessage(content="hi", tool_calls=[{"name": "t", "args": {}, "id": "tc_1"}]),
        ToolMessage(content="ok", tool_call_id="tc_1", name="t"),
        SystemMessage(content="you are helpful"),
    ]
    restored = deserialize_messages(serialize_messages(msgs))
    assert len(restored) == 4
    assert restored[0].content == "hello"
    assert isinstance(restored[0], HumanMessage)
    assert isinstance(restored[1], AIMessage)
    # tool_calls id preserved
    assert restored[1].tool_calls[0]["id"] == "tc_1"
    assert isinstance(restored[2], ToolMessage)
    assert restored[2].tool_call_id == "tc_1"
    assert isinstance(restored[3], SystemMessage)


def test_serialize_preserves_tool_call_id_on_ToolMessage() -> None:
    msg = ToolMessage(content="result", tool_call_id="tc_42", name="fake_tool")
    [restored] = deserialize_messages(serialize_messages([msg]))
    assert isinstance(restored, ToolMessage)
    assert restored.tool_call_id == "tc_42"


def test_serialize_preserves_ai_message_tool_calls() -> None:
    tool_call = {"name": "fake_tool", "args": {"path": "/tmp"}, "id": "tc_99"}
    msg = AIMessage(content="", tool_calls=[tool_call])
    [restored] = deserialize_messages(serialize_messages([msg]))
    assert isinstance(restored, AIMessage)
    assert restored.tool_calls[0]["id"] == "tc_99"
    assert restored.tool_calls[0]["name"] == "fake_tool"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_deserialize_unknown_type_raises_ValueError() -> None:
    with pytest.raises(ValueError, match="unknown"):
        deserialize_messages([{"type": "unknown", "content": "x"}])


def test_deserialize_missing_type_raises_ValueError() -> None:
    with pytest.raises(ValueError, match="unknown"):
        deserialize_messages([{"content": "no type here"}])


# ---------------------------------------------------------------------------
# Edge case
# ---------------------------------------------------------------------------


def test_empty_list_roundtrips() -> None:
    assert serialize_messages([]) == []
    assert deserialize_messages([]) == []
