"""Tests for aura.core.history.tool_schema_for."""

from __future__ import annotations

from pydantic import BaseModel

from aura.core.history import tool_schema_for
from aura.tools.base import ToolResult


class _FakeParams(BaseModel):
    path: str
    count: int = 10


class _FakeTool:
    name = "fake_tool"
    description = "does fake things"
    input_model = _FakeParams
    is_read_only = True
    is_destructive = False
    is_concurrency_safe = True

    async def acall(self, params: BaseModel) -> ToolResult:  # pragma: no cover
        return ToolResult(ok=True)


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
