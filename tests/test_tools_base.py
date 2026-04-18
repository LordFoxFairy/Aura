"""Tests for aura.tools.base — AuraTool dataclass and ToolResult."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from aura.tools.base import AuraTool, ToolResult, build_tool

# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


def test_tool_result_required_ok() -> None:
    tr = ToolResult(ok=True)
    assert tr.ok is True
    assert tr.output is None
    assert tr.error is None
    assert tr.display is None


def test_tool_result_full_fields() -> None:
    tr = ToolResult(ok=False, output={"x": 1}, error="oops", display="boom")
    assert tr.ok is False
    assert tr.output == {"x": 1}
    assert tr.error == "oops"
    assert tr.display == "boom"


def test_tool_result_ok_is_required() -> None:
    with pytest.raises(TypeError):
        ToolResult()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# AuraTool nominal isinstance checks
# ---------------------------------------------------------------------------


class _Empty(BaseModel):
    pass


@pytest.mark.asyncio
async def test_build_tool_returns_aura_tool_instance() -> None:
    def _call(params: BaseModel) -> ToolResult:
        return ToolResult(ok=True)

    tool = build_tool(
        name="x", description="x", input_model=_Empty, call=_call, is_read_only=True,
    )
    assert isinstance(tool, AuraTool)
    assert tool.name == "x"


@pytest.mark.asyncio
async def test_aura_tool_acall_returns_tool_result() -> None:
    def _call(params: BaseModel) -> ToolResult:
        return ToolResult(ok=True, output={"k": 1})

    tool = build_tool(
        name="x", description="x", input_model=_Empty, call=_call, is_read_only=True,
    )
    result = await tool.acall(_Empty())
    assert result.ok is True
    assert result.output == {"k": 1}
