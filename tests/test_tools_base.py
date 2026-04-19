"""Tests for aura.tools.base."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.tools.base import ToolError, ToolResult, build_tool


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


class _Empty(BaseModel):
    pass


@pytest.mark.asyncio
async def test_build_tool_returns_base_tool_instance() -> None:
    def _run() -> dict[str, Any]:
        return {"ok": True}

    tool = build_tool(
        name="x", description="x", args_schema=_Empty, func=_run, is_read_only=True,
    )
    assert isinstance(tool, BaseTool)
    assert tool.name == "x"
    assert (tool.metadata or {}).get("is_read_only") is True


@pytest.mark.asyncio
async def test_build_tool_ainvoke_returns_raw_output() -> None:
    def _run() -> dict[str, Any]:
        return {"k": 1}

    tool = build_tool(
        name="x", description="x", args_schema=_Empty, func=_run, is_read_only=True,
    )
    result = await tool.ainvoke({})
    assert result == {"k": 1}


@pytest.mark.asyncio
async def test_build_tool_ainvoke_propagates_tool_error() -> None:
    def _boom() -> dict[str, Any]:
        raise ToolError("kaboom")

    tool = build_tool(
        name="x", description="x", args_schema=_Empty, func=_boom,
    )
    with pytest.raises(ToolError, match="kaboom"):
        await tool.ainvoke({})
