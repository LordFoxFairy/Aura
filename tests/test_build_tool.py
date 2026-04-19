"""Tests for aura.tools.base.build_tool — factory that returns BaseTool."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.tools.base import ToolError, build_tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MinParams(BaseModel):
    value: str


def _fixed_run(value: str) -> dict[str, Any]:
    return {"fixed": True, "value": value}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_tool_fail_closed_defaults() -> None:
    tool = build_tool(
        name="minimal",
        description="minimal tool",
        args_schema=_MinParams,
        func=_fixed_run,
    )
    meta = tool.metadata or {}
    assert meta.get("is_read_only") is False
    assert meta.get("is_destructive") is False
    assert meta.get("is_concurrency_safe") is False


def test_build_tool_returns_base_tool_instance() -> None:
    tool = build_tool(
        name="minimal",
        description="minimal tool",
        args_schema=_MinParams,
        func=_fixed_run,
    )
    assert isinstance(tool, BaseTool)


@pytest.mark.asyncio
async def test_build_tool_ainvoke_calls_func() -> None:
    tool = build_tool(
        name="minimal",
        description="minimal tool",
        args_schema=_MinParams,
        func=_fixed_run,
    )
    result = await tool.ainvoke({"value": "x"})
    assert result == {"fixed": True, "value": "x"}


def test_build_tool_max_result_size_chars_defaults_to_none() -> None:
    tool = build_tool(
        name="x",
        description="x",
        args_schema=_MinParams,
        func=_fixed_run,
    )
    assert (tool.metadata or {}).get("max_result_size_chars") is None


def test_build_tool_max_result_size_chars_is_stored() -> None:
    tool = build_tool(
        name="x",
        description="x",
        args_schema=_MinParams,
        func=_fixed_run,
        max_result_size_chars=500,
    )
    assert (tool.metadata or {}).get("max_result_size_chars") == 500


class _DoublerParams(BaseModel):
    x: int


class _EchoParams(BaseModel):
    x: int


def _sync_doubler(x: int) -> dict[str, Any]:
    return {"doubled": x * 2}


async def _async_echo(x: int) -> dict[str, Any]:
    return {"echoed": x}


@pytest.mark.asyncio
async def test_build_tool_auto_wraps_sync_function() -> None:
    tool = build_tool(
        name="doubler", description="x2", args_schema=_DoublerParams, func=_sync_doubler,
        is_read_only=True,
    )
    result = await tool.ainvoke({"x": 21})
    assert result == {"doubled": 42}


@pytest.mark.asyncio
async def test_build_tool_passes_through_async_function() -> None:
    tool = build_tool(
        name="echoer", description="echo", args_schema=_EchoParams, coroutine=_async_echo,
        is_read_only=True,
    )
    result = await tool.ainvoke({"x": 7})
    assert result == {"echoed": 7}


@pytest.mark.asyncio
async def test_build_tool_tool_error_propagates() -> None:
    def _boom(value: str) -> dict[str, Any]:
        raise ToolError("boom!")

    tool = build_tool(
        name="boomer", description="always boom", args_schema=_MinParams, func=_boom,
    )
    with pytest.raises(ToolError, match="boom!"):
        await tool.ainvoke({"value": "x"})
