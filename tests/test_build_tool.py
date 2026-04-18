"""Tests for aura.tools.base — build_tool factory and _Tool dataclass."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import BaseModel

from aura.tools.base import AuraTool, ToolResult, build_tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MinParams(BaseModel):
    value: str


async def _fixed_call(params: BaseModel) -> ToolResult:
    return ToolResult(ok=True, output={"fixed": True})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_tool_fail_closed_defaults() -> None:
    tool = build_tool(
        name="minimal",
        description="minimal tool",
        input_model=_MinParams,
        call=_fixed_call,
    )
    assert tool.is_read_only is False
    assert tool.is_destructive is False
    assert tool.is_concurrency_safe is False


def test_build_tool_satisfies_protocol() -> None:
    tool = build_tool(
        name="minimal",
        description="minimal tool",
        input_model=_MinParams,
        call=_fixed_call,
    )
    assert isinstance(tool, AuraTool) is True


@pytest.mark.asyncio
async def test_build_tool_acall_delegates_to_call() -> None:
    tool = build_tool(
        name="minimal",
        description="minimal tool",
        input_model=_MinParams,
        call=_fixed_call,
    )
    result = await tool.acall(_MinParams(value="x"))
    assert result.ok is True
    assert result.output == {"fixed": True}


def test_build_tool_is_frozen() -> None:
    tool = build_tool(
        name="minimal",
        description="minimal tool",
        input_model=_MinParams,
        call=_fixed_call,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        tool.name = "changed"


def test_build_tool_max_result_size_chars_defaults_to_none() -> None:
    tool = build_tool(
        name="x",
        description="x",
        input_model=_MinParams,
        call=_fixed_call,
    )
    assert getattr(tool, "max_result_size_chars", "not set") is None


def test_build_tool_max_result_size_chars_is_stored() -> None:
    from aura.tools.base import _Tool

    tool = build_tool(
        name="x",
        description="x",
        input_model=_MinParams,
        call=_fixed_call,
        max_result_size_chars=500,
    )
    assert isinstance(tool, _Tool)
    assert tool.max_result_size_chars == 500
