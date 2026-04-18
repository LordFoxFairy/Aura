"""Tests for aura.tools.base — AuraTool protocol and ToolResult."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from aura.tools.base import AuraTool, ToolResult

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
# Minimal duck-typed class for Protocol tests
# ---------------------------------------------------------------------------


class _Empty(BaseModel):
    pass


class _EchoTool:
    name: str = "echo"
    description: str = "echoes input"
    input_model: type[BaseModel] = _Empty
    is_read_only: bool = True
    is_destructive: bool = False
    is_concurrency_safe: bool = True

    async def acall(self, params: BaseModel) -> ToolResult:
        return ToolResult(ok=True, output={"echoed": True})


# ---------------------------------------------------------------------------
# Protocol isinstance checks
# ---------------------------------------------------------------------------


def test_minimal_duck_typed_class_satisfies_protocol() -> None:
    assert isinstance(_EchoTool(), AuraTool) is True


def test_acall_returns_tool_result() -> None:
    tool = _EchoTool()
    result = asyncio.run(tool.acall(_Empty()))
    assert isinstance(result, ToolResult)
    assert result.ok is True


def test_non_conforming_class_fails_isinstance() -> None:
    class _Missing:
        name: str = "bad"
        description: str = "missing attrs"
        # is_read_only, is_destructive, is_concurrency_safe omitted
        input_model: type[BaseModel] = _Empty

        async def acall(self, params: BaseModel) -> ToolResult:
            return ToolResult(ok=True)

    assert isinstance(_Missing(), AuraTool) is False
