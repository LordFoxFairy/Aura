"""Tests for aura.core.registry — ToolRegistry."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from aura.core.history import tool_schema_for
from aura.core.registry import ToolRegistry
from aura.tools.base import AuraTool, ToolResult, build_tool
from aura.tools.read_file import read_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AParams(BaseModel):
    x: str


class _BParams(BaseModel):
    y: int


async def _noop(params: BaseModel) -> ToolResult:
    return ToolResult(ok=True)


_tool_a: AuraTool = build_tool(
    name="tool_a",
    description="first tool",
    input_model=_AParams,
    call=_noop,
    is_read_only=True,
)

_tool_b: AuraTool = build_tool(
    name="tool_b",
    description="second tool",
    input_model=_BParams,
    call=_noop,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_registry_empty_is_empty() -> None:
    reg = ToolRegistry(())
    assert reg.names() == []
    assert len(reg) == 0


def test_registry_accepts_tools() -> None:
    reg = ToolRegistry([_tool_a, _tool_b])
    assert len(reg) == 2
    assert "tool_a" in reg.names()
    assert "tool_b" in reg.names()


def test_registry_duplicate_name_raises() -> None:
    dup = build_tool(
        name="tool_a",
        description="duplicate",
        input_model=_AParams,
        call=_noop,
    )
    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolRegistry([_tool_a, dup])


def test_registry_getitem_by_name() -> None:
    reg = ToolRegistry([read_file])
    assert reg["read_file"] is read_file


def test_registry_missing_name_raises_keyerror() -> None:
    reg = ToolRegistry([_tool_a])
    with pytest.raises(KeyError):
        _ = reg["ghost"]


def test_registry_schemas_match_tool_schema_for() -> None:
    tools = [_tool_a, _tool_b]
    reg = ToolRegistry(tools)
    expected = [tool_schema_for(t) for t in tools]
    assert reg.schemas() == expected


def test_registry_contains() -> None:
    reg = ToolRegistry([read_file])
    assert "read_file" in reg
    assert "ghost" not in reg
