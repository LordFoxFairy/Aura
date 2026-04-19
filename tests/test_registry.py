"""Tests for aura.core.registry — ToolRegistry."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.loop import ToolStep, partition_batches
from aura.core.registry import ToolRegistry
from aura.tools.base import ToolResult, build_tool
from aura.tools.read_file import read_file


class _AParams(BaseModel):
    x: str


class _BParams(BaseModel):
    y: int


def _noop_a(x: str) -> dict[str, Any]:
    return {}


def _noop_b(y: int) -> dict[str, Any]:
    return {}


_tool_a: BaseTool = build_tool(
    name="tool_a",
    description="first tool",
    args_schema=_AParams,
    func=_noop_a,
    is_read_only=True,
)

_tool_b: BaseTool = build_tool(
    name="tool_b",
    description="second tool",
    args_schema=_BParams,
    func=_noop_b,
)


def test_registry_empty_is_empty() -> None:
    reg = ToolRegistry(())
    assert len(reg) == 0


def test_registry_accepts_tools() -> None:
    reg = ToolRegistry([_tool_a, _tool_b])
    assert len(reg) == 2
    assert "tool_a" in reg
    assert "tool_b" in reg


def test_registry_duplicate_name_raises() -> None:
    dup = build_tool(
        name="tool_a",
        description="duplicate",
        args_schema=_AParams,
        func=_noop_a,
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


def test_registry_contains() -> None:
    reg = ToolRegistry([read_file])
    assert "read_file" in reg
    assert "ghost" not in reg


def test_registry_tools_returns_list_in_order() -> None:
    reg = ToolRegistry([_tool_a, _tool_b])
    tools = reg.tools()
    assert [t.name for t in tools] == ["tool_a", "tool_b"]


async def test_partition_empty_steps_returns_empty() -> None:
    assert partition_batches([]) == []


async def test_partition_all_safe_returns_single_batch() -> None:
    class _P(BaseModel):
        pass

    def _c() -> dict[str, Any]:
        return {}

    t = build_tool(
        name="t", description="t", args_schema=_P, func=_c,
        is_read_only=True, is_concurrency_safe=True,
    )
    steps = [
        ToolStep(
            tool_call={"name": "t", "args": {}, "id": f"tc_{i}"},
            tool=t, args={}, decision=None,
        )
        for i in range(3)
    ]
    batches = partition_batches(steps)
    assert len(batches) == 1
    assert len(batches[0]) == 3


async def test_partition_interleaved_unsafe_breaks_batches() -> None:
    """[safe, safe, UNSAFE, safe, UNSAFE, safe] → [[s,s], [U], [s], [U], [s]]"""
    class _P(BaseModel):
        pass

    def _c() -> dict[str, Any]:
        return {}

    safe_tool = build_tool(
        name="s", description="s", args_schema=_P, func=_c,
        is_read_only=True, is_concurrency_safe=True,
    )
    unsafe_tool = build_tool(
        name="u", description="u", args_schema=_P, func=_c,
        is_destructive=True, is_concurrency_safe=False,
    )

    def _step(tool: BaseTool, name: str, idx: int) -> ToolStep:
        return ToolStep(
            tool_call={"name": name, "args": {}, "id": f"tc_{idx}"},
            tool=tool, args={}, decision=None,
        )

    steps = [
        _step(safe_tool, "s", 0), _step(safe_tool, "s", 1),
        _step(unsafe_tool, "u", 2),
        _step(safe_tool, "s", 3),
        _step(unsafe_tool, "u", 4),
        _step(safe_tool, "s", 5),
    ]
    batches = partition_batches(steps)
    sizes = [len(b) for b in batches]
    assert sizes == [2, 1, 1, 1, 1]


async def test_partition_short_circuited_step_goes_solo() -> None:
    """A step with decision != None runs solo even if its tool is_concurrency_safe."""
    class _P(BaseModel):
        pass

    def _c() -> dict[str, Any]:
        return {}

    safe = build_tool(
        name="s", description="s", args_schema=_P, func=_c,
        is_read_only=True, is_concurrency_safe=True,
    )

    steps = [
        ToolStep(
            tool_call={"name": "s", "args": {}, "id": "1"},
            tool=safe, args={}, decision=None,
        ),
        ToolStep(
            tool_call={"name": "s", "args": {}, "id": "2"},
            tool=safe, args={}, decision=ToolResult(ok=False, error="denied"),
        ),
        ToolStep(
            tool_call={"name": "s", "args": {}, "id": "3"},
            tool=safe, args={}, decision=None,
        ),
    ]
    batches = partition_batches(steps)
    sizes = [len(b) for b in batches]
    assert sizes == [1, 1, 1]
