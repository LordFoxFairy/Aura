"""Tests for aura.core.registry — ToolRegistry."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from aura.core.history import tool_schema_for
from aura.core.loop import ToolStep
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


async def test_partition_empty_steps_returns_empty() -> None:
    assert ToolRegistry.partition_batches([]) == []


async def test_partition_all_safe_returns_single_batch() -> None:
    class _P(BaseModel):
        pass

    async def _c(params: BaseModel) -> ToolResult:
        return ToolResult(ok=True)

    t = build_tool(
        name="t", description="t", input_model=_P, call=_c,
        is_read_only=True, is_concurrency_safe=True,
    )
    steps = [
        ToolStep(
            tool_call={"name": "t", "args": {}, "id": f"tc_{i}"},
            tool=t, params=_P(), decision=None,
        )
        for i in range(3)
    ]
    batches = ToolRegistry.partition_batches(steps)
    assert len(batches) == 1
    assert len(batches[0]) == 3


async def test_partition_interleaved_unsafe_breaks_batches() -> None:
    """[safe, safe, UNSAFE, safe, UNSAFE, safe] → [[s,s], [U], [s], [U], [s]]"""
    class _P(BaseModel):
        pass

    async def _c(p: BaseModel) -> ToolResult:
        return ToolResult(ok=True)

    safe_tool = build_tool(
        name="s", description="s", input_model=_P, call=_c,
        is_read_only=True, is_concurrency_safe=True,
    )
    unsafe_tool = build_tool(
        name="u", description="u", input_model=_P, call=_c,
        is_destructive=True, is_concurrency_safe=False,
    )

    def _step(tool: AuraTool, name: str, idx: int) -> ToolStep:
        return ToolStep(
            tool_call={"name": name, "args": {}, "id": f"tc_{idx}"},
            tool=tool, params=_P(), decision=None,
        )

    steps = [
        _step(safe_tool, "s", 0), _step(safe_tool, "s", 1),
        _step(unsafe_tool, "u", 2),
        _step(safe_tool, "s", 3),
        _step(unsafe_tool, "u", 4),
        _step(safe_tool, "s", 5),
    ]
    batches = ToolRegistry.partition_batches(steps)
    sizes = [len(b) for b in batches]
    assert sizes == [2, 1, 1, 1, 1]


async def test_partition_short_circuited_step_goes_solo() -> None:
    """A step with decision != None runs solo even if its tool is_concurrency_safe."""
    class _P(BaseModel):
        pass

    async def _c(p: BaseModel) -> ToolResult:
        return ToolResult(ok=True)

    safe = build_tool(
        name="s", description="s", input_model=_P, call=_c,
        is_read_only=True, is_concurrency_safe=True,
    )

    steps = [
        ToolStep(
            tool_call={"name": "s", "args": {}, "id": "1"},
            tool=safe, params=_P(), decision=None,
        ),
        ToolStep(
            tool_call={"name": "s", "args": {}, "id": "2"},
            tool=safe, params=_P(), decision=ToolResult(ok=False, error="denied"),
        ),
        ToolStep(
            tool_call={"name": "s", "args": {}, "id": "3"},
            tool=safe, params=_P(), decision=None,
        ),
    ]
    batches = ToolRegistry.partition_batches(steps)
    sizes = [len(b) for b in batches]
    assert sizes == [1, 1, 1]
