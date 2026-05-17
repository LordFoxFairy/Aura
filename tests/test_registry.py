"""Tests for capabilities tool registry facades and assemble_tool_pool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from aura.capabilities.tools.catalog import assemble_tool_pool
from aura.capabilities.tools.catalog import assemble_tool_pool as assemble_capability_tool_pool
from aura.capabilities.tools.registry import ToolRegistry, ToolRegistryError
from aura.capabilities.tools.registry import ToolRegistry as CapabilityToolRegistry
from aura.capabilities.tools.registry import ToolRegistryError as CapabilityToolRegistryError
from aura.core.loop import ToolStep, partition_batches
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool
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


def test_core_tool_registry_facade_points_at_capabilities_module() -> None:
    assert ToolRegistry is CapabilityToolRegistry
    assert ToolRegistryError is CapabilityToolRegistryError
    assert assemble_tool_pool is assemble_capability_tool_pool
    assert ToolRegistry.__module__ == "aura.capabilities.tools.registry"
    assert assemble_tool_pool.__module__ == "aura.capabilities.tools.catalog"


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


def test_registry_register_adds_tool() -> None:
    reg = ToolRegistry([_tool_a])
    reg.register(_tool_b)
    assert "tool_b" in reg
    assert reg["tool_b"] is _tool_b


def test_registry_register_rejects_duplicate_name() -> None:
    reg = ToolRegistry([_tool_a])
    dup = build_tool(
        name="tool_a",
        description="duplicate",
        args_schema=_AParams,
        func=_noop_a,
    )
    with pytest.raises(ValueError, match="already registered"):
        reg.register(dup)


def test_registry_unregister_removes_tool() -> None:
    reg = ToolRegistry([_tool_a, _tool_b])
    reg.unregister("tool_a")
    assert "tool_a" not in reg
    assert "tool_b" in reg


def test_registry_unregister_is_idempotent_on_missing_name() -> None:
    reg = ToolRegistry([_tool_a])
    reg.unregister("ghost")
    reg.unregister("ghost")
    assert "tool_a" in reg


def _mk(name: str) -> BaseTool:
    return build_tool(
        name=name,
        description=name,
        args_schema=_AParams,
        func=_noop_a,
    )


def test_assemble_tool_pool_concats_builtins_then_mcp_sorted() -> None:
    builtin_b = _mk("b_builtin")
    builtin_a = _mk("a_builtin")
    mcp_y = _mk("y_mcp")
    mcp_x = _mk("x_mcp")
    pool = assemble_tool_pool([builtin_b, builtin_a], [mcp_y, mcp_x])
    assert list(pool.keys()) == ["a_builtin", "b_builtin", "x_mcp", "y_mcp"]


def test_assemble_tool_pool_builtin_wins_on_collision_and_journals(
    tmp_path: Path,
) -> None:
    from aura.core import journal

    journal_path = tmp_path / "audit.jsonl"
    journal.configure(journal_path)
    try:
        builtin_bash = _mk("bash")
        mcp_bash = _mk("bash")
        mcp_other = _mk("zzz_other")
        pool = assemble_tool_pool([builtin_bash], [mcp_bash, mcp_other])
        assert pool["bash"] is builtin_bash
        assert "zzz_other" in pool
        assert len(pool) == 2
    finally:
        journal.reset()
    events = [
        json.loads(line)
        for line in journal_path.read_text().splitlines()
        if line.strip()
    ]
    shadow_events = [e for e in events if e["event"] == "mcp_tool_shadowed"]
    assert len(shadow_events) == 1
    assert shadow_events[0]["tool"] == "bash"
    assert shadow_events[0]["winner"] == "builtin"
    assert shadow_events[0]["shadowed_by"] == "builtin"


def test_assemble_tool_pool_empty_inputs() -> None:
    assert assemble_tool_pool([], []) == {}


def test_assemble_tool_pool_dedupes_intra_mcp_silently(tmp_path: Path) -> None:
    from aura.core import journal

    journal_path = tmp_path / "audit.jsonl"
    journal.configure(journal_path)
    try:
        mcp_first = _mk("dup")
        mcp_second = _mk("dup")
        pool = assemble_tool_pool([], [mcp_first, mcp_second])
        assert len(pool) == 1
        assert "dup" in pool
    finally:
        journal.reset()
    events = [
        json.loads(line)
        for line in journal_path.read_text().splitlines()
        if line.strip()
    ]
    shadow_events = [e for e in events if e["event"] == "mcp_tool_shadowed"]
    assert len(shadow_events) == 1
    assert shadow_events[0]["winner"] == "mcp"
    assert shadow_events[0]["shadowed_by"] == "mcp"


# ---------------------------------------------------------------------------
# Phase 2 Task 7 — ``mcp_overrides_builtin`` collision policy.
# Default (False): builtin wins (preserves historical behavior).
# Opt-in (True): MCP wins, builtin is shadowed.
# Either branch emits ``mcp_tool_shadowed`` with a ``winner`` field that
# captures the policy outcome.
# ---------------------------------------------------------------------------


def test_assemble_tool_pool_default_builtin_wins_and_winner_builtin(
    tmp_path: Path,
) -> None:
    """Default flag (False) preserves builtin-wins: the builtin entry stays
    in the pool and the journal event names ``winner="builtin"``."""
    from aura.core import journal

    journal_path = tmp_path / "audit.jsonl"
    journal.configure(journal_path)
    try:
        builtin = _mk("collide")
        mcp = _mk("collide")
        pool = assemble_tool_pool([builtin], [mcp])
        assert pool["collide"] is builtin
        assert len(pool) == 1
    finally:
        journal.reset()
    events = [
        json.loads(line)
        for line in journal_path.read_text().splitlines()
        if line.strip()
    ]
    shadow_events = [e for e in events if e["event"] == "mcp_tool_shadowed"]
    assert len(shadow_events) == 1
    assert shadow_events[0]["tool"] == "collide"
    assert shadow_events[0]["winner"] == "builtin"


def test_assemble_tool_pool_mcp_overrides_true_mcp_wins_and_winner_mcp(
    tmp_path: Path,
) -> None:
    """Flag flipped to True: the MCP tool replaces the builtin in the pool
    and the journal event names ``winner="mcp"``."""
    from aura.core import journal

    journal_path = tmp_path / "audit.jsonl"
    journal.configure(journal_path)
    try:
        builtin = _mk("collide")
        mcp = _mk("collide")
        pool = assemble_tool_pool([builtin], [mcp], mcp_overrides=True)
        assert pool["collide"] is mcp
        assert len(pool) == 1
    finally:
        journal.reset()
    events = [
        json.loads(line)
        for line in journal_path.read_text().splitlines()
        if line.strip()
    ]
    shadow_events = [e for e in events if e["event"] == "mcp_tool_shadowed"]
    assert len(shadow_events) == 1
    assert shadow_events[0]["tool"] == "collide"
    assert shadow_events[0]["winner"] == "mcp"


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


# ---------------------------------------------------------------------------
# Phase 2 Task 4 — ``aura_metadata: ToolMetadata`` enforcement at registration.
# Every Aura tool must declare its capability flags through ``ToolMetadata``;
# the registry refuses anything else so loop / hook / CLI readers can trust
# the typed surface.
# ---------------------------------------------------------------------------


def _bare_tool_without_aura_metadata(name: str = "bare") -> BaseTool:
    """Synthetic tool that bypasses ``build_tool`` so ``aura_metadata`` is
    never set. Stand-in for the failure mode the registry must reject."""

    async def _coro(x: str = "") -> dict[str, Any]:
        return {}

    return StructuredTool(
        name=name,
        description="bare tool — no aura_metadata",
        args_schema=_AParams,
        coroutine=_coro,
    )


def test_registry_register_rejects_tool_without_aura_metadata() -> None:
    bare = _bare_tool_without_aura_metadata("bare_tool")
    reg = ToolRegistry()
    with pytest.raises(ToolRegistryError, match="aura_metadata"):
        reg.register(bare)
    assert "bare_tool" not in reg


def test_registry_init_rejects_tool_without_aura_metadata() -> None:
    bare = _bare_tool_without_aura_metadata("bare_tool")
    with pytest.raises(ToolRegistryError, match="aura_metadata"):
        ToolRegistry([bare])


def test_registry_register_error_names_offending_tool() -> None:
    """Operators land on ``ToolRegistryError`` straight from a stack trace;
    the message must point at the specific tool whose registration failed,
    not a generic 'missing metadata' line."""
    bare = _bare_tool_without_aura_metadata("offender")
    reg = ToolRegistry()
    with pytest.raises(ToolRegistryError, match="'offender'"):
        reg.register(bare)


def test_registry_error_inherits_aura_error() -> None:
    """``ToolRegistryError`` is part of the AuraError hierarchy so callers
    using ``except AuraError`` catch it uniformly with other expected errors."""
    from aura.errors import AuraError

    assert issubclass(ToolRegistryError, AuraError)
