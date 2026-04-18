"""Tests for aura.core.system_prompt.build_system_prompt."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from pydantic import BaseModel

import pytest

from aura.core.registry import ToolRegistry
from aura.core.system_prompt import (
    _AURA_MD_MAX_BYTES,
    build_system_prompt,
)
from aura.tools.base import AuraTool, ToolResult, build_tool


def _empty_registry() -> ToolRegistry:
    return ToolRegistry([])


def _make_tool(name: str, description: str, read_only: bool = False, destructive: bool = False) -> AuraTool:
    class _P(BaseModel):
        pass

    async def _call(params: BaseModel) -> ToolResult:
        return ToolResult(ok=True)

    return build_tool(
        name=name,
        description=description,
        input_model=_P,
        call=_call,
        is_read_only=read_only,
        is_destructive=destructive,
    )


def _registry_with(*tools: AuraTool) -> ToolRegistry:
    return ToolRegistry(list(tools))


def test_identity_section_mentions_aura(tmp_path: Path) -> None:
    result = build_system_prompt(registry=_empty_registry(), cwd=tmp_path)
    assert "Aura" in result


def test_environment_section_includes_cwd_and_date(tmp_path: Path) -> None:
    fixed_now = dt.datetime(2026, 4, 17, tzinfo=dt.timezone.utc)
    result = build_system_prompt(
        registry=_empty_registry(),
        cwd=tmp_path,
        now=fixed_now,
    )
    assert "2026-04-17" in result
    assert str(tmp_path) in result


def test_tools_section_lists_all_tools(tmp_path: Path) -> None:
    tool_a = _make_tool("alpha", "does alpha")
    tool_b = _make_tool("beta", "does beta")
    result = build_system_prompt(
        registry=_registry_with(tool_a, tool_b),
        cwd=tmp_path,
    )
    assert "alpha" in result
    assert "beta" in result


def test_tools_section_empty_is_explicit(tmp_path: Path) -> None:
    result = build_system_prompt(registry=_empty_registry(), cwd=tmp_path)
    assert "<tools>none enabled</tools>" in result


def test_aura_md_discovered_in_cwd(tmp_path: Path) -> None:
    aura_md = tmp_path / "AURA.md"
    aura_md.write_text("project: hello-world")

    result = build_system_prompt(registry=_empty_registry(), cwd=tmp_path)
    assert "project: hello-world" in result
    assert "project_memory" in result


def test_aura_md_walk_up_finds_parent(tmp_path: Path) -> None:
    # AURA.md 在 tmp_path；cwd 在更深的子目录
    aura_md = tmp_path / "AURA.md"
    aura_md.write_text("ancestor: found-me")

    deep = tmp_path / "sub" / "deep"
    deep.mkdir(parents=True)

    result = build_system_prompt(registry=_empty_registry(), cwd=deep)
    assert "ancestor: found-me" in result


def test_aura_md_over_cap_is_truncated(tmp_path: Path) -> None:
    # 写入 20 KB，超出 10 KB 上限
    big_content = "x" * (20 * 1024)
    (tmp_path / "AURA.md").write_text(big_content)

    result = build_system_prompt(registry=_empty_registry(), cwd=tmp_path)
    assert "(truncated)" in result
    # 实际注入的内容不超过上限（加若干标签开销）
    project_memory_len = len(result.encode("utf-8"))
    # 粗估：system prompt 基底 + 10 KB max + 少量标签，远小于 20 KB raw
    assert project_memory_len < len(big_content)


def test_aura_md_absent_skipped(tmp_path: Path) -> None:
    # 确保没有 AURA.md 影响测试（tmp_path 是隔离目录）
    result = build_system_prompt(registry=_empty_registry(), cwd=tmp_path)
    assert "project_memory" not in result


def test_tools_flags_appear_in_output(tmp_path: Path) -> None:
    ro_tool = _make_tool("reader", "reads things", read_only=True)
    destr_tool = _make_tool("destroyer", "destroys things", destructive=True)
    result = build_system_prompt(
        registry=_registry_with(ro_tool, destr_tool),
        cwd=tmp_path,
    )
    assert "read-only" in result
    assert "destructive" in result
