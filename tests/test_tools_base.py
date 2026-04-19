"""Tests for aura.tools.base."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.schemas.tool import ToolError, ToolResult
from aura.tools.base import build_tool


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


# ---------------------------------------------------------------------------
# Phase D additions: rule_matcher + args_preview metadata slots
# ---------------------------------------------------------------------------


def _noop() -> dict[str, Any]:
    return {}


def test_tool_metadata_defaults_rule_matcher_and_args_preview_to_none() -> None:
    from aura.schemas.tool import tool_metadata
    meta = tool_metadata()
    assert meta["rule_matcher"] is None
    assert meta["args_preview"] is None


def test_tool_metadata_accepts_rule_matcher_callable() -> None:
    from aura.schemas.tool import tool_metadata

    def matcher(args: dict[str, Any], content: str) -> bool:
        return args.get("cmd") == content

    meta = tool_metadata(rule_matcher=matcher)
    assert meta["rule_matcher"] is matcher


def test_tool_metadata_accepts_args_preview_callable() -> None:
    from aura.schemas.tool import tool_metadata

    def preview(args: dict[str, Any]) -> str:
        return f"cmd: {args.get('command', '')}"

    meta = tool_metadata(args_preview=preview)
    assert meta["args_preview"] is preview


def test_build_tool_stores_rule_matcher_in_metadata() -> None:
    def matcher(args: dict[str, Any], content: str) -> bool:
        return True

    tool = build_tool(
        name="x", description="x", args_schema=_Empty, func=_noop,
        rule_matcher=matcher,
    )
    assert (tool.metadata or {}).get("rule_matcher") is matcher


def test_build_tool_stores_args_preview_in_metadata() -> None:
    def preview(args: dict[str, Any]) -> str:
        return "x"

    tool = build_tool(
        name="x", description="x", args_schema=_Empty, func=_noop,
        args_preview=preview,
    )
    assert (tool.metadata or {}).get("args_preview") is preview


def test_build_tool_without_new_kwargs_has_none_slots() -> None:
    tool = build_tool(
        name="x", description="x", args_schema=_Empty, func=_noop,
    )
    meta = tool.metadata or {}
    assert meta.get("rule_matcher") is None
    assert meta.get("args_preview") is None
