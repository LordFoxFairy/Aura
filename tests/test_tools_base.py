"""Tests for aura.tools.base."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.domain.tool import (
    ToolError,
    ToolMetadata,
    ToolResult,
    ValidationResult,
)
from aura.domain.tool_meta_access import meta_dict
from aura.tools.base import Tool, build_tool


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
        ToolResult(**{})  # exercising missing required arg path


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
    assert meta_dict(tool).get("is_read_only") is True


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


def _noop() -> dict[str, Any]:
    return {}


def test_tool_metadata_defaults_rule_matcher_and_args_preview_to_none() -> None:
    from aura.domain.tool import tool_metadata
    meta = tool_metadata()
    assert meta["rule_matcher"] is None
    assert meta["args_preview"] is None


def test_tool_metadata_accepts_rule_matcher_callable() -> None:
    from aura.domain.tool import tool_metadata

    def matcher(args: dict[str, Any], content: str) -> bool:
        return args.get("cmd") == content

    meta = tool_metadata(rule_matcher=matcher)
    assert meta["rule_matcher"] is matcher


def test_tool_metadata_accepts_args_preview_callable() -> None:
    from aura.domain.tool import tool_metadata

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
    assert meta_dict(tool).get("rule_matcher") is matcher


def test_build_tool_stores_args_preview_in_metadata() -> None:
    def preview(args: dict[str, Any]) -> str:
        return "x"

    tool = build_tool(
        name="x", description="x", args_schema=_Empty, func=_noop,
        args_preview=preview,
    )
    assert meta_dict(tool).get("args_preview") is preview


def test_build_tool_without_new_kwargs_has_none_slots() -> None:
    tool = build_tool(
        name="x", description="x", args_schema=_Empty, func=_noop,
    )
    meta = meta_dict(tool)
    assert meta.get("rule_matcher") is None
    assert meta.get("args_preview") is None


class _AcceptAllTool(Tool):
    """Minimal ``Tool`` subclass used to exercise the default
    ``validate_input`` — it doesn't override the method, so callers
    should observe the base class's accept-everything behaviour.
    """

    name: str = "accept_all"
    description: str = "test fixture"
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=None,
        args_preview=None,
        timeout_sec=None,
    )

    def _run(self, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}


def test_tool_default_validate_input_accepts_empty_args() -> None:
    """Spec §4 — the base ``Tool.validate_input`` returns
    ``ValidationResult(invalid=False)`` regardless of args. Empty
    dict path: nothing to inspect, still accepted.
    """
    result = _AcceptAllTool().validate_input({})
    assert isinstance(result, ValidationResult)
    assert result.invalid is False
    assert result.reason == ""


def test_tool_default_validate_input_accepts_arbitrary_args() -> None:
    """Default impl is shape-agnostic — it ignores the contents and
    accepts. Subclasses opt into validation by overriding the method.
    """
    result = _AcceptAllTool().validate_input(
        {"path": "../etc/passwd", "scheme": "file://", "anything": object()},
    )
    assert result.invalid is False
    assert result.reason == ""


def test_tool_subclass_can_override_validate_input() -> None:
    """A subclass that overrides ``validate_input`` returns its own
    verdict — pins the override hook contract that Task 2 relies on.
    """

    class _RejectingTool(_AcceptAllTool):
        def validate_input(self, args: dict[str, Any]) -> ValidationResult:
            if args.get("path", "").startswith(".."):
                return ValidationResult(invalid=True, reason="relative path")
            return ValidationResult(invalid=False)

    tool = _RejectingTool()
    bad = tool.validate_input({"path": "../x"})
    assert bad.invalid is True
    assert bad.reason == "relative path"

    ok = tool.validate_input({"path": "/abs/x"})
    assert ok.invalid is False
