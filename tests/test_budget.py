"""Tests for aura.core.budget.make_size_budget_hook."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from aura.core.budget import make_size_budget_hook
from aura.core.hooks import PostToolHook
from aura.core.state import LoopState
from aura.tools.base import AuraTool, ToolResult, build_tool


class _P(BaseModel):
    pass


async def _no_op_call(params: BaseModel) -> ToolResult:
    return ToolResult(ok=True)


_stub_tool: AuraTool = build_tool(
    name="stub",
    description="stub",
    input_model=_P,
    call=_no_op_call,
)


async def _invoke(hook: PostToolHook, result: ToolResult) -> ToolResult:
    return await hook(
        tool=_stub_tool,
        params=_P(),
        result=result,
        state=LoopState(),
    )


async def test_small_output_passes_through_unchanged() -> None:
    hook = make_size_budget_hook(max_chars=1000)
    original = ToolResult(ok=True, output={"a": 1})
    result = await _invoke(hook, original)
    assert result.output == {"a": 1}
    assert result.ok is True


async def test_error_result_passes_through_regardless_of_size() -> None:
    hook = make_size_budget_hook(max_chars=10)
    large_error = "e" * 50_000
    original = ToolResult(ok=False, error=large_error)
    result = await _invoke(hook, original)
    assert result.ok is False
    assert result.output is None
    assert result.error == large_error


async def test_none_output_passes_through() -> None:
    hook = make_size_budget_hook(max_chars=10)
    original = ToolResult(ok=True, output=None)
    result = await _invoke(hook, original)
    assert result.ok is True
    assert result.output is None


async def test_large_output_truncates_without_spill_dir() -> None:
    hook = make_size_budget_hook(max_chars=100)
    original = ToolResult(ok=True, output={"content": "x" * 50_000})
    result = await _invoke(hook, original)
    assert result.ok is True
    assert isinstance(result.output, dict)
    assert result.output["truncated"] is True
    assert result.output["total_chars"] > 100
    assert len(result.output["preview"]) <= 100
    assert "spill_path" not in result.output


async def test_large_output_writes_spill_file_when_spill_dir_given(tmp_path: Path) -> None:
    hook = make_size_budget_hook(max_chars=100, spill_dir=tmp_path)
    original_output = {"content": "x" * 50_000}
    original = ToolResult(ok=True, output=original_output)
    result = await _invoke(hook, original)
    assert isinstance(result.output, dict)
    assert "spill_path" in result.output
    spill_path = Path(result.output["spill_path"])
    assert spill_path.exists()
    assert spill_path.read_text(encoding="utf-8") == json.dumps(original_output)


async def test_spill_dir_auto_created_if_missing(tmp_path: Path) -> None:
    new_dir = tmp_path / "nonexistent_subdir"
    assert not new_dir.exists()
    hook = make_size_budget_hook(max_chars=100, spill_dir=new_dir)
    original = ToolResult(ok=True, output={"content": "x" * 50_000})
    await _invoke(hook, original)
    assert new_dir.exists()


async def test_preview_is_json_prefix(tmp_path: Path) -> None:
    max_chars = 200
    hook = make_size_budget_hook(max_chars=max_chars, spill_dir=tmp_path)
    original_output = {"content": "y" * 50_000}
    original = ToolResult(ok=True, output=original_output)
    result = await _invoke(hook, original)
    assert isinstance(result.output, dict)
    expected_preview = json.dumps(original_output)[:max_chars]
    assert result.output["preview"] == expected_preview
