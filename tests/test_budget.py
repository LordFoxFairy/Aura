"""Tests for aura.core.hooks.budget."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks import PostToolHook
from aura.core.hooks.budget import make_size_budget_hook
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool


class _P(BaseModel):
    pass


def _no_op() -> dict[str, Any]:
    return {}


_stub_tool: BaseTool = build_tool(
    name="stub",
    description="stub",
    args_schema=_P,
    func=_no_op,
)


async def _invoke(hook: PostToolHook, result: ToolResult) -> ToolResult:
    args: dict[str, Any] = {}
    return await hook(
        tool=_stub_tool,
        args=args,
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


async def test_per_tool_max_result_size_chars_overrides_global() -> None:
    narrow_tool: BaseTool = build_tool(
        name="narrow",
        description="small budget",
        args_schema=_P,
        func=_no_op,
        max_result_size_chars=50,
    )

    hook = make_size_budget_hook(max_chars=10_000)
    result = ToolResult(ok=True, output={"content": "x" * 1000})

    out = await hook(
        tool=narrow_tool, args={}, result=result, state=LoopState(),
    )

    assert isinstance(out.output, dict)
    assert out.output.get("truncated") is True
    total_chars = out.output.get("total_chars")
    assert isinstance(total_chars, int) and total_chars > 50


async def test_no_per_tool_budget_falls_back_to_global() -> None:
    plain_tool: BaseTool = build_tool(
        name="plain",
        description="no budget",
        args_schema=_P,
        func=_no_op,
    )

    hook = make_size_budget_hook(max_chars=100)
    result = ToolResult(ok=True, output={"content": "x" * 500})

    out = await hook(
        tool=plain_tool, args={}, result=result, state=LoopState(),
    )

    assert out.output.get("truncated") is True


async def test_usage_tracking_hook_accumulates_total_tokens() -> None:
    from aura.core.hooks.budget import make_usage_tracking_hook

    hook = make_usage_tracking_hook()
    state = LoopState()

    ai1 = AIMessage(content="hi", usage_metadata={
        "input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
    })
    await hook(ai_message=ai1, history=[], state=state)
    assert state.total_tokens_used == 30

    ai2 = AIMessage(content="ok", usage_metadata={
        "input_tokens": 5, "output_tokens": 15, "total_tokens": 20,
    })
    await hook(ai_message=ai2, history=[], state=state)
    assert state.total_tokens_used == 50


async def test_usage_tracking_hook_falls_back_to_estimator_when_usage_missing() -> None:
    """Round 11 audit fix — when ``usage_metadata`` is missing the hook
    falls back to a char/4 estimator (DashScope, some Ollama, self-hosted).
    Pre-fix, ``state.total_tokens_used`` stayed pinned at zero so auto-
    compact never armed and the status bar lied about utilization.
    """
    from aura.core.hooks.budget import make_usage_tracking_hook

    hook = make_usage_tracking_hook()
    state = LoopState()

    ai = AIMessage(content="no usage here")  # 13 chars → 3 tokens
    await hook(ai_message=ai, history=[], state=state)
    assert state.total_tokens_used == 13 // 4


def test_default_hooks_returns_populated_chain() -> None:
    from aura.core.hooks import HookChain
    from aura.core.hooks.budget import default_hooks

    hooks = default_hooks()
    assert isinstance(hooks, HookChain)
    # v0.13: default_hooks now ships a pre_model hook that flips the
    # buddy mood to "thinking" during model invocation. Pre-buddy
    # default_hooks shipped no pre_model entries; updating the bound
    # to >= 1 keeps the assertion meaningful (chain is populated)
    # without pinning the exact count.
    assert len(hooks.pre_model) >= 1
    assert len(hooks.post_model) >= 1
    assert len(hooks.post_tool) >= 1
