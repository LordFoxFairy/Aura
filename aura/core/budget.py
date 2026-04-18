"""Size-budget post-tool hook and usage-tracking post-model hook."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from aura.core.hooks import PostModelHook, PostToolHook
from aura.core.state import LoopState
from aura.tools.base import AuraTool, ToolResult


def make_size_budget_hook(
    *,
    max_chars: int = 10_000,
    spill_dir: Path | None = None,
) -> PostToolHook:
    async def _hook(
        *,
        tool: AuraTool,
        params: BaseModel,
        result: ToolResult,
        state: LoopState,
        **_: Any,
    ) -> ToolResult:
        if not result.ok or result.output is None:
            return result

        effective_max = getattr(tool, "max_result_size_chars", None) or max_chars

        serialized = json.dumps(result.output)
        if len(serialized) <= effective_max:
            return result

        truncation: dict[str, Any] = {
            "truncated": True,
            "total_chars": len(serialized),
            "preview": serialized[:effective_max],
        }
        if spill_dir is not None:
            spill_dir.mkdir(parents=True, exist_ok=True)
            spill_path = spill_dir / f"{uuid.uuid4().hex}.json"
            spill_path.write_text(serialized, encoding="utf-8")
            truncation["spill_path"] = str(spill_path)

        return ToolResult(
            ok=True,
            output=truncation,
            display=result.display,
        )

    return _hook


def make_usage_tracking_hook() -> PostModelHook:
    async def _hook(
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],
        state: LoopState,
        **_: Any,
    ) -> None:
        usage = getattr(ai_message, "usage_metadata", None)
        if not usage:
            return
        total = usage.get("total_tokens")
        if isinstance(total, int):
            state.total_tokens_used += total

    return _hook
