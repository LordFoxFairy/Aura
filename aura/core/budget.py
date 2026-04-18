"""Size-budget post-tool hook: truncate large outputs, optionally spill to disk."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aura.core.hooks import PostToolHook
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
        serialized = json.dumps(result.output)
        if len(serialized) <= max_chars:
            return result

        truncation: dict[str, Any] = {
            "truncated": True,
            "total_chars": len(serialized),
            "preview": serialized[:max_chars],
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
