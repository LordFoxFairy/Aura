"""read_file tool — UTF-8 read with 1MB cap."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from aura.tools.base import AuraTool, ToolResult, build_tool

_MAX_BYTES = 1024 * 1024  # 1 MB


class ReadFileParams(BaseModel):
    path: str = Field(description="Absolute or relative file path to read.")


def _read_sync(path: str) -> ToolResult:
    p = Path(path)
    if not p.exists():
        return ToolResult(ok=False, error=f"not found: {path}")
    size = p.stat().st_size
    if size > _MAX_BYTES:
        return ToolResult(ok=False, error=f"too large: {size} bytes > 1 MB")
    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return ToolResult(ok=False, error=f"not UTF-8: {exc}")
    lines = len(content.splitlines())
    return ToolResult(ok=True, output={"content": content, "lines": lines})


async def _acall(params: BaseModel) -> ToolResult:
    assert isinstance(params, ReadFileParams)
    return await asyncio.to_thread(_read_sync, params.path)


read_file: AuraTool = build_tool(
    name="read_file",
    description="Read a UTF-8 text file (up to 1 MB).",
    input_model=ReadFileParams,
    call=_acall,
    is_read_only=True,
    is_concurrency_safe=True,
)
