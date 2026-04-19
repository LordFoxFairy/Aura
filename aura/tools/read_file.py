"""read_file tool — UTF-8 read with 1MB cap."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from aura.tools.base import AuraTool, ToolResult, build_tool

_MAX_BYTES = 1024 * 1024


class ReadFileParams(BaseModel):
    path: str = Field(description="Absolute or relative file path to read.")


def _read(params: ReadFileParams) -> ToolResult:
    p = Path(params.path)
    if not p.exists():
        return ToolResult(ok=False, error=f"not found: {params.path}")
    size = p.stat().st_size
    if size > _MAX_BYTES:
        return ToolResult(ok=False, error=f"too large: {size} bytes > 1 MB")
    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return ToolResult(ok=False, error=f"not UTF-8: {exc}")
    return ToolResult(ok=True, output={"content": content, "lines": len(content.splitlines())})


read_file: AuraTool = build_tool(
    name="read_file",
    description="Read a UTF-8 text file (up to 1 MB).",
    input_model=ReadFileParams,
    call=_read,
    is_read_only=True,
    is_concurrency_safe=True,
)
