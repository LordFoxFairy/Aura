"""write_file tool — create or overwrite UTF-8 text files."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from aura.tools.base import AuraTool, ToolResult, build_tool


class WriteFileParams(BaseModel):
    path: str = Field(description="File path. Parent directory must exist.")
    content: str = Field(description="UTF-8 text content to write. Overwrites any existing file.")


def _write(params: WriteFileParams) -> ToolResult:
    p = Path(params.path)
    if not p.parent.exists():
        return ToolResult(ok=False, error=f"parent directory does not exist: {p.parent}")
    if p.is_dir():
        return ToolResult(ok=False, error=f"path is a directory: {params.path}")
    data = params.content.encode("utf-8")
    p.write_bytes(data)
    return ToolResult(ok=True, output={"written": len(data)})


write_file: AuraTool = build_tool(
    name="write_file",
    description="Create or overwrite a UTF-8 text file. Parent directory must exist.",
    input_model=WriteFileParams,
    call=_write,
    is_destructive=True,
)
