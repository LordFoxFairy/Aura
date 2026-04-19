"""read_file tool — UTF-8 read with 1MB cap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.schemas.tool import ToolError, tool_metadata

_MAX_BYTES = 1024 * 1024


class ReadFileParams(BaseModel):
    path: str = Field(description="Absolute or relative file path to read.")


class ReadFile(BaseTool):
    name: str = "read_file"
    description: str = "Read a UTF-8 text file (up to 1 MB)."
    args_schema: type[BaseModel] = ReadFileParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True, is_concurrency_safe=True,
    )

    def _run(self, path: str) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise ToolError(f"not found: {path}")
        size = p.stat().st_size
        if size > _MAX_BYTES:
            raise ToolError(f"too large: {size} bytes > 1 MB")
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"not UTF-8: {exc}") from exc
        return {"content": content, "lines": len(content.splitlines())}


read_file: BaseTool = ReadFile()
