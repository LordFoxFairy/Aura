"""read_file built-in tool — read UTF-8 text files up to 1 MB."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field

from aura.tools.base import (  # noqa: F401  (AuraTool used for Protocol)
    AuraTool,
    PermissionResult,
    ToolResult,
)

_MAX_BYTES = 1024 * 1024  # 1 MB


class ReadFileParams(BaseModel):
    path: str = Field(description="Absolute or relative file path to read.")


class ReadFileTool:
    name: str = "read_file"
    description: str = "Read a UTF-8 text file (up to 1 MB)."
    input_model: type[BaseModel] = ReadFileParams
    is_read_only: bool = True
    is_destructive: bool = False
    is_concurrency_safe: bool = True

    def check_permissions(self, params: BaseModel) -> PermissionResult:
        return PermissionResult(allow=True)

    async def acall(self, params: BaseModel) -> ToolResult:
        p = cast(ReadFileParams, params)
        return await asyncio.to_thread(self._read_sync, p.path)

    @staticmethod
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
