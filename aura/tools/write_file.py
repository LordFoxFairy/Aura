"""write_file built-in tool — create or overwrite UTF-8 text files."""
from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from aura.tools.base import (  # noqa: F401  (AuraTool used for Protocol)
    AuraTool,
    ToolResult,
)


class WriteFileParams(BaseModel):
    path: str = Field(description="File path. Parent directory must exist.")
    content: str = Field(
        description="UTF-8 text content to write. Overwrites any existing file."
    )


class WriteFileTool:
    name: str = "write_file"
    description: str = "Create or overwrite a UTF-8 text file. Parent directory must exist."
    input_model: type[BaseModel] = WriteFileParams
    is_read_only: bool = False
    is_destructive: bool = True
    is_concurrency_safe: bool = False  # parallel writes to same path corrupt state

    async def acall(self, params: BaseModel) -> ToolResult:
        assert isinstance(params, WriteFileParams)
        return await asyncio.to_thread(self._write_sync, params.path, params.content)

    @staticmethod
    def _write_sync(path: str, content: str) -> ToolResult:
        p = Path(path)

        if not p.parent.exists():
            return ToolResult(
                ok=False,
                error=f"parent directory does not exist: {p.parent}",
            )

        if p.is_dir():
            return ToolResult(ok=False, error=f"path is a directory: {path}")

        data = content.encode("utf-8")
        p.write_bytes(data)
        return ToolResult(ok=True, output={"written": len(data)})
