"""write_file tool — create or overwrite UTF-8 text files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.tools.base import ToolError, tool_metadata


class WriteFileParams(BaseModel):
    path: str = Field(description="File path. Parent directory must exist.")
    content: str = Field(description="UTF-8 text content to write. Overwrites any existing file.")


class WriteFile(BaseTool):
    name: str = "write_file"
    description: str = "Create or overwrite a UTF-8 text file. Parent directory must exist."
    args_schema: type[BaseModel] = WriteFileParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_destructive=True,
    )

    def _run(self, path: str, content: str) -> dict[str, Any]:
        p = Path(path)
        if not p.parent.exists():
            raise ToolError(f"parent directory does not exist: {p.parent}")
        if p.is_dir():
            raise ToolError(f"path is a directory: {path}")
        data = content.encode("utf-8")
        p.write_bytes(data)
        return {"written": len(data)}


write_file: BaseTool = WriteFile()
