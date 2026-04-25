"""write_file tool — create or overwrite UTF-8 text files.

Round 3B (F-02-007) — must-read-first staleness gate. The actual mtime
check that rejects writes when the on-disk file is newer than the
parent's last read lives in :mod:`aura.core.hooks.must_read_first`,
not here: the tool itself is dumb-by-design (open + write + return),
which keeps the no-permission-bypass invariant trivially auditable.
The hook fires BEFORE this tool runs and short-circuits the call
whenever the read fingerprint is missing or stale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.core.permissions.matchers import path_prefix_on
from aura.schemas.tool import ToolError, tool_metadata


class WriteFileParams(BaseModel):
    path: str = Field(
        description="File path. Missing parent directories are created automatically.",
    )
    content: str = Field(description="UTF-8 text content to write. Overwrites any existing file.")


def _preview(args: dict[str, Any]) -> str:
    return f"path: {args.get('path', '')}  ({len(args.get('content', ''))} chars)"


class WriteFile(BaseTool):
    name: str = "write_file"
    description: str = (
        "Create or overwrite a UTF-8 text file. "
        "Missing parent directories are created automatically."
    )
    args_schema: type[BaseModel] = WriteFileParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_destructive=True,
        rule_matcher=path_prefix_on("path"),
        args_preview=_preview,
    )

    def _run(self, path: str, content: str) -> dict[str, Any]:
        p = Path(path)
        if p.is_dir():
            raise ToolError(f"path is a directory: {path}")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolError(f"cannot create parent dir: {exc}") from exc
        data = content.encode("utf-8")
        p.write_bytes(data)
        return {"written": len(data)}


write_file: BaseTool = WriteFile()
