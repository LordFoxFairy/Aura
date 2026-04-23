"""read_file tool — UTF-8 read with 1 MB cap + offset/limit slicing.

Mirrors claude-code FileReadTool: `offset` (0-indexed line start in Aura —
claude-code uses 1-indexed; we normalize to 0 for parity with Python slice
semantics) and `limit` (max lines). `partial` in the return dict flips true
when the caller saw less than the full file; the must-read-first invariant
uses it to reject edit_file after a sliced read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.core.permissions.matchers import path_prefix_on
from aura.schemas.tool import ToolError, tool_metadata

_MAX_BYTES = 1024 * 1024


class ReadFileParams(BaseModel):
    path: str = Field(description="Absolute or relative file path to read.")
    offset: int = Field(
        default=0,
        ge=0,
        description="0-indexed line offset; lines before this are skipped.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Max lines to return; None reads to end (subject to MB cap).",
    )


def _preview(args: dict[str, Any]) -> str:
    return f"path: {args.get('path', '')}"


class ReadFile(BaseTool):
    name: str = "read_file"
    description: str = "Read a UTF-8 text file (up to 1 MB) with optional offset/limit."
    args_schema: type[BaseModel] = ReadFileParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True, is_concurrency_safe=True,
        rule_matcher=path_prefix_on("path"),
        args_preview=_preview,
        # Plain filesystem I/O up to 1 MB. 10s is generous — anything slower
        # is a stuck NFS mount or a dying disk; surface the error rather
        # than hang.
        timeout_sec=10.0,
        # Partial reads (offset/limit) benefit from UI fold: they're
        # search-like slices. Full reads also fold — the renderer's
        # threshold (>20 lines) protects short reads.
        is_search_command=True,
    )

    def _run(
        self, path: str, offset: int = 0, limit: int | None = None,
    ) -> dict[str, Any]:
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

        # splitlines(keepends=True) preserves line terminators so re-joining
        # gives byte-identical slice output for the lines we return.
        all_lines = content.splitlines(keepends=True)
        total_lines = len(all_lines)

        if offset >= total_lines:
            # Over-shoot: be honest about it — partial=True so the caller
            # knows they didn't see the end of the file.
            return {
                "content": "",
                "lines": 0,
                "total_lines": total_lines,
                "offset": offset,
                "limit": limit,
                "partial": True,
            }

        end = offset + limit if limit is not None else None
        sliced = all_lines[offset:end]
        joined = "".join(sliced)

        partial = offset > 0 or (
            limit is not None and offset + limit < total_lines
        )
        return {
            "content": joined,
            "lines": len(sliced),
            "total_lines": total_lines,
            "offset": offset,
            "limit": limit,
            "partial": partial,
        }


read_file: BaseTool = ReadFile()
