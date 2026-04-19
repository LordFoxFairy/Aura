"""File name pattern matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.tools.base import ToolError, tool_metadata


class GlobParams(BaseModel):
    pattern: str = Field(
        description=(
            "Pathlib-style glob pattern, e.g. '**/*.py' or 'src/*.md'. "
            "Uses '**' for recursive."
        ),
    )
    path: str = Field(
        default=".", description="Root directory. Defaults to current working directory.",
    )
    max_results: int = Field(
        default=500, ge=1, le=10_000,
        description="Maximum matches to return.",
    )


class Glob(BaseTool):
    name: str = "glob"
    description: str = (
        "List files matching a pathname pattern (e.g. '**/*.py' for all Python files "
        "recursively). Returns a sorted list of relative paths."
    )
    args_schema: type[BaseModel] = GlobParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True, is_concurrency_safe=True, max_result_size_chars=40_000,
    )

    def _run(self, pattern: str, path: str = ".", max_results: int = 500) -> dict[str, Any]:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            raise ToolError(f"not found: {root}")
        if not root.is_dir():
            raise ToolError(f"not a directory: {root}")

        files: list[str] = []
        truncated = False
        for p in sorted(root.glob(pattern)):
            if not p.is_file():
                continue
            rel = p.relative_to(root) if p.is_relative_to(root) else p
            files.append(str(rel))
            if len(files) >= max_results:
                truncated = True
                break

        return {"files": files, "count": len(files), "truncated": truncated}


glob: BaseTool = Glob()
