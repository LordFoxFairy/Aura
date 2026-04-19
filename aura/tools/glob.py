"""File name pattern matching."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from aura.tools.base import AuraTool, ToolResult, build_tool


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


def _glob(params: GlobParams) -> ToolResult:
    root = Path(params.path).expanduser().resolve()
    if not root.exists():
        return ToolResult(ok=False, error=f"not found: {root}")
    if not root.is_dir():
        return ToolResult(ok=False, error=f"not a directory: {root}")

    files: list[str] = []
    truncated = False
    for p in sorted(root.glob(params.pattern)):
        if not p.is_file():
            continue
        rel = p.relative_to(root) if p.is_relative_to(root) else p
        files.append(str(rel))
        if len(files) >= params.max_results:
            truncated = True
            break

    return ToolResult(
        ok=True,
        output={"files": files, "count": len(files), "truncated": truncated},
    )


glob: AuraTool = build_tool(
    name="glob",
    description=(
        "List files matching a pathname pattern (e.g. '**/*.py' for all Python files "
        "recursively). Returns a sorted list of relative paths."
    ),
    input_model=GlobParams,
    call=_glob,
    is_read_only=True,
    is_concurrency_safe=True,
    max_result_size_chars=40_000,
)
