"""File name pattern matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.core.permissions.matchers import exact_match_on
from aura.schemas.tool import ToolError, tool_metadata


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
    sort: Literal["mtime", "alphabetical"] = Field(
        default="mtime",
        description=(
            "'mtime' (default) returns newest-first, matching claude-code. "
            "'alphabetical' returns lexicographic — use when stable ordering matters."
        ),
    )


def _preview(args: dict[str, Any]) -> str:
    return f"pattern: {args.get('pattern', '')}  @ {args.get('path', '.')}"


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


class Glob(BaseTool):
    name: str = "glob"
    description: str = (
        "List files matching a pathname pattern (e.g. '**/*.py' for all Python files "
        "recursively). Returns paths sorted newest-first by default."
    )
    args_schema: type[BaseModel] = GlobParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True, is_concurrency_safe=True, max_result_size_chars=40_000,
        rule_matcher=exact_match_on("pattern"),
        args_preview=_preview,
        # Glob is pathlib-only (no subprocess). 10s already indicates a
        # pathological repo (10M+ entries or NFS stall); bail rather than
        # freeze the turn.
        timeout_sec=10.0,
        is_search_command=True,
    )

    def _run(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 500,
        sort: Literal["mtime", "alphabetical"] = "mtime",
    ) -> dict[str, Any]:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            raise ToolError(f"not found: {root}")
        if not root.is_dir():
            raise ToolError(f"not a directory: {root}")

        matches: list[Path] = [p for p in root.glob(pattern) if p.is_file()]

        if sort == "mtime":
            matches.sort(key=_safe_mtime, reverse=True)
        else:
            matches.sort()

        truncated = len(matches) > max_results
        if truncated:
            matches = matches[:max_results]

        files = [
            str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
            for p in matches
        ]
        return {"files": files, "count": len(files), "truncated": truncated}


glob: BaseTool = Glob()
