"""Regex search over file contents."""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aura.tools.base import AuraTool, ToolResult, build_tool


class GrepParams(BaseModel):
    pattern: str = Field(description="Python regex pattern to search for.")
    path: str = Field(
        default=".",
        description="Root directory to search. Defaults to current working directory.",
    )
    glob: str | None = Field(
        default=None,
        description=(
            "Optional fnmatch-style glob to filter file names (e.g. '*.py'). "
            "Applied to basename only."
        ),
    )
    max_results: int = Field(default=100, ge=1, le=10_000, description="Maximum matches to return.")


def _search(params: GrepParams) -> ToolResult:
    try:
        compiled = re.compile(params.pattern)
    except re.error as exc:
        return ToolResult(ok=False, error=f"invalid regex: {exc}")

    root = Path(params.path).expanduser().resolve()
    if not root.exists():
        return ToolResult(ok=False, error=f"not found: {root}")
    if not root.is_dir():
        return ToolResult(ok=False, error=f"not a directory: {root}")

    matches: list[dict[str, Any]] = []
    truncated = False

    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        if params.glob is not None and not fnmatch(file.name, params.glob):
            continue
        try:
            with file.open("r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, start=1):
                    if compiled.search(line):
                        rel = file.relative_to(root) if file.is_relative_to(root) else file
                        matches.append({
                            "file": str(rel),
                            "line_num": line_num,
                            "line": line.rstrip("\n"),
                        })
                        if len(matches) >= params.max_results:
                            truncated = True
                            break
        except (OSError, UnicodeDecodeError):
            continue
        if truncated:
            break

    return ToolResult(
        ok=True,
        output={"matches": matches, "count": len(matches), "truncated": truncated},
    )


grep: AuraTool = build_tool(
    name="grep",
    description=(
        "Search file contents by regex. Returns matches with file path, line number, "
        "and line text. Optional fnmatch glob filters file names (e.g. glob='*.py')."
    ),
    input_model=GrepParams,
    call=_search,
    is_read_only=True,
    is_concurrency_safe=True,
    max_result_size_chars=80_000,
)
