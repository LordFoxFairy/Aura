"""Regex search over file contents."""

from __future__ import annotations

import asyncio
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
            "Optional fnmatch-style glob to filter file names "
            "(e.g. '*.py'). Applied to basename only."
        ),
    )
    max_results: int = Field(
        default=100, ge=1, le=10_000,
        description="Maximum matches to return.",
    )


def _search_sync(
    pattern: str, root: Path, glob: str | None, max_results: int,
) -> ToolResult:
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return ToolResult(ok=False, error=f"invalid regex: {exc}")

    if not root.exists():
        return ToolResult(ok=False, error=f"not found: {root}")
    if not root.is_dir():
        return ToolResult(ok=False, error=f"not a directory: {root}")

    matches: list[dict[str, Any]] = []
    truncated = False

    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        if glob is not None and not fnmatch(file.name, glob):
            continue
        try:
            with file.open("r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, start=1):
                    if compiled.search(line):
                        rel = (
                            str(file.relative_to(root))
                            if file.is_relative_to(root)
                            else str(file)
                        )
                        matches.append({
                            "file": rel,
                            "line_num": line_num,
                            "line": line.rstrip("\n"),
                        })
                        if len(matches) >= max_results:
                            truncated = True
                            break
        except (OSError, UnicodeDecodeError):
            continue
        if truncated:
            break

    return ToolResult(
        ok=True,
        output={
            "matches": matches,
            "count": len(matches),
            "truncated": truncated,
        },
    )


async def _acall(params: BaseModel) -> ToolResult:
    assert isinstance(params, GrepParams)
    return await asyncio.to_thread(
        _search_sync,
        params.pattern,
        Path(params.path).expanduser().resolve(),
        params.glob,
        params.max_results,
    )


grep: AuraTool = build_tool(
    name="grep",
    description=(
        "Search file contents by regex. Returns matches with file path, line number, "
        "and line text. Optional fnmatch glob filters file names (e.g. glob='*.py')."
    ),
    input_model=GrepParams,
    call=_acall,
    is_read_only=True,
    is_concurrency_safe=True,
    max_result_size_chars=80_000,
)
