"""Regex search over file contents."""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.tools.base import ToolError, tool_metadata


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
    max_results: int = Field(
        default=100, ge=1, le=10_000, description="Maximum matches to return.",
    )


class Grep(BaseTool):
    name: str = "grep"
    description: str = (
        "Search file contents by regex. Returns matches with file path, line number, "
        "and line text. Optional fnmatch glob filters file names (e.g. glob='*.py')."
    )
    args_schema: type[BaseModel] = GrepParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_size_chars=80_000,
    )

    def _run(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        max_results: int = 100,
    ) -> dict[str, Any]:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regex: {exc}") from exc

        root = Path(path).expanduser().resolve()
        if not root.exists():
            raise ToolError(f"not found: {root}")
        if not root.is_dir():
            raise ToolError(f"not a directory: {root}")

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
                            rel = file.relative_to(root) if file.is_relative_to(root) else file
                            matches.append({
                                "file": str(rel),
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

        return {"matches": matches, "count": len(matches), "truncated": truncated}


grep: BaseTool = Grep()
