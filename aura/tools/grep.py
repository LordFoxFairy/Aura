"""Ripgrep-backed content search.

Shells out to ``rg`` (ripgrep). The three output modes mirror claude-code's
GrepTool: ``files_with_matches`` (default, just paths), ``count`` (per-file
match counts), and ``content`` (line-level matches with optional context).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from aura.core.permissions.matchers import exact_match_on
from aura.schemas.tool import ToolError, tool_metadata

OutputMode = Literal["content", "files_with_matches", "count"]


class GrepParams(BaseModel):
    pattern: str = Field(description="Regex pattern (ripgrep syntax) to search for.")
    path: str = Field(
        default=".",
        description="Directory or file to search. Defaults to current working directory.",
    )
    output_mode: OutputMode = Field(
        default="files_with_matches",
        description="'files_with_matches' (paths), 'count' (per-file), or 'content' (lines).",
    )
    case_insensitive: bool = Field(
        default=False, description="Case-insensitive match (rg -i).",
    )
    glob: str | None = Field(
        default=None,
        description="Glob filter passed to rg --glob (e.g. '*.py').",
    )
    file_type: str | None = Field(
        default=None,
        alias="type",
        description="File type filter passed to rg --type (e.g. 'python').",
    )
    multiline: bool = Field(
        default=False,
        description="Enable cross-line matching (rg -U --multiline-dotall).",
    )
    context_before: int = Field(
        default=0, ge=0, description="Lines of context before each match (rg -B).",
    )
    context_after: int = Field(
        default=0, ge=0, description="Lines of context after each match (rg -A).",
    )
    head_limit: int = Field(
        default=250,
        ge=0,
        le=5000,
        description="Cap on entries/lines returned.",
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _context_only_in_content_mode(self) -> GrepParams:
        if self.output_mode != "content" and (self.context_before or self.context_after):
            raise ValueError("context flags require output_mode='content'")
        return self


def _preview(args: dict[str, Any]) -> str:
    return f"pattern: {args.get('pattern', '')}  @ {args.get('path', '.')}"


def _build_argv(p: GrepParams) -> list[str]:
    argv: list[str] = ["rg", "--line-number"]
    if p.case_insensitive:
        argv.append("-i")
    if p.multiline:
        argv += ["-U", "--multiline-dotall"]
    if p.output_mode == "content":
        if p.context_after > 0:
            argv += ["-A", str(p.context_after)]
        if p.context_before > 0:
            argv += ["-B", str(p.context_before)]
    if p.glob:
        argv += ["--glob", p.glob]
    if p.file_type:
        argv += ["--type", p.file_type]
    if p.output_mode == "files_with_matches":
        argv.append("--files-with-matches")
    elif p.output_mode == "count":
        argv.append("--count")
    argv += ["--", p.pattern, p.path]
    return argv


def _parse_content_line(line: str, has_context: bool) -> dict[str, Any] | None:
    # rg --line-number emits `path:lineno:text` for matches and, under
    # -A/-B, `path-lineno-text` for context lines. Path may contain '-',
    # so scan for the first `-<digits>-` separator.
    mparts = line.split(":", 2)
    if len(mparts) == 3 and mparts[1].isdigit():
        return {
            "path": mparts[0],
            "line": int(mparts[1]),
            "text": mparts[2],
        }
    if has_context:
        i = 0
        while True:
            j = line.find("-", i)
            if j == -1:
                break
            k = line.find("-", j + 1)
            if k == -1:
                break
            mid = line[j + 1 : k]
            if mid.isdigit():
                return {
                    "path": line[:j],
                    "line": int(mid),
                    "text": line[k + 1 :],
                    "is_context": True,
                }
            i = j + 1
    return None


def _parse_count_line(line: str) -> tuple[str, int] | None:
    path, _, n = line.rpartition(":")
    if path and n.isdigit():
        return path, int(n)
    return None


class Grep(BaseTool):
    name: str = "grep"
    description: str = (
        "Search file contents via ripgrep. Default returns matching file paths "
        "('files_with_matches'); other modes: 'content' (lines with optional "
        "context_before/after), 'count' (per-file match totals). Supports "
        "case_insensitive (rg -i), glob, type, and multiline."
    )
    args_schema: type[BaseModel] = GrepParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_size_chars=80_000,
        rule_matcher=exact_match_on("pattern"),
        args_preview=_preview,
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        if shutil.which("rg") is None:
            raise ToolError(
                "grep requires ripgrep (rg) on PATH. "
                "Install via 'brew install ripgrep' or platform equivalent.",
            )
        params = GrepParams.model_validate(kwargs)
        argv = _build_argv(params)
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=30, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError("grep: ripgrep timed out after 30s") from exc

        if proc.returncode >= 2:
            raise ToolError(f"grep: {(proc.stderr or '').strip()}")

        # exit 1 = no matches; exit 0 = matches. Both yield parseable stdout.
        lines = [ln for ln in proc.stdout.splitlines() if ln]
        limit = params.head_limit

        if params.output_mode == "files_with_matches":
            truncated = len(lines) > limit
            return {
                "mode": "files_with_matches",
                "files": lines[:limit],
                "truncated": truncated,
            }

        if params.output_mode == "count":
            pairs = [pair for ln in lines if (pair := _parse_count_line(ln))]
            truncated = len(pairs) > limit
            kept = pairs[:limit]
            counts = {p: n for p, n in kept}
            return {
                "mode": "count",
                "counts": counts,
                "total": sum(counts.values()),
                "truncated": truncated,
            }

        # content mode — parse matches + optional context lines
        has_context = params.context_before > 0 or params.context_after > 0
        parsed: list[dict[str, Any]] = []
        for ln in lines:
            if ln == "--":
                continue  # rg inserts -- between context groups
            entry = _parse_content_line(ln, has_context)
            if entry is not None:
                parsed.append(entry)
        truncated = len(parsed) > limit
        return {
            "mode": "content",
            "matches": parsed[:limit],
            "truncated": truncated,
        }


grep: BaseTool = Grep()
