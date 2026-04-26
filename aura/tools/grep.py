"""Ripgrep-backed content search.

Shells out to ``rg`` (ripgrep). The three output modes mirror claude-code's
GrepTool: ``files_with_matches`` (default, just paths), ``count`` (per-file
match counts), and ``content`` (line-level matches with optional context).

VCS metadata directories (``.git``, ``.svn``, ``.hg``) are auto-excluded —
mirrors claude-code's VCS_DIRECTORIES_TO_EXCLUDE. Lines longer than
``max_columns`` (default 500) are truncated by rg with a ``[...]`` indicator.
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
    max_columns: int = Field(
        default=500,
        ge=1,
        le=10_000,
        description=(
            "Per-line character cap (rg --max-columns); over-long lines are "
            "elided with '[...]'."
        ),
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _context_only_in_content_mode(self) -> GrepParams:
        if self.output_mode != "content" and (self.context_before or self.context_after):
            raise ValueError("context flags require output_mode='content'")
        return self


def _preview(args: dict[str, Any]) -> str:
    return f"pattern: {args.get('pattern', '')}  @ {args.get('path', '.')}"


# Sentinel separators for content-mode output. rg's default separators
# (``:`` for matches, ``-`` for context) are ambiguous inside paths that
# legitimately contain them (e.g. ``src/v-42-release/foo.rs`` — a path
# whose ``-42-`` chews through the naïve ``-<digits>-`` heuristic). Using
# characters that cannot appear in POSIX file paths nor in rg's other
# output fields gives an unambiguous parse. ``\x1f`` (ASCII unit
# separator) for match lines; ``\x02`` for context lines — both are
# POSIX-safe argv (not NUL) and not part of Python's ``str.splitlines``
# terminator set (so we avoid the ``\x1e`` record-separator trap).
_MATCH_SEP = "\x1f"
_CTX_SEP = "\x02"


_VCS_EXCLUDE_GLOBS: tuple[str, ...] = ("!.git", "!.svn", "!.hg")


def _build_argv(p: GrepParams) -> list[str]:
    argv: list[str] = ["rg", "--line-number", f"--max-columns={p.max_columns}"]
    # VCS-metadata excludes go before user globs so an explicit positive
    # glob from the caller can still re-include them.
    for g in _VCS_EXCLUDE_GLOBS:
        argv += ["--glob", g]
    if p.case_insensitive:
        argv.append("-i")
    if p.multiline:
        argv += ["-U", "--multiline-dotall"]
    if p.output_mode == "content":
        argv += [
            f"--field-match-separator={_MATCH_SEP}",
            f"--field-context-separator={_CTX_SEP}",
        ]
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
    """Parse one rg --line-number content-mode output line.

    Match lines use ``_MATCH_SEP`` (unit separator); context lines use
    ``_CTX_SEP`` (STX). Split with maxsplit=2 so the ``text`` field may
    itself contain the separator byte without corrupting the parse.
    """
    mparts = line.split(_MATCH_SEP, 2)
    if len(mparts) == 3 and mparts[1].isdigit():
        return {
            "path": mparts[0],
            "line": int(mparts[1]),
            "text": mparts[2],
        }
    if has_context:
        cparts = line.split(_CTX_SEP, 2)
        if len(cparts) == 3 and cparts[1].isdigit():
            return {
                "path": cparts[0],
                "line": int(cparts[1]),
                "text": cparts[2],
                "is_context": True,
            }
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
        "case_insensitive (rg -i), glob, type, and multiline. VCS metadata "
        "dirs (.git/.svn/.hg) are auto-excluded; lines over max_columns "
        "(default 500 chars) are elided with '[...]'."
    )
    args_schema: type[BaseModel] = GrepParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_size_chars=80_000,
        rule_matcher=exact_match_on("pattern"),
        args_preview=_preview,
        # 30s matches the internal subprocess.run timeout we pass to ripgrep;
        # the outer loop wrapper is a belt-and-braces guard in case the rg
        # process ignores SIGALRM or Python's timeout plumbing is bypassed.
        timeout_sec=30.0,
        is_search_command=True,
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
