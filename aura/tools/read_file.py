"""read_file — UTF-8/UTF-16 read with 1 MB cap + offset/limit slicing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aura.core.permissions.matchers import path_prefix_on
from aura.schemas.tool import ToolError, ToolMetadata, ValidationResult
from aura.tools.base import Tool

_MAX_BYTES = 1024 * 1024

_TOKEN_BUDGET = 25_000
_CHARS_PER_TOKEN_HEURISTIC = 4

# Interactive / pseudo-file / kernel-memory paths the tool always refuses.
_BLOCKED_DEVICE_PATHS: frozenset[str] = frozenset({
    "/dev/stdin",
    "/dev/tty",
    "/dev/console",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/zero",
    "/dev/random",
    "/dev/urandom",
    "/dev/full",
    "/dev/fd/0",
    "/dev/fd/1",
    "/dev/fd/2",
    "/proc/self/fd/0",
    "/proc/self/fd/1",
    "/proc/self/fd/2",
    "/proc/kcore",
    "/proc/kmem",
})


def _resolve_blocked_device(path: str) -> str | None:
    # strict=False lets missing files surface "not found" via the main path.
    try:
        resolved = str(Path(path).resolve(strict=False))
    except (OSError, RuntimeError):
        return None
    if resolved in _BLOCKED_DEVICE_PATHS:
        return resolved
    return None


def _reject_blocked_device(path: str) -> None:
    resolved = _resolve_blocked_device(path)
    if resolved is not None:
        raise ToolError(
            f"refusing to read {resolved!r} — kernel/interactive device "
            "endpoint (would block, return garbage, or expose kernel memory)",
        )


def _decode_with_bom(data: bytes) -> str:
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8")
    return data.decode("utf-8")


def _validate_content_tokens(content: str) -> None:
    estimated_tokens = len(content) // _CHARS_PER_TOKEN_HEURISTIC
    if estimated_tokens > _TOKEN_BUDGET:
        raise ToolError(
            f"file content too large: ~{estimated_tokens} tokens "
            f"exceeds budget {_TOKEN_BUDGET}; use offset+limit to slice",
        )


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


class ReadFile(Tool):
    name: str = "read_file"
    description: str = (
        "Read a text file (UTF-8 / UTF-16 LE / UTF-16 BE / UTF-8-BOM) with "
        "optional offset/limit. Files exceeding 1 MB are head-truncated to "
        "the cap with partial=True and truncated_at_bytes set."
    )
    args_schema: type[BaseModel] = ReadFileParams
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=path_prefix_on("path"),
        args_preview=_preview,
        timeout_sec=10.0,
        capability_flags=frozenset({"search_command"}),
    )

    def validate_input(self, args: dict[str, Any]) -> ValidationResult:
        path = args.get("path", "")
        if not isinstance(path, str):
            return ValidationResult(invalid=False)
        resolved = _resolve_blocked_device(path)
        if resolved is not None:
            return ValidationResult(
                invalid=True,
                reason=(
                    f"refusing to read {resolved!r} — kernel/interactive "
                    "device endpoint (would block, return garbage, or "
                    "expose kernel memory)"
                ),
            )
        return ValidationResult(invalid=False)

    def _run(
        self, path: str, offset: int = 0, limit: int | None = None,
    ) -> dict[str, Any]:
        _reject_blocked_device(path)
        p = Path(path)
        if not p.exists():
            raise ToolError(f"not found: {path}")
        size = p.stat().st_size
        truncated_at_bytes: int | None = None
        if size > _MAX_BYTES:
            with p.open("rb") as fh:
                raw = fh.read(_MAX_BYTES)
            truncated_at_bytes = _MAX_BYTES
        else:
            raw = p.read_bytes()
        try:
            content = _decode_with_bom(raw)
        except UnicodeDecodeError as exc:
            raise ToolError(f"not UTF-8: {exc}") from exc

        all_lines = content.splitlines(keepends=True)
        total_lines = len(all_lines)

        if offset >= total_lines:
            return {
                "content": "",
                "lines": 0,
                "total_lines": total_lines,
                "offset": offset,
                "limit": limit,
                "partial": True,
                "truncated_at_bytes": truncated_at_bytes,
            }

        end = offset + limit if limit is not None else None
        sliced = all_lines[offset:end]
        joined = "".join(sliced)

        _validate_content_tokens(joined)

        partial = (
            truncated_at_bytes is not None
            or offset > 0
            or (limit is not None and offset + limit < total_lines)
        )
        return {
            "content": joined,
            "lines": len(sliced),
            "total_lines": total_lines,
            "offset": offset,
            "limit": limit,
            "partial": partial,
            "truncated_at_bytes": truncated_at_bytes,
        }


read_file: ReadFile = ReadFile()
