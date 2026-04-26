"""read_file tool — UTF-8 read with 1 MB cap + offset/limit slicing.

Mirrors claude-code FileReadTool: `offset` (0-indexed line start in Aura —
claude-code uses 1-indexed; we normalize to 0 for parity with Python slice
semantics) and `limit` (max lines). `partial` in the return dict flips true
when the caller saw less than the full file; the must-read-first invariant
uses it to reject edit_file after a sliced read.

Round 1C — device path block. Reading from ``/dev/stdin`` /
``/dev/random`` etc. either hangs the process forever (stdin / tty),
returns garbage that wastes the LLM's context (zero / random), or
exposes kernel memory (``/proc/kcore``). Reject the closed set so the
LLM can't accidentally tunnel into one even if the path is constructed
indirectly.

Round 3B — token budget. A perfectly valid 1 MB UTF-8 text file is
still ~250k tokens — enough to evict every other context message. Cap
at :data:`_TOKEN_BUDGET` and reject (rather than silently truncate) so
the LLM sees a clear error and learns to slice.

F-02-003 — head-truncation at the byte cap. Files exceeding
:data:`_MAX_BYTES` are read up to the cap (head bytes), with
``partial=True`` and ``truncated_at_bytes`` set on the result, instead
of being rejected outright. The token-budget check above still fires on
the truncated content, so oversized text files surface as a clear
"slice with offset/limit" error rather than a silent partial.

F-02-005 — BOM-aware decode. We sniff the first 2-3 bytes for UTF-16
(LE/BE) and UTF-8 BOMs and decode accordingly; UTF-8 BOM is stripped
from the returned string. No BOM ⇒ plain UTF-8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.core.permissions.matchers import path_prefix_on
from aura.schemas.tool import ToolError, tool_metadata

_MAX_BYTES = 1024 * 1024

# Round 3B token budget. 25k tokens is generous (~100 KB of text) but
# still well within a single-turn context. The standard char/token
# heuristic (4 chars per token) is a low-end estimate — actual
# tokenisers vary, but biased-low here means we reject only when the
# file is unambiguously oversized rather than borderline.
_TOKEN_BUDGET = 25_000
_CHARS_PER_TOKEN_HEURISTIC = 4

# Round 1C — closed set of paths the tool always refuses, irrespective
# of permission rules. Three categories:
#   1. Interactive endpoints (``/dev/stdin``, ``/dev/tty``, ``/dev/console``)
#      — read() blocks forever on these in a non-interactive process.
#   2. Pseudo-files that produce useless input (``/dev/zero``,
#      ``/dev/random``, ``/dev/urandom``, ``/dev/full``) — would just
#      burn the tool's 1 MB cap on garbage / never return.
#   3. Kernel memory (``/proc/kcore``, ``/proc/kmem``) — sensitive
#      contents, also typically unreadable as user but worth listing
#      so a SUID context doesn't accidentally exfil.
# ``/dev/fd/*`` and ``/proc/self/fd/*`` cover the "stdin via /dev/fd"
# trick that bypasses a literal ``/dev/stdin`` allowlist.
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


def _reject_blocked_device(path: str) -> None:
    """Refuse to read paths that point at non-file kernel surfaces.

    Resolves the path WITHOUT requiring it to exist (``strict=False``)
    so a missing file gets the existing "not found" diagnostic from the
    main read path rather than this device-block error. Comparing the
    resolved string against the closed set catches both literal paths
    and symlink chains that point at the same target.
    """
    try:
        resolved = str(Path(path).resolve(strict=False))
    except (OSError, RuntimeError):
        # Resolution itself failed (cycle, permission). Fall through —
        # the main read will surface the OS error verbatim.
        return
    if resolved in _BLOCKED_DEVICE_PATHS:
        raise ToolError(
            f"refusing to read {resolved!r} — kernel/interactive device "
            "endpoint (would block, return garbage, or expose kernel memory)",
        )


def _decode_with_bom(data: bytes) -> str:
    """Decode bytes with BOM-aware encoding selection.

    UTF-16 LE/BE BOMs ⇒ decode via the ``utf-16`` codec, which honours
    the leading BOM and strips it from the result. UTF-8 BOM ⇒ skip the
    3-byte BOM and decode as utf-8. No BOM ⇒ plain utf-8.
    """
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8")
    return data.decode("utf-8")


def _validate_content_tokens(content: str) -> None:
    """Reject reads that would blow the token budget.

    Applied to BOTH the full-file read and any sliced (offset+limit)
    read — a slice is still a slice, but if the user requested 30k
    lines via ``limit=`` we should still refuse rather than truncate
    silently. The error message tells the LLM to slice; the standard
    self-correction is ``read_file(path, offset=, limit=)``.
    """
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


class ReadFile(BaseTool):
    name: str = "read_file"
    description: str = (
        "Read a text file (UTF-8 / UTF-16 LE / UTF-16 BE / UTF-8-BOM) with "
        "optional offset/limit. Files exceeding 1 MB are head-truncated to "
        "the cap with partial=True and truncated_at_bytes set."
    )
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
        # Round 1C — block the kernel/interactive device set BEFORE the
        # filesystem touch. Catches direct paths and symlink chains;
        # missing files fall through to the main read path's "not found".
        _reject_blocked_device(path)
        p = Path(path)
        if not p.exists():
            raise ToolError(f"not found: {path}")
        size = p.stat().st_size
        # F-02-003 — head-truncate at the byte cap (instead of rejecting),
        # surface partial=True + truncated_at_bytes so the caller can detect
        # the cut. Read bytes (not text) so BOM sniffing happens before
        # decode.
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
                "truncated_at_bytes": truncated_at_bytes,
            }

        end = offset + limit if limit is not None else None
        sliced = all_lines[offset:end]
        joined = "".join(sliced)

        # Round 3B — apply the token budget to the slice we're about to
        # return (covers both full reads and oversized slices).
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


read_file: BaseTool = ReadFile()
