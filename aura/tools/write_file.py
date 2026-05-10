"""write_file tool — create or overwrite UTF-8 text files.

Round 3B (F-02-007) — must-read-first staleness gate. The actual mtime
check that rejects writes when the on-disk file is newer than the
parent's last read lives in :mod:`aura.core.hooks.must_read_first`,
not here: the tool itself is dumb-by-design (open + write + return),
which keeps the no-permission-bypass invariant trivially auditable.
The hook fires BEFORE this tool runs and short-circuits the call
whenever the read fingerprint is missing or stale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aura.core.permissions.matchers import path_prefix_on
from aura.schemas.tool import ToolError, ToolMetadata, ValidationResult
from aura.tools.base import Tool


class WriteFileParams(BaseModel):
    path: str = Field(
        description="File path. Missing parent directories are created automatically.",
    )
    content: str = Field(description="UTF-8 text content to write. Overwrites any existing file.")


def _preview(args: dict[str, Any]) -> str:
    return f"path: {args.get('path', '')}  ({len(args.get('content', ''))} chars)"


class WriteFile(Tool):
    name: str = "write_file"
    description: str = (
        "Create or overwrite a UTF-8 text file. "
        "Missing parent directories are created automatically."
    )
    args_schema: type[BaseModel] = WriteFileParams
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=True,
        is_concurrency_safe=False,
        rule_matcher=path_prefix_on("path"),
        args_preview=_preview,
        timeout_sec=None,
    )

    def validate_input(self, args: dict[str, Any]) -> ValidationResult:
        """Reject writes with structurally unusable args.

        Phase 5 Task 2 — args-only check. ``is_dir`` / ``mkdir``
        rejections need filesystem state and stay in ``_run``. An
        empty path is a pure args-shape problem and surfaces here.
        """
        path = args.get("path", "")
        if not isinstance(path, str) or path == "":
            return ValidationResult(
                invalid=True,
                reason="write_file requires a non-empty path",
            )
        return ValidationResult(invalid=False)

    def _run(self, path: str, content: str) -> dict[str, Any]:
        p = Path(path)
        if p.is_dir():
            raise ToolError(f"path is a directory: {path}")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolError(f"cannot create parent dir: {exc}") from exc
        data = content.encode("utf-8")
        p.write_bytes(data)
        return {"written": len(data)}


write_file: WriteFile = WriteFile()
