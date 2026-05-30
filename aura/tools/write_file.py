"""write_file — create or overwrite UTF-8 text files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel, Field

from aura.domain.permission.matchers import path_prefix_on
from aura.domain.tool import ToolError, ToolMetadata, ValidationResult
from aura.tools.base import Tool


class WriteFileParams(BaseModel):
    path: str = Field(
        description="File path. Missing parent directories are created automatically.",
    )
    content: str = Field(description="UTF-8 text content to write. Overwrites any existing file.")


class WriteFileResult(TypedDict):
    written: int


def _preview(args: dict[str, Any]) -> str:
    return f"path: {args.get('path', '')}  ({len(args.get('content', ''))} chars)"


class WriteFile(Tool):
    name: str = "write_file"
    description: str = (
        "Create or overwrite a UTF-8 text file. "
        "Missing parent directories are created automatically."
    )
    args_schema: type[BaseModel] = WriteFileParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=True,
        is_concurrency_safe=False,
        rule_matcher=path_prefix_on("path"),
        args_preview=_preview,
        timeout_sec=None,
    )

    def validate_input(self, args: dict[str, Any]) -> ValidationResult:
        path = args.get("path", "")
        if not isinstance(path, str) or path == "":
            return ValidationResult(
                invalid=True,
                reason="write_file requires a non-empty path",
            )
        return ValidationResult(invalid=False)

    def _run(self, path: str, content: str) -> WriteFileResult:
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
