"""Edit a file by string replacement."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel, Field

from aura.domain.permission.matchers import path_prefix_on
from aura.domain.tool import ToolError, ToolMetadata, ValidationResult
from aura.tools.base import Tool

_MAX_EDIT_SIZE = 256 * 1024 * 1024


class EditFileParams(BaseModel):
    path: str = Field(description="Path to the file to edit.")
    old_str: str = Field(
        description=(
            "Exact string to find. If it matches more than once, replace_all=True is "
            "required, otherwise the edit fails to avoid ambiguity."
        ),
    )
    new_str: str = Field(description="Replacement string. Can be empty to delete old_str.")
    replace_all: bool = Field(
        default=False,
        description=(
            "If True, replace every occurrence. "
            "If False (default), exactly one match required."
        ),
    )


class EditFileResult(TypedDict):
    replacements: int
    # Present only when the edit created a previously-missing file.
    created: NotRequired[bool]


def _preview(args: dict[str, Any]) -> str:
    new_lines = len(args.get("new_str", "").splitlines())
    old_lines = len(args.get("old_str", "").splitlines())
    return f"path: {args.get('path', '')}  +{new_lines}/-{old_lines} lines"


class EditFile(Tool):
    name: str = "edit_file"
    description: str = (
        "Edit a file by string replacement. Finds old_str (which must be unique unless "
        "replace_all=True) and replaces it with new_str. Fails loudly on 0 or "
        "ambiguous matches."
    )
    args_schema: type[BaseModel] = EditFileParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
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
                reason="edit_file requires a non-empty path",
            )
        return ValidationResult(invalid=False)

    def _run(
        self, path: str, old_str: str, new_str: str, replace_all: bool = False,
    ) -> dict[str, Any]:
        p = Path(path).expanduser()

        if not p.exists():
            if old_str == "":
                p.write_bytes(new_str.encode("utf-8"))
                return {"replacements": 1, "created": True}
            raise ToolError(f"not found: {path}")
        if not p.is_file():
            raise ToolError(f"not a file: {path}")
        if old_str == "":
            raise ToolError(
                "cannot edit with empty old_str when file exists; "
                "use a non-empty old_str to identify the region"
            )

        size = p.stat().st_size
        if size > _MAX_EDIT_SIZE:
            raise ToolError(
                f"too large to edit: {size} bytes > {_MAX_EDIT_SIZE} "
                f"({_MAX_EDIT_SIZE // (1024 * 1024)} MB cap)"
            )

        try:
            raw = p.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"not UTF-8: {exc}") from exc

        # Mixed CRLF + bare-LF: preserve as-is — normalizing would corrupt the bare-LF lines.
        crlf_count = raw.count("\r\n")
        bare_lf_count = raw.count("\n") - crlf_count
        has_bare_cr_only = "\r" in raw and "\n" not in raw
        mixed_endings = crlf_count > 0 and bare_lf_count > 0

        if mixed_endings:
            content = raw
            old_str_n = old_str
            new_str_n = new_str
        else:
            if crlf_count > 0:
                original_newline = "\r\n"
            elif has_bare_cr_only:
                original_newline = "\r"
            else:
                original_newline = "\n"
            content = raw.replace("\r\n", "\n").replace("\r", "\n")
            old_str_n = old_str.replace("\r\n", "\n").replace("\r", "\n")
            new_str_n = new_str.replace("\r\n", "\n").replace("\r", "\n")

        occurrences = content.count(old_str_n)
        if occurrences == 0:
            echo = repr(old_str_n)
            if len(echo) > 120:
                echo = echo[:120] + "\u2026"
            raise ToolError(
                f"old_str not found in {path}\n  missing: {echo}"
            )
        if occurrences > 1 and not replace_all:
            raise ToolError(
                f"old_str matches {occurrences} times; set replace_all=True "
                "or narrow old_str to a unique region"
            )

        new_content = (
            content.replace(old_str_n, new_str_n)
            if replace_all
            else content.replace(old_str_n, new_str_n, 1)
        )

        if not mixed_endings and original_newline != "\n":
            new_content = new_content.replace("\n", original_newline)
        p.write_bytes(new_content.encode("utf-8"))
        return {"replacements": occurrences if replace_all else 1}


edit_file: EditFile = EditFile()
