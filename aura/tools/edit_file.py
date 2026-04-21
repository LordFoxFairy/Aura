"""Edit a file by string replacement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.core.permissions.matchers import path_prefix_on
from aura.schemas.tool import ToolError, tool_metadata


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


def _preview(args: dict[str, Any]) -> str:
    new_lines = len(args.get("new_str", "").splitlines())
    old_lines = len(args.get("old_str", "").splitlines())
    return f"path: {args.get('path', '')}  +{new_lines}/-{old_lines} lines"


class EditFile(BaseTool):
    name: str = "edit_file"
    description: str = (
        "Edit a file by string replacement. Finds old_str (which must be unique unless "
        "replace_all=True) and replaces it with new_str. Fails loudly on 0 or "
        "ambiguous matches."
    )
    args_schema: type[BaseModel] = EditFileParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_destructive=True,
        rule_matcher=path_prefix_on("path"),
        args_preview=_preview,
    )

    def _run(
        self, path: str, old_str: str, new_str: str, replace_all: bool = False,
    ) -> dict[str, Any]:
        p = Path(path).expanduser()

        # --- New-file branch (mirrors claude-code FileEditTool.ts:226-227).
        # Only reachable through direct SDK ainvoke for non-existent paths.
        # Via the agent loop, the must-read-first hook has a matching
        # bypass for ``old_str == "" and not path.exists()``; any other
        # shape stays gated, so write_file remains the canonical
        # agent-driven creation surface.
        if not p.exists():
            if old_str == "":
                # New file: no pre-existing line-ending style to preserve;
                # write LF as-is (bytes to avoid universal-newline translation).
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

        # Read bytes (not ``read_text``) so universal-newline translation
        # does not erase the original line-ending signal before we detect it.
        try:
            raw = p.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"not UTF-8: {exc}") from exc

        # Detect original line-ending style BEFORE normalization.
        if "\r\n" in raw:
            original_newline = "\r\n"
        elif "\r" in raw and "\n" not in raw:
            original_newline = "\r"
        else:
            original_newline = "\n"

        # Normalize read content to LF for matching (two passes handles
        # CRLF -> LF, then any bare CR -> LF for old-Mac files).
        content = raw.replace("\r\n", "\n").replace("\r", "\n")
        # Defensive: LLM-supplied args may carry CRLF of their own.
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

        # Restore original line-ending style on write.
        if original_newline != "\n":
            new_content = new_content.replace("\n", original_newline)
        # Write as bytes to bypass universal-newline translation on write,
        # which would otherwise convert \n -> os.linesep on Windows.
        p.write_bytes(new_content.encode("utf-8"))
        return {"replacements": occurrences if replace_all else 1}


edit_file: BaseTool = EditFile()
