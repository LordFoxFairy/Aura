"""Edit a file by string replacement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

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


class EditFile(BaseTool):
    name: str = "edit_file"
    description: str = (
        "Edit a file by string replacement. Finds old_str (which must be unique unless "
        "replace_all=True) and replaces it with new_str. Fails loudly on 0 or "
        "ambiguous matches."
    )
    args_schema: type[BaseModel] = EditFileParams
    metadata: dict[str, Any] | None = tool_metadata(is_destructive=True)

    def _run(
        self, path: str, old_str: str, new_str: str, replace_all: bool = False,
    ) -> dict[str, Any]:
        p = Path(path).expanduser()
        if not p.exists():
            raise ToolError(f"not found: {path}")
        if not p.is_file():
            raise ToolError(f"not a file: {path}")
        if not old_str:
            raise ToolError("old_str must be non-empty")

        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"not UTF-8: {exc}") from exc

        occurrences = content.count(old_str)
        if occurrences == 0:
            raise ToolError(f"old_str not found in {path}")
        if occurrences > 1 and not replace_all:
            raise ToolError(
                f"old_str matches {occurrences} times; set replace_all=True "
                "or narrow old_str to a unique region"
            )

        new_content = (
            content.replace(old_str, new_str)
            if replace_all
            else content.replace(old_str, new_str, 1)
        )
        p.write_text(new_content, encoding="utf-8")
        return {"replacements": occurrences if replace_all else 1}


edit_file: BaseTool = EditFile()
