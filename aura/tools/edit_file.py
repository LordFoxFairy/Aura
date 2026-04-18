"""Edit a file by string replacement."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from aura.tools.base import AuraTool, ToolResult, build_tool


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


def _edit(params: BaseModel) -> ToolResult:
    assert isinstance(params, EditFileParams)
    p = Path(params.path).expanduser()
    if not p.exists():
        return ToolResult(ok=False, error=f"not found: {params.path}")
    if not p.is_file():
        return ToolResult(ok=False, error=f"not a file: {params.path}")
    if not params.old_str:
        return ToolResult(ok=False, error="old_str must be non-empty")

    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return ToolResult(ok=False, error=f"not UTF-8: {exc}")

    occurrences = content.count(params.old_str)
    if occurrences == 0:
        return ToolResult(ok=False, error=f"old_str not found in {params.path}")
    if occurrences > 1 and not params.replace_all:
        return ToolResult(
            ok=False,
            error=(
                f"old_str matches {occurrences} times; set replace_all=True "
                "or narrow old_str to a unique region"
            ),
        )

    new_content = (
        content.replace(params.old_str, params.new_str)
        if params.replace_all
        else content.replace(params.old_str, params.new_str, 1)
    )
    p.write_text(new_content, encoding="utf-8")
    return ToolResult(
        ok=True,
        output={"replacements": occurrences if params.replace_all else 1},
    )


edit_file: AuraTool = build_tool(
    name="edit_file",
    description=(
        "Edit a file by string replacement. Finds old_str (which must be unique unless "
        "replace_all=True) and replaces it with new_str. Fails loudly on 0 or "
        "ambiguous matches."
    ),
    input_model=EditFileParams,
    call=_edit,
    is_destructive=True,
)
