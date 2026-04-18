"""Edit a file by string replacement."""

from __future__ import annotations

import asyncio
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


def _edit_sync(
    path: str, old_str: str, new_str: str, replace_all: bool,
) -> ToolResult:
    p = Path(path).expanduser()
    if not p.exists():
        return ToolResult(ok=False, error=f"not found: {path}")
    if not p.is_file():
        return ToolResult(ok=False, error=f"not a file: {path}")
    if not old_str:
        return ToolResult(ok=False, error="old_str must be non-empty")

    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return ToolResult(ok=False, error=f"not UTF-8: {exc}")

    occurrences = content.count(old_str)
    if occurrences == 0:
        return ToolResult(ok=False, error=f"old_str not found in {path}")
    if occurrences > 1 and not replace_all:
        return ToolResult(
            ok=False,
            error=(
                f"old_str matches {occurrences} times; set replace_all=True "
                "or narrow old_str to a unique region"
            ),
        )

    new_content = (
        content.replace(old_str, new_str)
        if replace_all
        else content.replace(old_str, new_str, 1)
    )
    p.write_text(new_content, encoding="utf-8")
    return ToolResult(ok=True, output={"replacements": occurrences if replace_all else 1})


async def _acall(params: BaseModel) -> ToolResult:
    assert isinstance(params, EditFileParams)
    return await asyncio.to_thread(
        _edit_sync,
        params.path, params.old_str, params.new_str, params.replace_all,
    )


edit_file: AuraTool = build_tool(
    name="edit_file",
    description=(
        "Edit a file by string replacement. Finds old_str (which must be unique unless "
        "replace_all=True) and replaces it with new_str. Fails loudly on 0 or "
        "ambiguous matches."
    ),
    input_model=EditFileParams,
    call=_acall,
    is_destructive=True,
)
