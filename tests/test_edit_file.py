"""Tests for aura.tools.edit_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.tools.base import ToolError
from aura.tools.edit_file import edit_file


async def test_edit_file_single_replacement(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello world\n", encoding="utf-8")
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "hello", "new_str": "goodbye"}
    )
    assert out == {"replacements": 1}
    assert f.read_text(encoding="utf-8") == "goodbye world\n"


async def test_edit_file_replace_all(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("foo foo foo\n", encoding="utf-8")
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "foo", "new_str": "bar", "replace_all": True}
    )
    assert out == {"replacements": 3}
    assert f.read_text(encoding="utf-8") == "bar bar bar\n"


async def test_edit_file_ambiguous_match_errors(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x x\n", encoding="utf-8")
    with pytest.raises(ToolError, match="2"):
        await edit_file.ainvoke({"path": str(f), "old_str": "x", "new_str": "y"})
    assert f.read_text(encoding="utf-8") == "x x\n"


async def test_edit_file_not_found_errors(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello\n", encoding="utf-8")
    with pytest.raises(ToolError, match="not found"):
        await edit_file.ainvoke({"path": str(f), "old_str": "zzz", "new_str": "aaa"})


async def test_edit_file_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="not found"):
        await edit_file.ainvoke(
            {"path": str(tmp_path / "ghost.txt"), "old_str": "x", "new_str": "y"}
        )


async def test_edit_file_empty_old_str_errors(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello\n", encoding="utf-8")
    with pytest.raises(ToolError, match="non-empty"):
        await edit_file.ainvoke({"path": str(f), "old_str": "", "new_str": "x"})


async def test_edit_file_delete_via_empty_new_str(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello world\n", encoding="utf-8")
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": " world", "new_str": ""}
    )
    assert out == {"replacements": 1}
    assert f.read_text(encoding="utf-8") == "hello\n"


def test_edit_file_capability_flags() -> None:
    meta = edit_file.metadata or {}
    assert meta.get("is_destructive") is True
    assert meta.get("is_read_only") is False
