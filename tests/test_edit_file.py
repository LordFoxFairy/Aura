"""Tests for aura.tools.edit_file — edit_file singleton."""

from __future__ import annotations

from pathlib import Path

from aura.tools.base import AuraTool
from aura.tools.edit_file import EditFileParams, edit_file


async def test_edit_file_single_replacement(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello world\n", encoding="utf-8")
    result = await edit_file.acall(
        EditFileParams(path=str(f), old_str="hello", new_str="goodbye")
    )
    assert result.ok is True
    assert result.output == {"replacements": 1}
    assert f.read_text(encoding="utf-8") == "goodbye world\n"


async def test_edit_file_replace_all(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("foo foo foo\n", encoding="utf-8")
    result = await edit_file.acall(
        EditFileParams(path=str(f), old_str="foo", new_str="bar", replace_all=True)
    )
    assert result.ok is True
    assert result.output == {"replacements": 3}
    assert f.read_text(encoding="utf-8") == "bar bar bar\n"


async def test_edit_file_ambiguous_match_errors(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x x\n", encoding="utf-8")
    result = await edit_file.acall(
        EditFileParams(path=str(f), old_str="x", new_str="y")
    )
    assert result.ok is False
    assert result.error is not None
    assert "2" in result.error
    assert f.read_text(encoding="utf-8") == "x x\n"


async def test_edit_file_not_found_errors(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello\n", encoding="utf-8")
    result = await edit_file.acall(
        EditFileParams(path=str(f), old_str="zzz", new_str="aaa")
    )
    assert result.ok is False
    assert result.error is not None
    assert "not found" in result.error


async def test_edit_file_missing_file_errors(tmp_path: Path) -> None:
    result = await edit_file.acall(
        EditFileParams(path=str(tmp_path / "ghost.txt"), old_str="x", new_str="y")
    )
    assert result.ok is False
    assert result.error is not None
    assert "not found" in result.error


async def test_edit_file_empty_old_str_errors(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello\n", encoding="utf-8")
    result = await edit_file.acall(
        EditFileParams(path=str(f), old_str="", new_str="x")
    )
    assert result.ok is False
    assert result.error is not None
    assert "non-empty" in result.error


async def test_edit_file_delete_via_empty_new_str(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello world\n", encoding="utf-8")
    result = await edit_file.acall(
        EditFileParams(path=str(f), old_str=" world", new_str="")
    )
    assert result.ok is True
    assert f.read_text(encoding="utf-8") == "hello\n"


def test_edit_file_capability_flags() -> None:
    assert edit_file.is_destructive is True
    assert edit_file.is_read_only is False
    assert isinstance(edit_file, AuraTool)
