"""Tests for aura.tools.write_file — WriteFileTool."""
from __future__ import annotations

from pathlib import Path

import pytest

from aura.tools.base import AuraTool
from aura.tools.write_file import WriteFileParams, WriteFileTool


@pytest.fixture()
def tool() -> WriteFileTool:
    return WriteFileTool()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_write_file_creates_new_file(tmp_path: Path, tool: WriteFileTool) -> None:
    target = tmp_path / "new.txt"
    result = await tool.acall(WriteFileParams(path=str(target), content="hello"))
    assert result.ok is True
    assert result.output == {"written": 5}
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello"


async def test_write_file_overwrites_existing(
    tmp_path: Path, tool: WriteFileTool
) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old", encoding="utf-8")
    result = await tool.acall(WriteFileParams(path=str(target), content="new content"))
    assert result.ok is True
    assert result.output == {"written": 11}
    assert target.read_text(encoding="utf-8") == "new content"


async def test_write_file_utf8_multibyte(tmp_path: Path, tool: WriteFileTool) -> None:
    content = "héllo 世界"
    expected_bytes = len(content.encode("utf-8"))
    target = tmp_path / "multibyte.txt"
    result = await tool.acall(WriteFileParams(path=str(target), content=content))
    assert result.ok is True
    assert result.output == {"written": expected_bytes}
    assert target.read_text(encoding="utf-8") == content


async def test_write_file_empty_content(tmp_path: Path, tool: WriteFileTool) -> None:
    target = tmp_path / "empty.txt"
    result = await tool.acall(WriteFileParams(path=str(target), content=""))
    assert result.ok is True
    assert result.output == {"written": 0}
    assert target.exists()
    assert target.stat().st_size == 0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_write_file_missing_parent_dir_fails(
    tmp_path: Path, tool: WriteFileTool
) -> None:
    target = tmp_path / "nonexistent_dir" / "file.txt"
    result = await tool.acall(WriteFileParams(path=str(target), content="data"))
    assert result.ok is False
    assert result.error is not None
    assert "parent directory" in result.error
    assert not target.exists()


async def test_write_file_does_not_auto_create_parent(
    tmp_path: Path, tool: WriteFileTool
) -> None:
    missing_parent = tmp_path / "nonexistent_dir"
    target = missing_parent / "file.txt"
    await tool.acall(WriteFileParams(path=str(target), content="data"))
    assert not missing_parent.exists()


async def test_write_file_path_is_directory_fails(
    tmp_path: Path, tool: WriteFileTool
) -> None:
    result = await tool.acall(WriteFileParams(path=str(tmp_path), content="data"))
    assert result.ok is False
    assert result.error is not None
    assert "directory" in result.error


# ---------------------------------------------------------------------------
# Capability flags / protocol
# ---------------------------------------------------------------------------


def test_write_file_capability_flags(tool: WriteFileTool) -> None:
    assert tool.is_read_only is False
    assert tool.is_destructive is True
    assert tool.is_concurrency_safe is False


def test_write_file_no_check_permissions_method(tool: WriteFileTool) -> None:
    assert not hasattr(tool, "check_permissions")


def test_write_file_satisfies_protocol(tool: WriteFileTool) -> None:
    assert isinstance(tool, AuraTool) is True
