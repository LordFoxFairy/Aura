"""Tests for aura.tools.write_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.tools.base import ToolError
from aura.tools.write_file import write_file


async def test_write_file_creates_new_file(tmp_path: Path) -> None:
    target = tmp_path / "new.txt"
    out = await write_file.ainvoke({"path": str(target), "content": "hello"})
    assert out == {"written": 5}
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello"


async def test_write_file_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old", encoding="utf-8")
    out = await write_file.ainvoke({"path": str(target), "content": "new content"})
    assert out == {"written": 11}
    assert target.read_text(encoding="utf-8") == "new content"


async def test_write_file_utf8_multibyte(tmp_path: Path) -> None:
    content = "héllo 世界"
    expected_bytes = len(content.encode("utf-8"))
    target = tmp_path / "multibyte.txt"
    out = await write_file.ainvoke({"path": str(target), "content": content})
    assert out == {"written": expected_bytes}
    assert target.read_text(encoding="utf-8") == content


async def test_write_file_empty_content(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    out = await write_file.ainvoke({"path": str(target), "content": ""})
    assert out == {"written": 0}
    assert target.exists()
    assert target.stat().st_size == 0


async def test_write_file_missing_parent_dir_fails(tmp_path: Path) -> None:
    target = tmp_path / "nonexistent_dir" / "file.txt"
    with pytest.raises(ToolError, match="parent directory"):
        await write_file.ainvoke({"path": str(target), "content": "data"})
    assert not target.exists()


async def test_write_file_does_not_auto_create_parent(tmp_path: Path) -> None:
    missing_parent = tmp_path / "nonexistent_dir"
    target = missing_parent / "file.txt"
    with pytest.raises(ToolError):
        await write_file.ainvoke({"path": str(target), "content": "data"})
    assert not missing_parent.exists()


async def test_write_file_path_is_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="directory"):
        await write_file.ainvoke({"path": str(tmp_path), "content": "data"})


def test_write_file_capability_flags() -> None:
    meta = write_file.metadata or {}
    assert meta.get("is_read_only") is False
    assert meta.get("is_destructive") is True
    assert meta.get("is_concurrency_safe") is False


def test_write_file_no_check_permissions_method() -> None:
    assert not hasattr(write_file, "check_permissions")
