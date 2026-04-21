"""Tests for aura.tools.write_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.schemas.tool import ToolError
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


async def test_write_file_creates_missing_parent_dir(tmp_path: Path) -> None:
    missing_parent = tmp_path / "nonexistent_dir"
    target = missing_parent / "file.txt"
    out = await write_file.ainvoke({"path": str(target), "content": "data"})
    assert out == {"written": 4}
    assert missing_parent.is_dir()
    assert target.read_text(encoding="utf-8") == "data"


async def test_write_file_creates_nested_missing_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c" / "deep.txt"
    out = await write_file.ainvoke({"path": str(target), "content": "deep"})
    assert out == {"written": 4}
    assert (tmp_path / "a").is_dir()
    assert (tmp_path / "a" / "b").is_dir()
    assert (tmp_path / "a" / "b" / "c").is_dir()
    assert target.read_text(encoding="utf-8") == "deep"


async def test_write_file_auto_create_idempotent_on_existing_parent(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    out = await write_file.ainvoke({"path": str(target), "content": "ok"})
    assert out == {"written": 2}
    assert target.read_text(encoding="utf-8") == "ok"


async def test_write_file_mkdir_permission_error_reported(tmp_path: Path) -> None:
    import os
    import sys

    if sys.platform == "win32" or os.geteuid() == 0:  # pragma: no cover
        pytest.skip("chmod-based permission test not reliable as root or on Windows")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        target = locked / "newdir" / "file.txt"
        with pytest.raises(ToolError, match="cannot create parent dir"):
            await write_file.ainvoke({"path": str(target), "content": "data"})
    finally:
        locked.chmod(0o700)


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


def test_write_file_metadata_includes_matcher_and_preview() -> None:
    from aura.tools.write_file import write_file

    meta = write_file.metadata or {}
    assert meta.get("rule_matcher") is not None
    preview = meta.get("args_preview")
    assert callable(preview)
    assert preview({"path": "x.md", "content": "hello"}) == "path: x.md  (5 chars)"
