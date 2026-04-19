"""Tests for aura.tools.read_file — read_file singleton."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.tools.base import ToolError
from aura.tools.read_file import read_file

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_read_file_utf8_success(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello\nworld\n", encoding="utf-8")
    out = await read_file.ainvoke({"path": str(f)})
    assert out == {"content": "hello\nworld\n", "lines": 2}


async def test_read_file_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    out = await read_file.ainvoke({"path": str(f)})
    assert out == {"content": "", "lines": 0}


async def test_read_file_single_line_no_newline(tmp_path: Path) -> None:
    f = tmp_path / "single.txt"
    f.write_text("hello", encoding="utf-8")
    out = await read_file.ainvoke({"path": str(f)})
    assert out["lines"] == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_read_file_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="not found"):
        await read_file.ainvoke({"path": str(tmp_path / "nope.txt")})


async def test_read_file_too_large(tmp_path: Path) -> None:
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(ToolError, match="too large"):
        await read_file.ainvoke({"path": str(f)})


async def test_read_file_invalid_utf8(tmp_path: Path) -> None:
    f = tmp_path / "bad.bin"
    f.write_bytes(b"\xff\xfe\x00\x00")
    with pytest.raises(ToolError, match="UTF-8"):
        await read_file.ainvoke({"path": str(f)})


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_read_file_is_read_only() -> None:
    meta = read_file.metadata or {}
    assert meta.get("is_read_only") is True
    assert meta.get("is_destructive") is False
    assert meta.get("is_concurrency_safe") is True
