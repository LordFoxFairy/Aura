"""Tests for aura.tools.read_file — ReadFileTool."""
from __future__ import annotations

from pathlib import Path

import pytest

from aura.tools.base import AuraTool
from aura.tools.read_file import ReadFileParams, ReadFileTool


@pytest.fixture()
def tool() -> ReadFileTool:
    return ReadFileTool()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_read_file_utf8_success(tmp_path: Path, tool: ReadFileTool) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello\nworld\n", encoding="utf-8")
    result = await tool.acall(ReadFileParams(path=str(f)))
    assert result.ok is True
    assert result.output == {"content": "hello\nworld\n", "lines": 2}


async def test_read_file_empty_file(tmp_path: Path, tool: ReadFileTool) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    result = await tool.acall(ReadFileParams(path=str(f)))
    assert result.ok is True
    assert result.output == {"content": "", "lines": 0}


async def test_read_file_single_line_no_newline(
    tmp_path: Path, tool: ReadFileTool
) -> None:
    f = tmp_path / "single.txt"
    f.write_text("hello", encoding="utf-8")
    result = await tool.acall(ReadFileParams(path=str(f)))
    assert result.ok is True
    assert result.output is not None
    assert result.output["lines"] == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_read_file_missing_file(tmp_path: Path, tool: ReadFileTool) -> None:
    result = await tool.acall(ReadFileParams(path=str(tmp_path / "nope.txt")))
    assert result.ok is False
    assert result.error is not None
    assert "not found" in result.error


async def test_read_file_too_large(tmp_path: Path, tool: ReadFileTool) -> None:
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * (1024 * 1024 + 1))
    result = await tool.acall(ReadFileParams(path=str(f)))
    assert result.ok is False
    assert result.error is not None
    assert "too large" in result.error


async def test_read_file_invalid_utf8(tmp_path: Path, tool: ReadFileTool) -> None:
    f = tmp_path / "bad.bin"
    f.write_bytes(b"\xff\xfe\x00\x00")
    result = await tool.acall(ReadFileParams(path=str(f)))
    assert result.ok is False
    assert result.error is not None
    assert "UTF-8" in result.error


# ---------------------------------------------------------------------------
# Metadata / protocol
# ---------------------------------------------------------------------------


def test_read_file_is_read_only(tool: ReadFileTool) -> None:
    assert tool.is_read_only is True
    assert tool.is_destructive is False
    assert tool.is_concurrency_safe is True


def test_read_file_satisfies_protocol(tool: ReadFileTool) -> None:
    assert isinstance(tool, AuraTool) is True
