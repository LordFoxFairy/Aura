"""Tests for aura.tools.read_file."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aura.schemas.tool import ToolError
from aura.tools.read_file import ReadFileParams, read_file


async def test_read_file_utf8_success(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello\nworld\n", encoding="utf-8")
    out = await read_file.ainvoke({"path": str(f)})
    assert out["content"] == "hello\nworld\n"
    assert out["lines"] == 2


async def test_read_file_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    out = await read_file.ainvoke({"path": str(f)})
    assert out["content"] == ""
    assert out["lines"] == 0


async def test_read_file_single_line_no_newline(tmp_path: Path) -> None:
    f = tmp_path / "single.txt"
    f.write_text("hello", encoding="utf-8")
    out = await read_file.ainvoke({"path": str(f)})
    assert out["lines"] == 1


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


def test_read_file_is_read_only() -> None:
    meta = read_file.metadata or {}
    assert meta.get("is_read_only") is True
    assert meta.get("is_destructive") is False
    assert meta.get("is_concurrency_safe") is True


def test_read_file_metadata_includes_matcher_and_preview() -> None:
    meta = read_file.metadata or {}
    matcher = meta.get("rule_matcher")
    assert callable(matcher)
    # Path-prefix matcher: /tmp covers /tmp/foo but not /tmpfoo.
    assert matcher({"path": "/tmp/foo"}, "/tmp") is True
    assert matcher({"path": "/tmpfoo"}, "/tmp") is False

    preview = meta.get("args_preview")
    assert callable(preview)
    assert preview({"path": "/tmp/a"}) == "path: /tmp/a"


# ---------------------------------------------------------------------------
# offset / limit / partial semantics — mirrors claude-code FileReadTool
# ---------------------------------------------------------------------------


def _five_line_file(tmp_path: Path, name: str = "lines.txt") -> Path:
    f = tmp_path / name
    f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    return f


async def test_read_with_offset_skips_early_lines(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    out = await read_file.ainvoke({"path": str(f), "offset": 2})
    assert out["content"] == "c\nd\ne\n"
    assert out["lines"] == 3
    assert out["total_lines"] == 5
    assert out["offset"] == 2
    assert out["limit"] is None
    assert out["partial"] is True


async def test_read_with_limit_caps_lines(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    out = await read_file.ainvoke({"path": str(f), "limit": 2})
    assert out["content"] == "a\nb\n"
    assert out["lines"] == 2
    assert out["total_lines"] == 5
    assert out["offset"] == 0
    assert out["limit"] == 2
    assert out["partial"] is True


async def test_read_with_offset_and_limit_returns_middle_slice(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    out = await read_file.ainvoke({"path": str(f), "offset": 1, "limit": 2})
    assert out["content"] == "b\nc\n"
    assert out["lines"] == 2
    assert out["total_lines"] == 5
    assert out["offset"] == 1
    assert out["limit"] == 2
    assert out["partial"] is True


async def test_read_full_file_reports_partial_false(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    # Explicit offset=0, limit=None — full read, partial must be False.
    out = await read_file.ainvoke({"path": str(f)})
    assert out["total_lines"] == 5
    assert out["lines"] == 5
    assert out["partial"] is False

    # And: limit large enough to cover the file also counts as non-partial.
    out2 = await read_file.ainvoke({"path": str(f), "limit": 99})
    assert out2["partial"] is False
    assert out2["lines"] == 5


async def test_read_offset_beyond_file_returns_empty(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    out = await read_file.ainvoke({"path": str(f), "offset": 99})
    assert out["content"] == ""
    assert out["lines"] == 0
    assert out["total_lines"] == 5
    assert out["offset"] == 99
    # Honest about over-shooting.
    assert out["partial"] is True


def test_offset_negative_rejected_by_schema() -> None:
    with pytest.raises(ValidationError):
        ReadFileParams.model_validate({"path": "/tmp/x", "offset": -1})


def test_limit_zero_rejected_by_schema() -> None:
    with pytest.raises(ValidationError):
        ReadFileParams.model_validate({"path": "/tmp/x", "limit": 0})


async def test_total_lines_reflects_file_not_slice(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    out = await read_file.ainvoke({"path": str(f), "offset": 3, "limit": 1})
    # Slice has 1 line, but total_lines is the whole file.
    assert out["lines"] == 1
    assert out["total_lines"] == 5
