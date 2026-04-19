"""Tests for aura.tools.grep."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.tools.base import ToolError
from aura.tools.grep import grep


async def test_grep_finds_single_match(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello world\nno match here\n", encoding="utf-8")
    out = await grep.ainvoke({"pattern": "hello", "path": str(tmp_path)})
    assert out["count"] == 1
    assert out["matches"][0]["line_num"] == 1
    assert out["matches"][0]["line"] == "hello world"
    assert out["truncated"] is False


async def test_grep_returns_multiple_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("foo bar\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("foo baz\n", encoding="utf-8")
    out = await grep.ainvoke({"pattern": "foo", "path": str(tmp_path)})
    assert out["count"] == 2


async def test_grep_respects_max_results(tmp_path: Path) -> None:
    f = tmp_path / "many.txt"
    f.write_text("\n".join(["match"] * 10), encoding="utf-8")
    out = await grep.ainvoke(
        {"pattern": "match", "path": str(tmp_path), "max_results": 3}
    )
    assert out["count"] == 3
    assert out["truncated"] is True


async def test_grep_glob_filter(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("import os\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("import something\n", encoding="utf-8")
    out = await grep.ainvoke(
        {"pattern": "import", "path": str(tmp_path), "glob": "*.py"}
    )
    assert out["count"] == 1
    assert out["matches"][0]["file"].endswith("code.py")


async def test_grep_invalid_regex_returns_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="invalid regex"):
        await grep.ainvoke({"pattern": "[", "path": str(tmp_path)})


async def test_grep_missing_path_returns_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="not found"):
        await grep.ainvoke({"pattern": "x", "path": str(tmp_path / "nonexistent")})


async def test_grep_non_utf8_files_skipped(tmp_path: Path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\x80\x81\x82\x83")
    (tmp_path / "text.txt").write_text("hello\n", encoding="utf-8")
    out = await grep.ainvoke({"pattern": "hello", "path": str(tmp_path)})
    assert out["count"] == 1


def test_grep_capability_flags() -> None:
    meta = grep.metadata or {}
    assert meta.get("is_read_only") is True
    assert meta.get("is_concurrency_safe") is True
    assert meta.get("is_destructive") is False
