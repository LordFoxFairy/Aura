"""Tests for aura.tools.glob."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.schemas.tool import ToolError
from aura.tools.glob import glob


async def test_glob_finds_py_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "c.txt").write_text("", encoding="utf-8")
    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert out["count"] == 2
    assert all(f.endswith(".py") for f in out["files"])


async def test_glob_recursive_double_star(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "top.py").write_text("", encoding="utf-8")
    (sub / "nested.py").write_text("", encoding="utf-8")
    out = await glob.ainvoke({"pattern": "**/*.py", "path": str(tmp_path)})
    assert out["count"] == 2


async def test_glob_respects_max_results(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("", encoding="utf-8")
    out = await glob.ainvoke(
        {"pattern": "*.py", "path": str(tmp_path), "max_results": 3}
    )
    assert out["count"] == 3
    assert out["truncated"] is True


async def test_glob_missing_path_returns_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="not found"):
        await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path / "nope")})


async def test_glob_empty_result_ok(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert out["count"] == 0
    assert out["files"] == []
    assert out["truncated"] is False


async def test_glob_returns_relative_paths(tmp_path: Path) -> None:
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("", encoding="utf-8")
    out = await glob.ainvoke({"pattern": "**/*.py", "path": str(tmp_path)})
    assert len(out["files"]) == 1
    assert not out["files"][0].startswith("/")


def test_glob_capability_flags() -> None:
    meta = glob.metadata or {}
    assert meta.get("is_read_only") is True
    assert meta.get("is_concurrency_safe") is True
    assert meta.get("is_destructive") is False
