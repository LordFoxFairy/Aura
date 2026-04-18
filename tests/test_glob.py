"""Tests for aura.tools.glob — glob singleton."""

from __future__ import annotations

from pathlib import Path

from aura.tools.base import AuraTool
from aura.tools.glob import GlobParams, glob


async def test_glob_finds_py_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "c.txt").write_text("", encoding="utf-8")
    result = await glob.acall(GlobParams(pattern="*.py", path=str(tmp_path)))
    assert result.ok is True
    assert result.output is not None
    assert result.output["count"] == 2
    files = result.output["files"]
    assert all(f.endswith(".py") for f in files)


async def test_glob_recursive_double_star(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "top.py").write_text("", encoding="utf-8")
    (sub / "nested.py").write_text("", encoding="utf-8")
    result = await glob.acall(GlobParams(pattern="**/*.py", path=str(tmp_path)))
    assert result.ok is True
    assert result.output is not None
    assert result.output["count"] == 2


async def test_glob_respects_max_results(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("", encoding="utf-8")
    result = await glob.acall(GlobParams(pattern="*.py", path=str(tmp_path), max_results=3))
    assert result.ok is True
    assert result.output is not None
    assert result.output["count"] == 3
    assert result.output["truncated"] is True


async def test_glob_missing_path_returns_error(tmp_path: Path) -> None:
    result = await glob.acall(GlobParams(pattern="*.py", path=str(tmp_path / "nope")))
    assert result.ok is False
    assert result.error is not None
    assert "not found" in result.error


async def test_glob_empty_result_ok(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    result = await glob.acall(GlobParams(pattern="*.py", path=str(tmp_path)))
    assert result.ok is True
    assert result.output is not None
    assert result.output["count"] == 0
    assert result.output["files"] == []
    assert result.output["truncated"] is False


async def test_glob_returns_relative_paths(tmp_path: Path) -> None:
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("", encoding="utf-8")
    result = await glob.acall(GlobParams(pattern="**/*.py", path=str(tmp_path)))
    assert result.ok is True
    assert result.output is not None
    assert len(result.output["files"]) == 1
    assert not result.output["files"][0].startswith("/")


def test_glob_capability_flags() -> None:
    assert glob.is_read_only is True
    assert glob.is_concurrency_safe is True
    assert glob.is_destructive is False
    assert isinstance(glob, AuraTool)
