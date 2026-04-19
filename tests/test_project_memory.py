"""Tests for aura.core.project_memory.load_project_memory (Task 1: discovery only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.core.project_memory import load_project_memory


def _patch_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    # 只打 Path.home()，避免污染进程 $HOME 影响其它测试
    monkeypatch.setattr(Path, "home", lambda: home)


def test_no_aura_md_anywhere_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    cwd = tmp_path / "project"
    cwd.mkdir()

    assert load_project_memory(cwd) == ""


def test_only_user_layer_returns_user_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".aura").mkdir(parents=True)
    (home / ".aura" / "AURA.md").write_text("user-memory")
    _patch_home(monkeypatch, home)

    cwd = tmp_path / "project"
    cwd.mkdir()

    assert load_project_memory(cwd) == "user-memory"


def test_only_project_aura_md_at_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "AURA.md").write_text("project-cwd")

    assert load_project_memory(cwd) == "project-cwd"


def test_aura_md_and_dot_aura_same_dir_both_included_aura_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "AURA.md").write_text("top-level")
    (cwd / ".aura").mkdir()
    (cwd / ".aura" / "AURA.md").write_text("dot-aura")

    assert load_project_memory(cwd) == "top-level\n\ndot-aura"


def test_nested_project_outer_first_inner_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    outer = tmp_path / "x"
    inner = outer / "y"
    inner.mkdir(parents=True)
    (outer / "AURA.md").write_text("outer")
    (inner / "AURA.md").write_text("inner")

    assert load_project_memory(inner) == "outer\n\ninner"


def test_full_stack_canonical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".aura").mkdir(parents=True)
    (home / ".aura" / "AURA.md").write_text("USER")
    _patch_home(monkeypatch, home)

    outer = tmp_path / "x"
    inner = outer / "y"
    inner.mkdir(parents=True)
    (outer / "AURA.md").write_text("PROJECT-OUTER")
    (inner / "AURA.md").write_text("PROJECT-INNER")
    (outer / "AURA.local.md").write_text("LOCAL-OUTER")
    (inner / "AURA.local.md").write_text("LOCAL-INNER")

    result = load_project_memory(inner)
    # User, Project(outer), Project(inner), Local(outer), Local(inner)
    assert result == (
        "USER\n\n"
        "PROJECT-OUTER\n\n"
        "PROJECT-INNER\n\n"
        "LOCAL-OUTER\n\n"
        "LOCAL-INNER"
    )


def test_walk_up_halts_at_filesystem_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 用一个孤立 tmp 目录，向上到根都没有 AURA.md —— 应无错误，空串
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    cwd = tmp_path / "deep" / "nested" / "project"
    cwd.mkdir(parents=True)

    assert load_project_memory(cwd) == ""


def test_aura_md_path_is_directory_is_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    cwd = tmp_path / "project"
    cwd.mkdir()
    # AURA.md 是一个目录而非文件 —— 应跳过，不抛异常
    (cwd / "AURA.md").mkdir()

    assert load_project_memory(cwd) == ""


def test_non_utf8_bytes_decoded_with_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    cwd = tmp_path / "project"
    cwd.mkdir()
    # 写入无效 UTF-8 字节序列
    (cwd / "AURA.md").write_bytes(b"good\xff\xfebad")

    result = load_project_memory(cwd)
    assert "\ufffd" in result  # 替换字符
    assert "good" in result
    assert "bad" in result


def test_symlinked_cwd_walks_resolved_ancestors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    real_outer = tmp_path / "real"
    real_inner = real_outer / "inner"
    real_inner.mkdir(parents=True)
    (real_outer / "AURA.md").write_text("real-outer")
    (real_inner / "AURA.md").write_text("real-inner")

    # 符号链接指向 real_inner
    link = tmp_path / "link_to_inner"
    link.symlink_to(real_inner)

    # 通过符号链接进入 —— resolve 后应看到 real_outer 和 real_inner 层级
    result = load_project_memory(link)
    assert result == "real-outer\n\nreal-inner"
