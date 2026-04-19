"""Tests for aura.core.project_memory.load_project_memory (Task 1+2)."""

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


class TestAtImports:
    """Covers B4: `@imports` expansion at load time (depth 5, cycle, fence, etc.)."""

    def test_01_relative_dot_slash_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("before\n@./child.md\nafter")
        (cwd / "child.md").write_text("CHILD-CONTENT")

        result = load_project_memory(cwd)
        assert result == "before\nCHILD-CONTENT\nafter"

    def test_02_tilde_home_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "global.md").write_text("GLOBAL")
        _patch_home(monkeypatch, home)

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("@~/global.md")

        assert load_project_memory(cwd) == "GLOBAL"

    def test_03_absolute_path_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        target = tmp_path / "abs.md"
        target.write_text("ABSOLUTE")

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text(f"@{target}")

        assert load_project_memory(cwd) == "ABSOLUTE"

    def test_04_depth_5_chain_all_expand(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        # a(=AURA.md) → b → c → d → e; e has plain content.
        (cwd / "AURA.md").write_text("A\n@./b.md")
        (cwd / "b.md").write_text("B\n@./c.md")
        (cwd / "c.md").write_text("C\n@./d.md")
        (cwd / "d.md").write_text("D\n@./e.md")
        (cwd / "e.md").write_text("E")

        assert load_project_memory(cwd) == "A\nB\nC\nD\nE"

    def test_05_depth_6_last_link_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        # a → b → c → d → e → f; f should NOT be expanded (line removed from e).
        (cwd / "AURA.md").write_text("A\n@./b.md")
        (cwd / "b.md").write_text("B\n@./c.md")
        (cwd / "c.md").write_text("C\n@./d.md")
        (cwd / "d.md").write_text("D\n@./e.md")
        (cwd / "e.md").write_text("E-pre\n@./f.md\nE-post")
        (cwd / "f.md").write_text("F")

        result = load_project_memory(cwd)
        # e's @./f.md line is dropped; E-pre + E-post remain; F never appears.
        assert "F" not in result
        assert "E-pre" in result and "E-post" in result
        assert "@./f.md" not in result
        assert result == "A\nB\nC\nD\nE-pre\nE-post"

    def test_06_cycle_inner_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("A-pre\n@./b.md\nA-post")
        (cwd / "b.md").write_text("B-pre\n@./AURA.md\nB-post")

        result = load_project_memory(cwd)
        assert result == "A-pre\nB-pre\nB-post\nA-post"

    def test_07_missing_target_line_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("before\n@./missing.md\nafter")

        assert load_project_memory(cwd) == "before\nafter"

    def test_08_directory_target_line_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "subdir").mkdir()
        (cwd / "AURA.md").write_text("before\n@./subdir\nafter")

        assert load_project_memory(cwd) == "before\nafter"

    def test_09_backtick_fence_preserves_literal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "child.md").write_text("EXPANDED")
        (cwd / "AURA.md").write_text("```\n@./child.md\n```")

        result = load_project_memory(cwd)
        assert result == "```\n@./child.md\n```"
        assert "EXPANDED" not in result

    def test_10_tilde_fence_still_expands(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "child.md").write_text("EXPANDED")
        (cwd / "AURA.md").write_text("~~~\n@./child.md\n~~~")

        result = load_project_memory(cwd)
        assert "EXPANDED" in result
        assert "@./child.md" not in result

    def test_11_leading_whitespace_not_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "child.md").write_text("EXPANDED")
        (cwd / "AURA.md").write_text("before\n  @./child.md\nafter")

        result = load_project_memory(cwd)
        assert result == "before\n  @./child.md\nafter"
        assert "EXPANDED" not in result

    def test_12_crlf_line_endings_expand(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "child.md").write_text("CHILD")
        (cwd / "AURA.md").write_bytes(b"before\r\n@./child.md\r\nafter")

        result = load_project_memory(cwd)
        assert "CHILD" in result
        assert "@./child.md" not in result

    def test_13_user_layer_relative_to_home_aura(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".aura").mkdir(parents=True)
        (home / ".aura" / "AURA.md").write_text("U\n@./piece.md")
        (home / ".aura" / "piece.md").write_text("PIECE")
        _patch_home(monkeypatch, home)

        cwd = tmp_path / "project"
        cwd.mkdir()

        assert load_project_memory(cwd) == "U\nPIECE"

    def test_14_relative_to_importing_file_not_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        outer = tmp_path / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        (outer / "AURA.md").write_text("@./child.md")
        (outer / "child.md").write_text("OUTER-CHILD")

        # cwd = inner；但 outer/AURA.md 中 @./child.md 解析相对于 outer/，而非 cwd。
        result = load_project_memory(inner)
        assert "OUTER-CHILD" in result

    def test_15_outer_import_line_leaves_no_residue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "child.md").write_text("CHILD")
        (cwd / "AURA.md").write_text("pre\n@./child.md\npost")

        result = load_project_memory(cwd)
        assert "@./child.md" not in result
        assert result == "pre\nCHILD\npost"
