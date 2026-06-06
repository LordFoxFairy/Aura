"""Tests for aura.application.memory.project_memory."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from aura.application.memory.project_memory import (
    clear_cache,
    load_project_memory,
    read_with_imports,
)
from aura.infrastructure.persistence import journal


def _patch_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    # 只打 Path.home()，避免污染进程 $HOME 影响其它测试。
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
    # 隔离 tmp 目录向上到根没有任何 AURA.md —— 应不报错，返回空串。
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
    # AURA.md 是一个目录而非文件 —— 应静默跳过，不抛异常。
    (cwd / "AURA.md").mkdir()

    assert load_project_memory(cwd) == ""


def test_non_utf8_bytes_decoded_with_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    cwd = tmp_path / "project"
    cwd.mkdir()
    # 写入无效 UTF-8 字节序列。
    (cwd / "AURA.md").write_bytes(b"good\xff\xfebad")

    result = load_project_memory(cwd)
    # \ufffd 是 UTF-8 解码失败时的替换字符。
    assert "\ufffd" in result
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

    # 符号链接指向 real_inner。
    link = tmp_path / "link_to_inner"
    link.symlink_to(real_inner)

    # 通过符号链接进入 —— resolve 后应看到 real_outer 和 real_inner 层级。
    result = load_project_memory(link)
    assert result == "real-outer\n\nreal-inner"


class TestAtImports:
    """`@imports` expansion at load time: depth cap, cycle, code fences, etc."""

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
        # Chain: AURA.md → b → c → d → e; e has plain content.
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
        # Chain: AURA.md → b → c → d → e → f; f must NOT be expanded (its
        # @-line is dropped from e once the depth cap is hit).
        (cwd / "AURA.md").write_text("A\n@./b.md")
        (cwd / "b.md").write_text("B\n@./c.md")
        (cwd / "c.md").write_text("C\n@./d.md")
        (cwd / "d.md").write_text("D\n@./e.md")
        (cwd / "e.md").write_text("E-pre\n@./f.md\nE-post")
        (cwd / "f.md").write_text("F")

        result = load_project_memory(cwd)
        # e's @./f.md line is dropped; E-pre and E-post remain; F never appears.
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

        # cwd 是 inner，但 outer/AURA.md 中的 @./child.md 解析相对于 outer/，
        # 而非 cwd。
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


class TestCache:
    """Memoization: force_reload and clear_cache semantics."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self) -> Iterator[None]:
        # 每个 test 前后都清空模块级缓存，避免跨 test 串味。
        clear_cache()
        yield
        clear_cache()

    @staticmethod
    def _install_open_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        # Path.open 是 read_bytes / read_text 的底层调用；计数它即可与具体
        # 读法解耦。
        counter = {"calls": 0}
        original_open = cast(Any, Path.open)

        def counting_open(self: Path, *args: Any, **kwargs: Any) -> Any:
            counter["calls"] += 1
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", counting_open)
        return counter

    def test_01_same_cwd_cached_single_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("HELLO")

        counter = self._install_open_counter(monkeypatch)
        first = load_project_memory(cwd)
        reads_after_first = counter["calls"]
        # 第一次必然有磁盘读。
        assert reads_after_first > 0
        second = load_project_memory(cwd)
        assert second == first
        # 第二次命中缓存，没有新增读。
        assert counter["calls"] == reads_after_first

    def test_02_force_reload_bypasses_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("HELLO")

        counter = self._install_open_counter(monkeypatch)
        load_project_memory(cwd)
        reads_after_first = counter["calls"]
        assert reads_after_first > 0
        load_project_memory(cwd, force_reload=True)
        assert counter["calls"] > reads_after_first

    def test_03_clear_cache_single_cwd_reloads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("HELLO")

        counter = self._install_open_counter(monkeypatch)
        load_project_memory(cwd)
        reads_after_first = counter["calls"]
        clear_cache(cwd)
        load_project_memory(cwd)
        # clear 之后再读应再次触发磁盘 I/O。
        assert counter["calls"] > reads_after_first

    def test_04_clear_cache_none_drops_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd_a = tmp_path / "a"
        cwd_b = tmp_path / "b"
        cwd_a.mkdir()
        cwd_b.mkdir()
        (cwd_a / "AURA.md").write_text("A")
        (cwd_b / "AURA.md").write_text("B")

        counter = self._install_open_counter(monkeypatch)
        load_project_memory(cwd_a)
        load_project_memory(cwd_b)
        reads_after_first_pass = counter["calls"]
        # 全清。
        clear_cache()
        load_project_memory(cwd_a)
        load_project_memory(cwd_b)
        # 两次都重读 —— 读数至少翻倍，且严格大于 first pass。
        assert counter["calls"] > reads_after_first_pass

    def test_05_clear_cache_absent_cwd_is_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        # 对从未加载过的 cwd 调用 clear_cache 应不报错。
        clear_cache(tmp_path / "never-loaded")

    def test_06_two_cwds_cached_independently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd_a = tmp_path / "a"
        cwd_b = tmp_path / "b"
        cwd_a.mkdir()
        cwd_b.mkdir()
        (cwd_a / "AURA.md").write_text("A")
        (cwd_b / "AURA.md").write_text("B")

        counter = self._install_open_counter(monkeypatch)
        load_project_memory(cwd_a)
        reads_after_a1 = counter["calls"]
        load_project_memory(cwd_b)
        reads_after_b1 = counter["calls"]
        # cwd_b 是新 cwd，必然读盘。
        assert reads_after_b1 > reads_after_a1

        # 命中 cwd_a 缓存。
        load_project_memory(cwd_a)
        assert counter["calls"] == reads_after_b1
        # 命中 cwd_b 缓存。
        load_project_memory(cwd_b)
        assert counter["calls"] == reads_after_b1


class TestReadWithImports:
    """`read_with_imports` 作为公共 API 被 Context 复用。"""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert read_with_imports(tmp_path / "nope.md") is None

    def test_directory_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "a_dir").mkdir()
        assert read_with_imports(tmp_path / "a_dir") is None

    def test_plain_content_returned_verbatim(self, tmp_path: Path) -> None:
        f = tmp_path / "x.md"
        f.write_text("HELLO")
        assert read_with_imports(f) == "HELLO"

    def test_expands_imports(self, tmp_path: Path) -> None:
        (tmp_path / "child.md").write_text("CHILD")
        f = tmp_path / "parent.md"
        f.write_text("pre\n@./child.md\npost")
        assert read_with_imports(f) == "pre\nCHILD\npost"

    def test_resolve_oserror_returns_raw_unexpanded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the file resolves with OSError, return its raw text rather than failing the read."""
        f = tmp_path / "parent.md"
        f.write_text("pre\n@./child.md\npost")
        (tmp_path / "child.md").write_text("CHILD")

        original_resolve = Path.resolve

        def _maybe_raise(self: Path, strict: bool = False) -> Path:
            if self.name == "parent.md":
                raise OSError("ELOOP")
            return original_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", _maybe_raise)
        # resolve 失败 → 跳过 @imports 展开，原文逐字返回（@-line 未被替换）。
        assert read_with_imports(f) == "pre\n@./child.md\npost"


def _fake_run_factory(
    *, returncode: int, stdout: str, raises: type[BaseException] | None = None
) -> Any:
    # subprocess.run 的替身：固定返回值或抛出指定异常，隔离真实 git。
    def _run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if raises is not None:
            raise raises("boom")
        return subprocess.CompletedProcess(
            args=["git"], returncode=returncode, stdout=stdout, stderr=""
        )

    return _run


class TestGitRootDetection:
    """`_detect_git_root` via the subprocess seam: success, failure, crash paths."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self) -> Iterator[None]:
        # git-root 行为会改变 ancestors，必须隔离模块级缓存防串味。
        clear_cache()
        yield
        clear_cache()

    def test_git_root_caps_walk_excludes_above_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inside a repo the walk must stop at git root — ancestor memory above is invisible."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        above = tmp_path / "above"
        root = above / "repo"
        inner = root / "pkg"
        inner.mkdir(parents=True)
        (above / "AURA.md").write_text("ABOVE-ROOT")
        (root / "AURA.md").write_text("REPO-ROOT")
        (inner / "AURA.md").write_text("PKG")

        monkeypatch.setattr(
            subprocess,
            "run",
            _fake_run_factory(returncode=0, stdout=f"{root.resolve()}\n"),
        )
        result = load_project_memory(inner)
        # 仅 root 及其下层被收录，above 被排除在 git 边界外。
        assert result == "REPO-ROOT\n\nPKG"
        assert "ABOVE-ROOT" not in result

    def test_git_root_equal_to_cwd_single_ancestor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When cwd is the git root itself, only that one directory is scanned."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        root = tmp_path / "repo"
        root.mkdir()
        (root / "AURA.md").write_text("ONLY-ROOT")

        monkeypatch.setattr(
            subprocess,
            "run",
            _fake_run_factory(returncode=0, stdout=f"{root.resolve()}\n"),
        )
        assert load_project_memory(root) == "ONLY-ROOT"

    def test_git_root_not_an_ancestor_degrades_to_fs_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reported git root unrelated to cwd must not corrupt the walk — degrade to fs root."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        outer = tmp_path / "x"
        inner = outer / "y"
        inner.mkdir(parents=True)
        (outer / "AURA.md").write_text("OUTER")
        (inner / "AURA.md").write_text("INNER")

        unrelated = tmp_path / "somewhere_else"
        unrelated.mkdir()
        monkeypatch.setattr(
            subprocess,
            "run",
            _fake_run_factory(returncode=0, stdout=f"{unrelated.resolve()}\n"),
        )
        # cwd 不在所谓 git_root 之下 —— 退化为完整 fs-root 走查，两层都收录。
        assert load_project_memory(inner) == "OUTER\n\nINNER"

    def test_git_nonzero_returncode_falls_back_to_fs_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`git rev-parse` failing (not a repo) must fall back to filesystem-root walk."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        outer = tmp_path / "x"
        inner = outer / "y"
        inner.mkdir(parents=True)
        (outer / "AURA.md").write_text("OUTER")
        (inner / "AURA.md").write_text("INNER")

        monkeypatch.setattr(
            subprocess, "run", _fake_run_factory(returncode=128, stdout="")
        )
        assert load_project_memory(inner) == "OUTER\n\nINNER"

    def test_git_empty_stdout_falls_back_to_fs_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero-length git output (zero/null boundary) is treated as 'no root', not as cwd."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        outer = tmp_path / "x"
        inner = outer / "y"
        inner.mkdir(parents=True)
        (outer / "AURA.md").write_text("OUTER")
        (inner / "AURA.md").write_text("INNER")

        monkeypatch.setattr(
            subprocess, "run", _fake_run_factory(returncode=0, stdout="   \n")
        )
        assert load_project_memory(inner) == "OUTER\n\nINNER"

    @pytest.mark.parametrize(
        "exc",
        [FileNotFoundError, subprocess.TimeoutExpired, OSError],
    )
    def test_git_subprocess_crash_falls_back_to_fs_walk(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        exc: type[BaseException],
    ) -> None:
        """git missing / timing out / OS error must never crash memory loading."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("PROJECT")

        def _raise(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if exc is subprocess.TimeoutExpired:
                raise subprocess.TimeoutExpired(cmd="git", timeout=2)
            raise exc("boom")

        monkeypatch.setattr(subprocess, "run", _raise)
        assert load_project_memory(cwd) == "PROJECT"

    def test_git_root_path_resolve_oserror_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A git root string that fails to resolve must degrade to fs walk, not crash."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        outer = tmp_path / "x"
        inner = outer / "y"
        inner.mkdir(parents=True)
        (outer / "AURA.md").write_text("OUTER")
        (inner / "AURA.md").write_text("INNER")

        bogus_root = "/nonexistent/git/root"
        monkeypatch.setattr(
            subprocess, "run", _fake_run_factory(returncode=0, stdout=f"{bogus_root}\n")
        )

        original_resolve = Path.resolve

        def _maybe_raise(self: Path, strict: bool = False) -> Path:
            # 仅 git 输出路径字符串触发 OSError，其余 resolve 正常工作。
            if str(self) == bogus_root:
                raise OSError("ENOENT")
            return original_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", _maybe_raise)
        # git_root 解析失败 → None → 退化为完整 fs-root 走查，两层都收录。
        assert load_project_memory(inner) == "OUTER\n\nINNER"


class TestAutoMemoryLayer:
    """`auto_memory_dir` MEMORY.md is the final recall layer with its own cache key."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self) -> Iterator[None]:
        clear_cache()
        yield
        clear_cache()

    def test_memory_md_appended_last(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto-recall MEMORY.md must concatenate after project layers, never before."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("PROJECT")
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("RECALL")

        result = load_project_memory(cwd, auto_memory_dir=mem_dir)
        assert result == "PROJECT\n\nRECALL"

    def test_missing_memory_md_dir_omits_layer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured recall dir with no MEMORY.md must drop the layer, not inject blanks."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("PROJECT")
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()

        assert load_project_memory(cwd, auto_memory_dir=mem_dir) == "PROJECT"

    def test_memory_md_keys_cache_separately_from_no_recall(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same cwd with vs without a recall dir are distinct cache keys — no cross-bleed."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("PROJECT")
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("RECALL")

        without = load_project_memory(cwd)
        with_recall = load_project_memory(cwd, auto_memory_dir=mem_dir)
        assert without == "PROJECT"
        assert with_recall == "PROJECT\n\nRECALL"
        # 再次取无 recall 变体应命中独立缓存，仍为纯 PROJECT。
        assert load_project_memory(cwd) == "PROJECT"


class TestReadRawBoundaries:
    """`_read_raw` byte-cap and I/O-error defenses surfaced through the public loader."""

    def test_oversize_file_truncated_with_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file over the 25 KB cap is truncated and tagged so the model knows to split it."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        big = "x" * 25_001
        (cwd / "AURA.md").write_text(big)

        result = load_project_memory(cwd)
        assert "WARNING: this file is 25001 bytes" in result
        assert "limit: 25000" in result
        # 截断后正文长度恰为 cap，超量被丢弃。
        assert result.count("x") == 25_000

    def test_exactly_at_cap_not_truncated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Off-by-one boundary: a file exactly at the cap must pass through untouched."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        exact = "y" * 25_000
        (cwd / "AURA.md").write_text(exact)

        result = load_project_memory(cwd)
        assert "WARNING" not in result
        assert result == exact

    def test_read_bytes_oserror_skips_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A readable-but-unreadable file (perm/IO error) is skipped, not fatal."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        target = cwd / "AURA.md"
        target.write_text("UNREACHABLE")

        original_read_bytes = Path.read_bytes

        def _maybe_fail(self: Path) -> bytes:
            if self.resolve() == target.resolve():
                raise OSError("EIO")
            return original_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _maybe_fail)
        assert load_project_memory(cwd) == ""


class TestResolveImportBoundaries:
    """`_resolve_import` tilde-home, resolve crash, and non-text extension guards."""

    def test_bare_tilde_imports_home_dir_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`@~` resolving to a home *dir* (not a file) yields no expansion, line dropped."""
        home = tmp_path / "home"
        home.mkdir()
        _patch_home(monkeypatch, home)

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("pre\n@~\npost")

        # ~ 解析为目录而非文件 —— import 被静默丢弃。
        assert load_project_memory(cwd) == "pre\npost"

    def test_resolve_oserror_on_import_drops_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An import target whose path resolution raises OSError is dropped, not propagated."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AURA.md").write_text("pre\n@./weird.md\npost")

        original_resolve = Path.resolve

        def _maybe_raise(self: Path, strict: bool = False) -> Path:
            if self.name == "weird.md":
                raise OSError("ELOOP")
            return original_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", _maybe_raise)
        assert load_project_memory(cwd) == "pre\npost"

    def test_non_text_extension_skipped_and_journaled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Binary/opaque extensions never enter the prompt and the skip is audited."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "data.bin").write_text("BINARY-PAYLOAD")
        (cwd / "AURA.md").write_text("pre\n@./data.bin\npost")

        events: list[tuple[str, dict[str, Any]]] = []

        def _capture(event: str, /, **fields: Any) -> None:
            events.append((event, fields))

        monkeypatch.setattr(journal, "write", _capture)
        result = load_project_memory(cwd)
        assert result == "pre\npost"
        assert "BINARY-PAYLOAD" not in result
        assert [e for e, _ in events] == ["import_non_text_skipped"]
        assert events[0][1]["suffix"] == ".bin"

    def test_text_extension_whitelist_allows_non_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whitelisted non-.md text (e.g. .txt) is allowed in, proving the guard is a filter."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "notes.txt").write_text("TXT-CONTENT")
        (cwd / "AURA.md").write_text("@./notes.txt")

        assert load_project_memory(cwd) == "TXT-CONTENT"


class TestExpandReadRace:
    """`_expand`: an import target that vanishes between resolution and read drops cleanly."""

    def test_child_read_returns_none_after_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resolved import that becomes unreadable mid-expansion is dropped, not crashed."""
        _patch_home(monkeypatch, tmp_path / "home")
        (tmp_path / "home").mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        child = cwd / "child.md"
        child.write_text("CHILD")
        (cwd / "AURA.md").write_text("pre\n@./child.md\npost")

        original_read_bytes = Path.read_bytes

        def _vanish_child(self: Path) -> bytes:
            if self.name == "child.md":
                raise OSError("vanished")
            return original_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _vanish_child)
        result = load_project_memory(cwd)
        assert result == "pre\npost"
        assert "CHILD" not in result
