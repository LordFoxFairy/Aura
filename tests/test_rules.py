"""Tests for aura.core.rules (Task 4): discovery, frontmatter, glob match, cache."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from aura.core.memory.rules import (
    Rule,
    RulesBundle,
    clear_cache,
    load_rules,
    match,
)


def _patch_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: home)


def _isolate_user_layer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ~ at an empty tmp dir so user-layer scans see nothing by default."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    _patch_home(monkeypatch, home)
    return home


@pytest.fixture(autouse=True)
def _reset_rules_cache() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Discovery (11 tests)
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_01_valid_paths_frontmatter_lands_in_conditional(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "foo.md").write_text(
            "---\npaths: \"src/**/*.py\"\n---\nBODY-TEXT\n"
        )

        bundle = load_rules(cwd)
        assert len(bundle.conditional) == 1
        assert bundle.unconditional == []
        rule = bundle.conditional[0]
        assert rule.globs == ("src/**/*.py",)
        assert rule.content.strip() == "BODY-TEXT"
        assert rule.source_path == (rules_dir / "foo.md").resolve()
        assert rule.base_dir == cwd.resolve()

    def test_02_no_frontmatter_is_unconditional(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "plain.md").write_text("just a body\nno frontmatter here\n")

        bundle = load_rules(cwd)
        assert bundle.conditional == []
        assert len(bundle.unconditional) == 1
        rule = bundle.unconditional[0]
        assert rule.globs == ()
        assert "just a body" in rule.content

    def test_03_frontmatter_without_paths_is_unconditional(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "meta.md").write_text(
            "---\ntitle: hello\n---\nbody-content\n"
        )

        bundle = load_rules(cwd)
        assert bundle.conditional == []
        assert len(bundle.unconditional) == 1
        assert bundle.unconditional[0].globs == ()
        assert "body-content" in bundle.unconditional[0].content

    def test_04_paths_comma_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "r.md").write_text(
            "---\npaths: \"a.py, b.py\"\n---\nbody\n"
        )

        bundle = load_rules(cwd)
        assert len(bundle.conditional) == 1
        assert bundle.conditional[0].globs == ("a.py", "b.py")

    def test_05_paths_yaml_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "r.md").write_text(
            "---\npaths:\n  - a.py\n  - b.py\n---\nbody\n"
        )

        bundle = load_rules(cwd)
        assert len(bundle.conditional) == 1
        assert bundle.conditional[0].globs == ("a.py", "b.py")

    def test_06_body_line_truncation_200(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        # 250 lines of short content (well under 4096 bytes)
        body = "\n".join(f"L{i}" for i in range(250))
        (rules_dir / "long.md").write_text(body + "\n")

        bundle = load_rules(cwd)
        assert len(bundle.unconditional) == 1
        content = bundle.unconditional[0].content
        assert content.endswith("\n… (truncated)")
        # 200 lines kept; L199 present, L200 onwards NOT present
        assert "L199" in content
        assert "L200" not in content

    def test_07_body_byte_truncation_4096(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        # 10 lines × 600 bytes/line = 6000 bytes > 4096, but only 10 lines (< 200)
        body = "\n".join("x" * 600 for _ in range(10))
        (rules_dir / "big.md").write_text(body)

        bundle = load_rules(cwd)
        assert len(bundle.unconditional) == 1
        content = bundle.unconditional[0].content
        assert content.endswith("\n… (truncated)")
        # body up-to-truncation is at most 4096 bytes
        trimmed = content[: -len("\n… (truncated)")]
        assert len(trimmed.encode("utf-8")) <= 4096

    def test_08_user_layer_scanned_recursively(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = _isolate_user_layer(monkeypatch, tmp_path)
        user_rules = home / ".aura" / "rules"
        (user_rules / "sub").mkdir(parents=True)
        (user_rules / "top.md").write_text("TOP-RULE\n")
        (user_rules / "sub" / "nested.md").write_text("NESTED-RULE\n")

        cwd = tmp_path / "project"
        cwd.mkdir()

        bundle = load_rules(cwd)
        sources = {r.source_path.name for r in bundle.unconditional}
        assert sources == {"top.md", "nested.md"}
        # user-layer base_dir is ~
        for r in bundle.unconditional:
            assert r.base_dir == home.resolve()

    def test_09_project_layer_no_walk_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        # ancestor has .aura/rules/ — it should NOT be scanned
        ancestor = tmp_path / "outer"
        (ancestor / ".aura" / "rules").mkdir(parents=True)
        (ancestor / ".aura" / "rules" / "ancestor.md").write_text("ANCESTOR\n")

        cwd = ancestor / "inner"
        (cwd / ".aura" / "rules").mkdir(parents=True)
        (cwd / ".aura" / "rules" / "sub").mkdir()
        (cwd / ".aura" / "rules" / "at-cwd.md").write_text("AT-CWD\n")
        (cwd / ".aura" / "rules" / "sub" / "nested.md").write_text("NESTED\n")

        bundle = load_rules(cwd)
        names = {r.source_path.name for r in bundle.unconditional}
        # "ancestor.md" must NOT appear; subdir of cwd's rules/ DOES
        assert "ancestor.md" not in names
        assert names == {"at-cwd.md", "nested.md"}

    def test_10_non_md_files_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "keep.md").write_text("KEEP\n")
        (rules_dir / "ignore.txt").write_text("IGNORE\n")
        (rules_dir / "README").write_text("IGNORE2\n")

        bundle = load_rules(cwd)
        names = {r.source_path.name for r in bundle.unconditional}
        assert names == {"keep.md"}

    def test_11_malformed_yaml_silent_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        # unbalanced bracket in YAML list
        (rules_dir / "bad.md").write_text(
            "---\npaths: [broken\n---\nbody\n"
        )
        (rules_dir / "ok.md").write_text(
            "---\npaths: \"*.py\"\n---\nok-body\n"
        )

        bundle = load_rules(cwd)
        # bad.md silently skipped; ok.md preserved; load does not crash
        names_c = {r.source_path.name for r in bundle.conditional}
        names_u = {r.source_path.name for r in bundle.unconditional}
        assert "bad.md" not in names_c
        assert "bad.md" not in names_u
        assert names_c == {"ok.md"}


# ---------------------------------------------------------------------------
# Matching (5 tests)
# ---------------------------------------------------------------------------


def _make_rule(source: Path, base_dir: Path, globs: tuple[str, ...]) -> Rule:
    return Rule(
        source_path=source.resolve(),
        base_dir=base_dir.resolve(),
        globs=globs,
        content="rule-body",
    )


class TestMatching:
    def test_12_relative_glob_matches_nested_under_base(
        self, tmp_path: Path
    ) -> None:
        proj = tmp_path / "proj"
        (proj / "src" / "a").mkdir(parents=True)
        path = proj / "src" / "a" / "b.py"
        path.write_text("")
        rule_file = proj / ".aura" / "rules" / "r.md"
        rule_file.parent.mkdir(parents=True)
        rule_file.write_text("body")
        rule = _make_rule(rule_file, proj, ("src/**/*.py",))

        bundle = RulesBundle(unconditional=[], conditional=[rule])
        result = match(bundle, path)
        assert result == [rule]

    def test_13_double_star_any_py_matches_abs(
        self, tmp_path: Path
    ) -> None:
        # user-layer style: base_dir=~, target is elsewhere
        home = tmp_path / "home"
        home.mkdir()
        rule_file = home / ".aura" / "rules" / "any.md"
        rule_file.parent.mkdir(parents=True)
        rule_file.write_text("body")
        rule = _make_rule(rule_file, home, ("**/*.py",))

        target = tmp_path / "anywhere" / "foo.py"
        target.parent.mkdir(parents=True)
        target.write_text("")

        bundle = RulesBundle(unconditional=[], conditional=[rule])
        assert match(bundle, target) == [rule]

    def test_14_single_star_does_not_cross_dirs(
        self, tmp_path: Path
    ) -> None:
        proj = tmp_path / "proj"
        (proj / "tests" / "nested").mkdir(parents=True)
        target = proj / "tests" / "nested" / "test_b.py"
        target.write_text("")
        rule_file = proj / ".aura" / "rules" / "t.md"
        rule_file.parent.mkdir(parents=True)
        rule_file.write_text("body")
        rule = _make_rule(rule_file, proj, ("tests/*.py",))

        bundle = RulesBundle(unconditional=[], conditional=[rule])
        assert match(bundle, target) == []

    def test_15_malformed_glob_silent_no_match(
        self, tmp_path: Path
    ) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        target = proj / "foo.py"
        target.write_text("")
        rule_file = proj / ".aura" / "rules" / "bad.md"
        rule_file.parent.mkdir(parents=True)
        rule_file.write_text("body")
        # `[z-a]` is a bad character range → pathspec raises on compile
        rule = _make_rule(rule_file, proj, ("[z-a]",))

        bundle = RulesBundle(unconditional=[], conditional=[rule])
        # Should not raise; returns empty list
        assert match(bundle, target) == []

    def test_16_match_output_dedup_and_sorted(
        self, tmp_path: Path
    ) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        target = proj / "foo.py"
        target.write_text("")
        # Two rules, order intentionally reversed alphabetically in input.
        rf_z = proj / ".aura" / "rules" / "z.md"
        rf_a = proj / ".aura" / "rules" / "a.md"
        rf_z.parent.mkdir(parents=True)
        rf_z.write_text("b")
        rf_a.write_text("b")
        r_z = _make_rule(rf_z, proj, ("*.py", "*.py"))  # dup globs still → single rule entry
        r_a = _make_rule(rf_a, proj, ("*.py",))

        bundle = RulesBundle(unconditional=[], conditional=[r_z, r_a])
        result = match(bundle, target)
        assert [r.source_path for r in result] == sorted(
            [rf_a.resolve(), rf_z.resolve()]
        )


# ---------------------------------------------------------------------------
# Cache (3 tests)
# ---------------------------------------------------------------------------


class TestCache:
    @staticmethod
    def _install_open_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        counter = {"calls": 0}
        original_open = cast(Any, Path.open)

        def counting_open(self: Path, *args: Any, **kwargs: Any) -> Any:
            counter["calls"] += 1
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", counting_open)
        return counter

    def test_17_memoized_single_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "r.md").write_text("---\npaths: \"*.py\"\n---\nbody\n")

        counter = self._install_open_counter(monkeypatch)
        first = load_rules(cwd)
        reads_after_first = counter["calls"]
        assert reads_after_first > 0
        second = load_rules(cwd)
        assert second is first  # cached reference
        assert counter["calls"] == reads_after_first

    def test_18_clear_cache_forces_rescan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "r.md").write_text("---\npaths: \"*.py\"\n---\nbody\n")

        counter = self._install_open_counter(monkeypatch)
        load_rules(cwd)
        reads_after_first = counter["calls"]
        clear_cache(cwd)
        load_rules(cwd)
        assert counter["calls"] > reads_after_first

    def test_19_force_reload_bypasses_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "r.md").write_text("---\npaths: \"*.py\"\n---\nbody\n")

        counter = self._install_open_counter(monkeypatch)
        load_rules(cwd)
        reads_after_first = counter["calls"]
        load_rules(cwd, force_reload=True)
        assert counter["calls"] > reads_after_first
