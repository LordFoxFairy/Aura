"""Tests for aura.application.memory.rules: discovery, frontmatter, glob match, cache."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from aura.application.memory.rules import (
    clear_cache,
    load_rules,
    match,
)
from aura.application.memory.rules_types import Rule, RulesBundle
from aura.core import journal as journal_module


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
        # 250 lines of short content (well under the 25 KB byte cap)
        body = "\n".join(f"L{i}" for i in range(250))
        (rules_dir / "long.md").write_text(body + "\n")

        bundle = load_rules(cwd)
        assert len(bundle.unconditional) == 1
        content = bundle.unconditional[0].content
        # F-03-007 — truncation marker is now an explicit WARNING with
        # byte counts so the model knows what got dropped + the limit.
        assert "WARNING:" in content
        assert "limit: 25000" in content
        # 200 lines retained: L199 is present, L200 onwards is not.
        assert "L199" in content
        assert "L200" not in content

    def test_07_body_byte_truncation_25k(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        # 30 lines × 1000 bytes/line = 30000 bytes > 25_000 cap, but only
        # 30 lines (< _MAX_LINES=200) so byte-cap is the trigger.
        body = "\n".join("x" * 1000 for _ in range(30))
        (rules_dir / "big.md").write_text(body)

        bundle = load_rules(cwd)
        assert len(bundle.unconditional) == 1
        content = bundle.unconditional[0].content
        assert "WARNING:" in content
        assert "limit: 25000" in content
        # Pre-marker body must respect the byte cap.
        marker_idx = content.index("\nWARNING:")
        assert len(content[:marker_idx].encode("utf-8")) <= 25_000

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
        # user-layer rules use ~ as their base_dir.
        for r in bundle.unconditional:
            assert r.base_dir == home.resolve()

    def test_09_project_layer_no_walk_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        # Ancestor has .aura/rules/ — it must NOT be scanned.
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
        # ancestor.md must NOT appear; subdirs of cwd's rules/ are included.
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
        # Unbalanced bracket in YAML list.
        (rules_dir / "bad.md").write_text(
            "---\npaths: [broken\n---\nbody\n"
        )
        (rules_dir / "ok.md").write_text(
            "---\npaths: \"*.py\"\n---\nok-body\n"
        )

        bundle = load_rules(cwd)
        # bad.md is silently skipped; ok.md preserved; load does not crash.
        names_c = {r.source_path.name for r in bundle.conditional}
        names_u = {r.source_path.name for r in bundle.unconditional}
        assert "bad.md" not in names_c
        assert "bad.md" not in names_u
        assert names_c == {"ok.md"}

    def test_yaml_parse_failure_emits_journal_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        bad_md = rules_dir / "bad.md"
        bad_md.write_text("---\npaths: [broken\n---\nbody\n")

        log = tmp_path / "events.jsonl"
        journal_module.configure(log)
        try:
            load_rules(cwd)
        finally:
            journal_module.reset()

        events = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matching = [e for e in events if e["event"] == "rule_yaml_parse_failed"]
        assert len(matching) == 1, f"expected 1 rule_yaml_parse_failed, got {events}"
        ev = matching[0]
        assert ev["path"] == str(bad_md.resolve()) or ev["path"] == str(bad_md)
        assert isinstance(ev["error"], str) and ev["error"]

    def test_invalid_paths_type_emits_journal_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        rule_md = rules_dir / "r.md"
        # `paths: 42` — int is not a supported type; _extract_globs returns _SKIP.
        rule_md.write_text("---\npaths: 42\n---\nbody\n")

        log = tmp_path / "events.jsonl"
        journal_module.configure(log)
        try:
            load_rules(cwd)
        finally:
            journal_module.reset()

        events = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matching = [e for e in events if e["event"] == "rule_paths_invalid_type"]
        assert len(matching) == 1, f"expected 1 rule_paths_invalid_type, got {events}"
        ev = matching[0]
        assert ev["path"] == str(rule_md.resolve()) or ev["path"] == str(rule_md)
        assert ev["actual_type"] == "int"


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
        # user-layer style: base_dir is ~, target lives elsewhere.
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
        # `[z-a]` is a bad character range; pathspec raises on compile.
        rule = _make_rule(rule_file, proj, ("[z-a]",))

        bundle = RulesBundle(unconditional=[], conditional=[rule])
        # match() must not raise; it returns an empty list.
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
        # Duplicate globs still collapse to a single rule entry.
        r_z = _make_rule(rf_z, proj, ("*.py", "*.py"))
        r_a = _make_rule(rf_a, proj, ("*.py",))

        bundle = RulesBundle(unconditional=[], conditional=[r_z, r_a])
        result = match(bundle, target)
        assert [r.source_path for r in result] == sorted(
            [rf_a.resolve(), rf_z.resolve()]
        )

    def test_malformed_glob_emits_journal_event(self, tmp_path: Path) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        target = proj / "foo.py"
        target.write_text("")
        rule_file = proj / ".aura" / "rules" / "bad.md"
        rule_file.parent.mkdir(parents=True)
        rule_file.write_text("body")
        rule = _make_rule(rule_file, proj, ("[z-a]",))
        bundle = RulesBundle(unconditional=[], conditional=[rule])

        log = tmp_path / "events.jsonl"
        journal_module.configure(log)
        try:
            match(bundle, target)
        finally:
            journal_module.reset()

        events = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matching = [e for e in events if e["event"] == "rule_glob_compile_failed"]
        assert len(matching) == 1, f"expected 1 rule_glob_compile_failed, got {events}"
        ev = matching[0]
        assert ev["path"] == str(rule.source_path)
        assert ev["glob"] == "[z-a]"
        assert isinstance(ev["error"], str) and ev["error"]


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
        # Same cached reference, no new reads.
        assert second is first
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


def _read_events(log: Path) -> list[dict[str, Any]]:
    """Read journal events; tolerate the file being absent (no events emitted)."""
    if not log.exists():
        return []
    return [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestOutOfCwdRuleWarning:
    """Phase 3 Task 6 — rules anchored outside cwd warn at load time."""

    def test_absolute_pattern_outside_cwd_emits_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        # Absolute pattern anchored outside the project root.
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        rule_md = rules_dir / "stray.md"
        rule_md.write_text(
            f"---\npaths: \"{outside}/**/*.py\"\n---\nSTRAY-BODY\n"
        )

        log = tmp_path / "events.jsonl"
        journal_module.configure(log)
        try:
            load_rules(cwd, force_reload=True)
        finally:
            journal_module.reset()

        events = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        warnings = [e for e in events if e["event"] == "out_of_cwd_rule_warning"]
        assert len(warnings) == 1, f"expected 1 warning, got {events}"
        ev = warnings[0]
        assert ev["path"] == str(rule_md.resolve())
        assert ev["patterns"] == [f"{outside}/**/*.py"]
        assert ev["cwd"] == str(cwd.resolve())

    def test_relative_pattern_does_not_warn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        rule_md = rules_dir / "ok.md"
        rule_md.write_text("---\npaths: \"src/**/*.py\"\n---\nbody\n")

        log = tmp_path / "events.jsonl"
        journal_module.configure(log)
        try:
            load_rules(cwd, force_reload=True)
        finally:
            journal_module.reset()

        events = _read_events(log)
        warnings = [e for e in events if e["event"] == "out_of_cwd_rule_warning"]
        assert warnings == []

    def test_absolute_pattern_under_cwd_does_not_warn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        rule_md = rules_dir / "abs_inside.md"
        rule_md.write_text(
            f"---\npaths: \"{cwd}/**/*.py\"\n---\nbody\n"
        )

        log = tmp_path / "events.jsonl"
        journal_module.configure(log)
        try:
            load_rules(cwd, force_reload=True)
        finally:
            journal_module.reset()

        events = _read_events(log)
        warnings = [e for e in events if e["event"] == "out_of_cwd_rule_warning"]
        assert warnings == []

    def test_warning_emitted_once_per_rule(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple offending patterns on one rule → single event with all of them."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        outside1 = tmp_path / "out1"
        outside2 = tmp_path / "out2"
        outside1.mkdir()
        outside2.mkdir()
        rule_md = rules_dir / "many.md"
        # YAML list with two absolute out-of-cwd patterns and one relative.
        rule_md.write_text(
            "---\npaths:\n"
            f"  - \"{outside1}/**/*.py\"\n"
            f"  - \"{outside2}/**/*.py\"\n"
            "  - \"src/*.py\"\n"
            "---\nbody\n"
        )

        log = tmp_path / "events.jsonl"
        journal_module.configure(log)
        try:
            load_rules(cwd, force_reload=True)
        finally:
            journal_module.reset()

        events = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        warnings = [e for e in events if e["event"] == "out_of_cwd_rule_warning"]
        assert len(warnings) == 1, f"expected 1 warning, got {events}"
        ev = warnings[0]
        assert ev["patterns"] == [
            f"{outside1}/**/*.py",
            f"{outside2}/**/*.py",
        ]


def _make_project_rules(cwd: Path) -> Path:
    rules_dir = cwd / ".aura" / "rules"
    rules_dir.mkdir(parents=True)
    return rules_dir


class TestFrontmatterEdgeCases:
    """Frontmatter parser must degrade to 'no frontmatter' on degenerate heads."""

    def test_empty_file_is_unconditional_empty_body(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A zero-byte rule file must not crash the scan; it lands unconditional."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        (rules_dir / "empty.md").write_text("")

        bundle = load_rules(cwd)
        assert len(bundle.unconditional) == 1
        assert bundle.conditional == []
        assert bundle.unconditional[0].globs == ()
        assert bundle.unconditional[0].content == ""

    def test_open_fence_without_close_treated_as_body(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unterminated frontmatter must not eat the body; whole file is content."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        # Opening '---' but no closing fence anywhere in the file.
        (rules_dir / "open.md").write_text("---\npaths: src/*.py\nstill body\n")

        bundle = load_rules(cwd)
        assert bundle.conditional == []
        assert len(bundle.unconditional) == 1
        rule = bundle.unconditional[0]
        assert rule.globs == ()
        # The raw '---' line and the fake 'paths:' survive as literal body.
        assert "paths: src/*.py" in rule.content
        assert "still body" in rule.content

    def test_dots_closing_fence_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """YAML '...' document-end terminator must close frontmatter like '---'."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        (rules_dir / "dots.md").write_text("---\npaths: \"*.py\"\n...\nbody-here\n")

        bundle = load_rules(cwd)
        assert len(bundle.conditional) == 1
        assert bundle.conditional[0].globs == ("*.py",)
        assert bundle.conditional[0].content.strip() == "body-here"

    def test_top_level_yaml_list_is_unconditional(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-mapping YAML head (a bare list) is not a paths spec → unconditional."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        # Frontmatter parses to a list, not a dict → _extract_globs returns ().
        (rules_dir / "list.md").write_text("---\n- a\n- b\n---\nbody\n")

        bundle = load_rules(cwd)
        assert bundle.conditional == []
        assert len(bundle.unconditional) == 1
        assert bundle.unconditional[0].globs == ()

    def test_top_level_yaml_scalar_is_unconditional(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scalar YAML head (a bare string) carries no paths → unconditional."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        (rules_dir / "scalar.md").write_text("---\njust-a-string\n---\nbody\n")

        bundle = load_rules(cwd)
        assert bundle.conditional == []
        assert len(bundle.unconditional) == 1
        assert bundle.unconditional[0].globs == ()


class TestUniversalGlobNormalization:
    """A rule globbing everything is unconditional — it must not be a path filter."""

    @pytest.mark.parametrize("spec", ["**", "**/*", "**, **/*"])
    def test_universal_globs_collapse_to_unconditional(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: str
    ) -> None:
        """Match-everything globs degrade to unconditional so they always apply."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        (rules_dir / "all.md").write_text(f"---\npaths: \"{spec}\"\n---\nbody\n")

        bundle = load_rules(cwd)
        assert bundle.conditional == []
        assert len(bundle.unconditional) == 1
        assert bundle.unconditional[0].globs == ()

    def test_universal_mixed_with_specific_stays_conditional(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only an all-universal set collapses; one specific glob keeps it conditional."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        (rules_dir / "mix.md").write_text("---\npaths: \"**, src/*.py\"\n---\nbody\n")

        bundle = load_rules(cwd)
        assert bundle.unconditional == []
        assert len(bundle.conditional) == 1
        assert bundle.conditional[0].globs == ("**", "src/*.py")


class TestMatchDedup:
    """Two rules sharing one source must yield at most one match, even on overlap."""

    def test_two_rules_same_source_collapse_to_one(self, tmp_path: Path) -> None:
        """Re-loaded duplicate rules from one file must never double-fire on a path."""
        proj = tmp_path / "proj"
        proj.mkdir()
        target = proj / "foo.py"
        target.write_text("")
        src = proj / ".aura" / "rules" / "r.md"
        src.parent.mkdir(parents=True)
        src.write_text("body")
        # Distinct Rule objects, identical resolved source_path.
        first = _make_rule(src, proj, ("*.py",))
        second = _make_rule(src, proj, ("*.py",))

        bundle = RulesBundle(unconditional=[], conditional=[first, second])
        result = match(bundle, target)
        assert len(result) == 1
        assert result[0] is first

    def test_match_resolve_oserror_falls_back_to_unresolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path that cannot be resolved (e.g. broken symlink) still matches by name."""
        proj = tmp_path / "proj"
        proj.mkdir()
        target = proj / "foo.py"
        target.write_text("")
        src = proj / ".aura" / "rules" / "r.md"
        src.parent.mkdir(parents=True)
        src.write_text("body")
        rule = _make_rule(src, proj, ("*.py",))

        original_resolve = cast(Callable[..., Path], Path.resolve)

        def failing_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
            if self == target:
                raise OSError("cannot resolve")
            return original_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", failing_resolve)
        result = match(RulesBundle(unconditional=[], conditional=[rule]), target)
        assert result == [rule]


class TestScanIoFailures:
    """Filesystem hiccups mid-scan must drop the offending file, never crash."""

    def test_rglob_oserror_yields_empty_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the rules dir becomes unreadable mid-walk, the layer yields no rules."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        (rules_dir / "a.md").write_text("A\n")

        def failing_rglob(self: Path, *args: Any, **kwargs: Any) -> Any:
            raise OSError("rglob denied")

        monkeypatch.setattr(Path, "rglob", failing_rglob)
        bundle = load_rules(cwd, force_reload=True)
        assert bundle.unconditional == []
        assert bundle.conditional == []

    def test_read_bytes_oserror_drops_only_that_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unreadable rule file is skipped; sibling readable rules survive."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        (rules_dir / "bad.md").write_text("BAD\n")
        (rules_dir / "ok.md").write_text("OK\n")

        original_read = cast(Callable[..., bytes], Path.read_bytes)

        def failing_read(self: Path, *args: Any, **kwargs: Any) -> bytes:
            if self.name == "bad.md":
                raise OSError("read denied")
            return original_read(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_bytes", failing_read)
        bundle = load_rules(cwd, force_reload=True)
        names = {r.source_path.name for r in bundle.unconditional}
        assert names == {"ok.md"}

    def test_build_rule_resolve_oserror_drops_only_that_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file whose source path can't be resolved is dropped, siblings kept."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        (rules_dir / "stale.md").write_text("STALE\n")
        (rules_dir / "good.md").write_text("GOOD\n")

        original_resolve = cast(Callable[..., Path], Path.resolve)

        def failing_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
            if self.name == "stale.md":
                raise OSError("resolve denied")
            return original_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", failing_resolve)
        bundle = load_rules(cwd, force_reload=True)
        names = {r.source_path.name for r in bundle.unconditional}
        assert names == {"good.md"}

    def test_file_vanishing_between_walk_and_read_is_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rule deleted after rglob lists it (TOCTOU) is skipped, not fatal."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        (rules_dir / "ghost.md").write_text("GHOST\n")
        (rules_dir / "real.md").write_text("REAL\n")

        original_is_file = cast(Callable[..., bool], Path.is_file)
        # ghost.md passes _scan_layer's is_file() but vanishes before _read_text's.
        seen_ghost = {"count": 0}

        def flaky_is_file(self: Path, *args: Any, **kwargs: Any) -> bool:
            if self.name == "ghost.md":
                seen_ghost["count"] += 1
                return seen_ghost["count"] == 1
            return original_is_file(self, *args, **kwargs)

        monkeypatch.setattr(Path, "is_file", flaky_is_file)
        bundle = load_rules(cwd, force_reload=True)
        names = {r.source_path.name for r in bundle.unconditional}
        assert names == {"real.md"}

    def test_directory_named_dot_md_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A subdirectory literally named 'x.md' must not be parsed as a rule file."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        (rules_dir / "fakedir.md").mkdir()
        (rules_dir / "good.md").write_text("GOOD\n")

        bundle = load_rules(cwd, force_reload=True)
        names = {r.source_path.name for r in bundle.unconditional}
        assert names == {"good.md"}


class TestOutOfCwdResolveFailure:
    """An absolute glob whose prefix can't be resolved is conservatively flagged."""

    def test_unresolvable_absolute_prefix_is_offender(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If we can't prove the absolute glob lives under cwd, warn rather than trust it."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        sentinel = "/SENTINEL_UNRESOLVABLE"
        rule_md = rules_dir / "stray.md"
        rule_md.write_text(f"---\npaths: \"{sentinel}/**/*.py\"\n---\nbody\n")

        original_resolve = cast(Callable[..., Path], Path.resolve)

        def failing_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
            if str(self).startswith(sentinel):
                raise OSError("resolve denied")
            return original_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", failing_resolve)
        log = tmp_path / "events.jsonl"
        journal_module.configure(log)
        try:
            load_rules(cwd, force_reload=True)
        finally:
            journal_module.reset()

        events = _read_events(log)
        warnings = [e for e in events if e["event"] == "out_of_cwd_rule_warning"]
        assert len(warnings) == 1, f"expected 1 warning, got {events}"
        assert warnings[0]["patterns"] == [f"{sentinel}/**/*.py"]


class TestCacheIdempotency:
    """Repeated load/clear cycles must stay deterministic and isolated per cwd."""

    def test_double_clear_cache_is_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clearing an already-clear cache must be a no-op, never raise KeyError."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd = tmp_path / "project"
        rules_dir = _make_project_rules(cwd)
        (rules_dir / "r.md").write_text("---\npaths: \"*.py\"\n---\nbody\n")

        load_rules(cwd)
        clear_cache(cwd)
        clear_cache(cwd)
        clear_cache()
        # A fresh load still succeeds and is stable across repeats.
        again = load_rules(cwd)
        assert again is load_rules(cwd)
        assert len(again.conditional) == 1

    def test_distinct_cwds_cache_independently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two projects must not share a cache slot; each keeps its own rules."""
        _isolate_user_layer(monkeypatch, tmp_path)
        cwd_a = tmp_path / "a"
        cwd_b = tmp_path / "b"
        (cwd_a / ".aura" / "rules").mkdir(parents=True)
        (cwd_b / ".aura" / "rules").mkdir(parents=True)
        (cwd_a / ".aura" / "rules" / "a.md").write_text("AAA\n")
        (cwd_b / ".aura" / "rules" / "b.md").write_text("BBB\n")

        bundle_a = load_rules(cwd_a)
        bundle_b = load_rules(cwd_b)
        assert {r.source_path.name for r in bundle_a.unconditional} == {"a.md"}
        assert {r.source_path.name for r in bundle_b.unconditional} == {"b.md"}
        # Clearing one leaves the other's cached reference intact.
        clear_cache(cwd_a)
        assert load_rules(cwd_b) is bundle_b
