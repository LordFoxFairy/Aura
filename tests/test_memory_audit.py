"""Coverage for audit findings F-03-003 / 005 / 006 / 007 / 008.

One assertion-per-finding here keeps the audit traceable; deeper behaviour
lives alongside its module's regular suite.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.messages import SystemMessage

from aura.core import journal as journal_module
from aura.core.memory import project_memory as pm
from aura.core.memory import rules as rules_module
from aura.core.memory.context import Context
from aura.core.memory.rules import (
    Rule,
    RulesBundle,
    _extract_globs,
    _truncate,
    load_rules,
)


def _patch_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: home)


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    pm.clear_cache()
    rules_module.clear_cache()
    yield
    pm.clear_cache()
    rules_module.clear_cache()


# ---------------------------------------------------------------------------
# F-03-003 — system-reminder OVERRIDE framing
# ---------------------------------------------------------------------------


def test_f03_003_project_memory_wrapped_in_system_reminder_with_override(
    tmp_path: Path,
) -> None:
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="PRIMARY-BODY",
        rules=RulesBundle(),
    )
    out = ctx.build([])
    # SystemMessage sibling that precedes history.
    eager = next(m for m in out if "PRIMARY-BODY" in str(m.content))
    assert isinstance(eager, SystemMessage)
    body = str(eager.content)
    assert body.startswith("<system-reminder>\n")
    assert body.endswith("\n</system-reminder>")
    # claude-code's exact phrasing.
    assert "These instructions OVERRIDE any default behavior" in body
    assert "you MUST follow them exactly as written" in body
    assert "<project-memory>\nPRIMARY-BODY\n</project-memory>" in body


def test_f03_003_no_eager_emits_no_system_reminder(tmp_path: Path) -> None:
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    out = ctx.build([])
    assert all("<system-reminder>" not in str(m.content) for m in out)


# ---------------------------------------------------------------------------
# F-03-005 — paths: ["**"] universal carve-out
# ---------------------------------------------------------------------------


def test_f03_005_paths_double_star_yields_unconditional_empty_tuple() -> None:
    assert _extract_globs({"paths": ["**"]}) == ()
    assert _extract_globs({"paths": ["**/*"]}) == ()
    assert _extract_globs({"paths": ["**", "**/*"]}) == ()
    assert _extract_globs({"paths": "**"}) == ()
    assert _extract_globs({"paths": "**/*"}) == ()
    # Mixed entries (one non-universal) keep the original tuple.
    assert _extract_globs({"paths": ["**", "src/**/*.py"]}) == ("**", "src/**/*.py")


def test_f03_005_universal_paths_lands_in_unconditional_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _patch_home(monkeypatch, home)

    cwd = tmp_path / "project"
    rules_dir = cwd / ".aura" / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "always.md").write_text(
        '---\npaths: ["**"]\n---\nALWAYS-ON\n'
    )

    bundle = load_rules(cwd)
    # Carve-out: universal paths is sugar for "unconditional".
    assert len(bundle.unconditional) == 1
    assert bundle.unconditional[0].globs == ()
    assert "ALWAYS-ON" in bundle.unconditional[0].content
    assert bundle.conditional == []


# ---------------------------------------------------------------------------
# F-03-006 — walk-up stops at git root
# ---------------------------------------------------------------------------


def test_f03_006_walk_caps_at_git_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    above = tmp_path / "above"
    above.mkdir()
    (above / "AURA.md").write_text("ABOVE-MUST-NOT-LOAD")

    repo = above / "repo"
    repo.mkdir()
    (repo / "AURA.md").write_text("REPO-LOAD")

    cwd = repo / "src"
    cwd.mkdir()
    subprocess.run(
        ["git", "init", "-q"], cwd=repo, check=True, timeout=10
    )

    result = pm.load_project_memory(cwd)
    assert "REPO-LOAD" in result
    assert "ABOVE-MUST-NOT-LOAD" not in result


def test_f03_006_no_git_falls_back_to_fs_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force `_detect_git_root` into the failure branch (FileNotFoundError);
    # walk should fall back to the legacy filesystem-root behaviour.
    monkeypatch.setattr(pm, "_detect_git_root", lambda _cwd: None)
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    (outer / "AURA.md").write_text("OUTER")
    (inner / "AURA.md").write_text("INNER")

    assert pm.load_project_memory(inner) == "OUTER\n\nINNER"


# ---------------------------------------------------------------------------
# F-03-007 — truncation marker carries WARNING + byte counts
# ---------------------------------------------------------------------------


def test_f03_007_truncate_marker_contains_warning_and_byte_counts() -> None:
    body = "x" * 30_000
    out = _truncate(body)
    # Marker must surface "WARNING" and both numbers.
    assert "WARNING" in out
    assert "30000" in out
    assert "25000" in out
    assert "split long content" in out
    # Body itself is at-most byte_cap bytes (excluding marker).
    marker_idx = out.index("\nWARNING")
    assert len(out[:marker_idx].encode("utf-8")) <= 25_000


def test_f03_007_byte_cap_param_overrides_default() -> None:
    body = "abc\n" * 1000  # 4_000 bytes
    out = _truncate(body, byte_cap=100)
    assert "WARNING" in out
    assert "100" in out


def test_f03_007_primary_memory_load_path_caps_oversize_aura_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "AURA.md").write_text("y" * 30_000)

    result = pm.load_project_memory(cwd)
    assert "WARNING" in result
    assert "30000" in result
    assert "25000" in result


# ---------------------------------------------------------------------------
# F-03-008 — @imports extension allowlist
# ---------------------------------------------------------------------------


def test_f03_008_exe_import_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "evil.exe").write_bytes(b"MZ\x90\x00malicious-binary")
    (cwd / "AURA.md").write_text("before\n@./evil.exe\nafter")

    result = pm.load_project_memory(cwd)
    # The @-line is dropped and the binary's bytes never enter the prompt.
    assert "malicious-binary" not in result
    assert "MZ" not in result
    assert "@./evil.exe" not in result
    assert result == "before\nafter"


def test_f03_008_md_import_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "child.md").write_text("CHILD")
    (cwd / "AURA.md").write_text("@./child.md")
    assert pm.load_project_memory(cwd) == "CHILD"


def test_f03_008_journal_event_emitted_for_rejected_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()

    journal_path = tmp_path / "events.jsonl"
    journal_module.configure(journal_path)

    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (cwd / "AURA.md").write_text("@./image.png\n")

    pm.load_project_memory(cwd)

    journal_module.reset()
    events = [
        json.loads(line)
        for line in journal_path.read_text().splitlines()
        if line.strip()
    ]
    rejected = [e for e in events if e.get("event") == "import_non_text_skipped"]
    assert rejected, f"expected import_non_text_skipped, got {events!r}"
    assert rejected[0]["suffix"] == ".png"


def test_f03_008_allowlist_covers_common_text_extensions(tmp_path: Path) -> None:
    # Spot-check: each whitelisted suffix resolves; non-whitelisted does not.
    base = tmp_path
    for suffix in (".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml",
                   ".sh", ".cfg", ".ini"):
        f = base / f"x{suffix}"
        f.write_text("x")
        assert pm._resolve_import(f.name, base) is not None, suffix
    bad = base / "x.exe"
    bad.write_text("x")
    assert pm._resolve_import("x.exe", base) is None


def _ensure_rule_record(tmp_path: Path) -> Rule:
    # Sanity construction; mainly to keep `Rule` exported on import path.
    return Rule(
        source_path=tmp_path / "r.md",
        base_dir=tmp_path,
        globs=(),
        content="",
    )


def test_module_imports_rule_dataclass(tmp_path: Path) -> None:
    assert isinstance(_ensure_rule_record(tmp_path), Rule)
