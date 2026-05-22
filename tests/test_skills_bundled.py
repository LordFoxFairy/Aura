"""F-0910-011 — bundled (managed-layer) skills shipped with Aura.

Decision: ship 3 real bundled skills (verify / simplify / code-review)
as code-defined bundled content. At runtime they are materialized into a
dedicated hidden root under ``~/.aura/plugins/.../skills`` so the active skill
catalogue is skill-centric and detached from package layout.
``include_bundled=True`` opts in.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aura.infrastructure.skills import loader
from aura.infrastructure.skills.loader import (
    clear_conditional_state,
    load_skills,
)


@pytest.fixture(autouse=True)
def _reset_conditional_state() -> Iterator[None]:
    clear_conditional_state()
    yield
    clear_conditional_state()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_bundled_skills_load_when_opted_in(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    cwd.mkdir()

    reg = load_skills(cwd=cwd, home=home, include_bundled=True)
    names = {s.name for s in reg.list()}
    assert {"verify", "simplify", "code-review"}.issubset(names)


def test_bundled_skills_carry_managed_layer_tag(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    cwd.mkdir()

    reg = load_skills(cwd=cwd, home=home, include_bundled=True)
    bundled = [s for s in reg.list() if s.name in {"verify", "simplify", "code-review"}]
    assert len(bundled) == 3
    for skill in bundled:
        assert skill.layer == "managed"


def test_bundled_skills_load_from_hidden_global_skills_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    cwd.mkdir()

    reg = load_skills(cwd=cwd, home=home, include_bundled=True)
    bundled = [s for s in reg.list() if s.name in {"verify", "simplify", "code-review"}]
    assert len(bundled) == 3
    for skill in bundled:
        rendered = skill.source_path.as_posix()
        assert "/.aura/plugins/bundled-skills/" in rendered
        assert skill.source_path.parent.name in {"verify", "simplify", "code-review"}
        assert skill.source_path.parent.parent.name == "skills"
        assert skill.source_path.parent.parent.parent.name == "aura-bundled-skills"
        assert "resources" not in skill.source_path.parts


def test_bundled_skills_first_writer_wins_over_user(tmp_path: Path) -> None:
    """First-writer-wins (claude-code parity) — bundled loads first so a
    user skill of the same name can NOT shadow it. This matches how the
    user layer wins over the project layer below it: outer wins."""
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "verify" / "SKILL.md",
        "---\ndescription: my custom verify.\n---\nUSER-VERIFY-BODY\n",
    )

    reg = load_skills(cwd=cwd, home=home, include_bundled=True)
    skill = reg.get("verify")
    assert skill is not None
    assert skill.layer == "managed"


def test_bundled_skills_disabled_by_default(tmp_path: Path) -> None:
    """The default ``include_bundled=False`` keeps existing test fixtures hermetic."""
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    cwd.mkdir()

    reg = load_skills(cwd=cwd, home=home)
    names = {s.name for s in reg.list()}
    assert names == set()


def test_bundled_skills_root_uses_hidden_global_plugins_dir(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"

    with loader._bundled_skills_root(home_dir=home) as bundled_root:
        assert bundled_root is not None
        assert bundled_root.is_dir()
        assert bundled_root.name == "skills"
        assert "/.aura/plugins/bundled-skills/" in bundled_root.as_posix()
        assert (bundled_root / "verify" / "SKILL.md").is_file()
        assert (bundled_root / "simplify" / "SKILL.md").is_file()
        assert (bundled_root / "code-review" / "SKILL.md").is_file()


def test_bundled_skills_root_reuses_session_extraction(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"

    with loader._bundled_skills_root(home_dir=home) as first_root:
        assert first_root is not None
        first_path = first_root
    assert first_path.is_dir()

    with loader._bundled_skills_root(home_dir=home) as second_root:
        assert second_root == first_path
        assert (second_root / "verify" / "SKILL.md").is_file()
