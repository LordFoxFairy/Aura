"""F-0910-011 — bundled (managed-layer) skills shipped with Aura.

Decision: ship 3 real bundled skills (verify / simplify / code-review)
under ``aura/skills/bundled/``. The loader prepends them to the load order
so user / project layers can override by name. ``include_bundled=True``
opts in.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aura.core.skills.loader import (
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
