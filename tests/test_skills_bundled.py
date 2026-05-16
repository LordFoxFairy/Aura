"""F-0910-011 — bundled (managed-layer) skills shipped with Aura.

Decision: ship 3 real bundled skills (verify / simplify / code-review)
under ``aura/resources/skills/``. The loader prepends them to the load order,
so bundled/managed skills win on same-name collisions under Aura's current
first-writer-wins policy. ``include_bundled=True`` opts in.
"""

from __future__ import annotations

import sys
import zipfile
from collections.abc import Iterator
from importlib import resources as pkg_resources
from pathlib import Path

import pytest

from aura.capabilities.skills_runtime import loader
from aura.capabilities.skills_runtime.loader import (
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


def test_bundled_skills_load_from_package_resources(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    cwd.mkdir()

    expected_root = Path(str(pkg_resources.files("aura.resources").joinpath("skills"))).resolve()

    reg = load_skills(cwd=cwd, home=home, include_bundled=True)
    bundled = [s for s in reg.list() if s.name in {"verify", "simplify", "code-review"}]
    assert len(bundled) == 3
    for skill in bundled:
        assert expected_root in skill.source_path.parents


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


def test_bundled_skills_root_supports_zip_backed_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg_root = tmp_path / "zip_pkg_src"
    skills_dir = pkg_root / "zipskills_pkg" / "skills" / "zip-skill"
    skills_dir.mkdir(parents=True)
    _write(pkg_root / "zipskills_pkg" / "__init__.py", "")
    _write(pkg_root / "zipskills_pkg" / "skills" / "__init__.py", "")
    _write(
        skills_dir / "SKILL.md",
        "---\ndescription: zip backed skill\n---\nZIP-SKILL-BODY\n",
    )

    archive_path = tmp_path / "zipskills_pkg.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        for path in pkg_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(pkg_root).as_posix())

    monkeypatch.syspath_prepend(str(archive_path))
    sys.modules.pop("zipskills_pkg", None)
    sys.modules.pop("zipskills_pkg.skills", None)

    monkeypatch.setattr(loader, "_BUNDLED_SKILLS_PACKAGE", "zipskills_pkg")

    with loader._bundled_skills_root() as bundled_root:
        assert bundled_root is not None
        assert bundled_root.is_dir()
        assert (bundled_root / "zip-skill" / "SKILL.md").is_file()
