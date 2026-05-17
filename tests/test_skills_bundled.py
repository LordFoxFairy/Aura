"""F-0910-011 — bundled (managed-layer) skills shipped with Aura.

Decision: ship 3 real bundled skills (verify / simplify / code-review)
from the packaged namespace ``aura/plugins/skills/``. At runtime they are
materialized into a dedicated hidden root under ``~/.aura/plugins/.../skills``
so the active skill catalogue is skill-centric and detached from package
layout. ``include_bundled=True`` opts in.
"""

from __future__ import annotations

import sys
import zipfile
from collections.abc import Iterator
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
        assert skill.source_path.parent.parent.parent.name == "aura.plugins.skills"
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


def _install_zip_backed_bundled_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg_root = tmp_path / "zip_pkg_src"
    skills_dir = pkg_root / "zipskills_pkg" / "plugins" / "skills" / "zip-skill"
    skills_dir.mkdir(parents=True)
    _write(pkg_root / "zipskills_pkg" / "__init__.py", "")
    _write(pkg_root / "zipskills_pkg" / "plugins" / "__init__.py", "")
    _write(pkg_root / "zipskills_pkg" / "plugins" / "skills" / "__init__.py", "")
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
    sys.modules.pop("zipskills_pkg.plugins", None)
    sys.modules.pop("zipskills_pkg.plugins.skills", None)
    monkeypatch.setattr(loader, "_BUNDLED_SKILLS_PACKAGE", "zipskills_pkg.plugins.skills")


def test_bundled_skills_root_supports_zip_backed_dedicated_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_zip_backed_bundled_package(tmp_path, monkeypatch)

    with loader._bundled_skills_root(home_dir=tmp_path / "home") as bundled_root:
        assert bundled_root is not None
        assert bundled_root.is_dir()
        assert bundled_root.name == "skills"
        assert (bundled_root / "zip-skill" / "SKILL.md").is_file()


def test_bundled_skills_root_reuses_session_extraction_for_zip_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_zip_backed_bundled_package(tmp_path, monkeypatch)

    with loader._bundled_skills_root(home_dir=tmp_path / "home") as first_root:
        assert first_root is not None
        first_path = first_root
    assert first_path.is_dir()

    with loader._bundled_skills_root(home_dir=tmp_path / "home") as second_root:
        assert second_root == first_path
        assert (second_root / "zip-skill" / "SKILL.md").is_file()
