"""Tests for aura.core.skills.loader: discovery, frontmatter, journal, collisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aura.core import journal as journal_module
from aura.core.skills.loader import load_skills


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _events(log: Path) -> list[dict[str, object]]:
    if not log.exists():
        return []
    return [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_valid_user_skill_loads(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "summarize.md",
        "---\nname: summarize\ndescription: Summarize the file.\n---\n# Body\nbody-text\n",
    )

    reg = load_skills(cwd=cwd, home=home)
    skills = reg.list()
    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "summarize"
    assert skill.description == "Summarize the file."
    assert skill.layer == "user"
    assert "# Body" in skill.body
    assert "body-text" in skill.body
    assert skill.source_path == (home / ".aura" / "skills" / "summarize.md").resolve()


def test_valid_project_skill_loads(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "refactor.md",
        "---\nname: refactor\ndescription: Project-level refactor helper.\n---\nP-BODY\n",
    )

    reg = load_skills(cwd=cwd, home=home)
    skills = reg.list()
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "refactor"
    assert s.layer == "project"
    assert "P-BODY" in s.body


def test_user_wins_on_name_collision(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    _write(
        home / ".aura" / "skills" / "dup.md",
        "---\nname: dup\ndescription: USER-VERSION\n---\nuser-body\n",
    )
    _write(
        cwd / ".aura" / "skills" / "dup.md",
        "---\nname: dup\ndescription: PROJECT-VERSION\n---\nproj-body\n",
    )

    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()

    skills = reg.list()
    assert len(skills) == 1
    assert skills[0].description == "USER-VERSION"
    assert skills[0].layer == "user"

    matching = [e for e in _events(log) if e["event"] == "skill_name_collision"]
    assert len(matching) == 1
    assert matching[0]["name"] == "dup"


def test_missing_name_silent_skip_journal(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    bad = cwd / ".aura" / "skills" / "noname.md"
    _write(bad, "---\ndescription: no name here\n---\nbody\n")

    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()

    assert reg.list() == []
    matching = [e for e in _events(log) if e["event"] == "skill_parse_failed"]
    assert len(matching) == 1
    ev = matching[0]
    assert ev["path"] == str(bad.resolve()) or ev["path"] == str(bad)
    assert isinstance(ev["error"], str) and ev["error"]


def test_missing_description_silent_skip_journal(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    bad = cwd / ".aura" / "skills" / "nodesc.md"
    _write(bad, "---\nname: x\n---\nbody\n")

    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()

    assert reg.list() == []
    matching = [e for e in _events(log) if e["event"] == "skill_parse_failed"]
    assert len(matching) == 1


def test_non_md_files_ignored(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    skills_dir = cwd / ".aura" / "skills"
    _write(skills_dir / "keep.md", "---\nname: keep\ndescription: d\n---\nb\n")
    _write(skills_dir / "ignore.txt", "---\nname: ignore\ndescription: d\n---\nb\n")
    _write(skills_dir / "README", "no frontmatter\n")

    reg = load_skills(cwd=cwd, home=home)
    names = {s.name for s in reg.list()}
    assert names == {"keep"}


def test_broken_yaml_silent_skip_journal(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "bad.md",
        "---\nname: [broken\ndescription: x\n---\nbody\n",
    )
    _write(
        cwd / ".aura" / "skills" / "ok.md",
        "---\nname: ok\ndescription: ok desc\n---\nok-body\n",
    )

    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()

    names = {s.name for s in reg.list()}
    assert names == {"ok"}

    matching = [e for e in _events(log) if e["event"] == "skill_parse_failed"]
    assert len(matching) == 1


def test_body_preserved_as_markdown_raw(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    body = "# Heading\n\n- item1\n- item2\n\n```python\nprint('hi')\n```\n"
    _write(
        cwd / ".aura" / "skills" / "rich.md",
        f"---\nname: rich\ndescription: d\n---\n{body}",
    )

    reg = load_skills(cwd=cwd, home=home)
    skills = reg.list()
    assert len(skills) == 1
    assert "# Heading" in skills[0].body
    assert "- item1" in skills[0].body
    assert "```python" in skills[0].body


def test_home_defaults_to_path_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_home = tmp_path / "fake_home"
    _write(
        fake_home / ".aura" / "skills" / "from-home.md",
        "---\nname: from-home\ndescription: d\n---\nb\n",
    )
    cwd = tmp_path / "proj"
    cwd.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    reg = load_skills(cwd=cwd)
    names = {s.name for s in reg.list()}
    assert names == {"from-home"}


def test_not_recursive_subdirs_ignored(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    skills_dir = cwd / ".aura" / "skills"
    _write(skills_dir / "top.md", "---\nname: top\ndescription: d\n---\nb\n")
    _write(
        skills_dir / "sub" / "nested.md",
        "---\nname: nested\ndescription: d\n---\nb\n",
    )

    reg = load_skills(cwd=cwd, home=home)
    names = {s.name for s in reg.list()}
    assert names == {"top"}
