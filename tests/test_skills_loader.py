"""Tests for aura.core.skills.loader — directory-per-skill (claude-code v2.1.88 format).

Covers:
- Directory-per-skill discovery: ``<root>/.aura/skills/<name>/SKILL.md``.
- Frontmatter parsing: name override, when_to_use, allowed-tools (list+str),
  arguments (list+str), argument-hint, version, paths, user-invocable,
  disable-model-invocation.
- Layer precedence: user wins over project on name collision.
- Walk-up: project layer collects ``.aura/skills/`` from cwd up to home exclusive.
- Realpath dedup across symlinks.
- Conditional skills activate lazily via ``activate_conditional_skills_for_paths``.
- Legacy plain ``.md`` files → journal ``skill_legacy_format_detected``, NOT loaded.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from aura.core.persistence import journal as journal_module
from aura.core.skills.loader import (
    activate_conditional_skills_for_paths,
    clear_conditional_state,
    get_conditional_skills,
    load_skills,
    render_skill_body,
)


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


@pytest.fixture(autouse=True)
def _reset_conditional_state() -> Iterator[None]:
    """Conditional-skill state is module-global; reset between tests."""
    clear_conditional_state()
    yield
    clear_conditional_state()


# ---------------------------------------------------------------------------
# Basic discovery
# ---------------------------------------------------------------------------


def test_user_skill_dir_layout_loads(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "summarize" / "SKILL.md",
        "---\ndescription: Summarize the file.\n---\n# Body\nbody-text\n",
    )

    reg = load_skills(cwd=cwd, home=home)
    skills = reg.list()
    assert len(skills) == 1
    skill = skills[0]
    # ``name`` derives from the directory name when frontmatter doesn't override.
    assert skill.name == "summarize"
    assert skill.description == "Summarize the file."
    assert skill.layer == "user"
    assert "# Body" in skill.body
    assert "body-text" in skill.body
    assert skill.source_path == (
        home / ".aura" / "skills" / "summarize" / "SKILL.md"
    ).resolve()
    assert skill.base_dir == (home / ".aura" / "skills" / "summarize").resolve()


def test_project_skill_dir_layout_loads(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "refactor" / "SKILL.md",
        "---\ndescription: Project-level refactor helper.\n---\nP-BODY\n",
    )

    reg = load_skills(cwd=cwd, home=home)
    skills = reg.list()
    assert len(skills) == 1
    assert skills[0].name == "refactor"
    assert skills[0].layer == "project"
    assert "P-BODY" in skills[0].body


# ---------------------------------------------------------------------------
# Frontmatter — all supported fields
# ---------------------------------------------------------------------------


def test_frontmatter_all_fields_roundtrip(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "kitchen-sink" / "SKILL.md",
        (
            "---\n"
            "name: my-skill\n"
            "description: The full enchilada.\n"
            "when_to_use: When testing every field at once.\n"
            "allowed-tools:\n"
            "  - bash\n"
            "  - read_file\n"
            "arguments:\n"
            "  - target\n"
            "  - mode\n"
            "argument-hint: <target> <mode>\n"
            "version: 1.2.3\n"
            "user-invocable: true\n"
            "disable-model-invocation: false\n"
            "---\n"
            "Body for ${target} in ${mode}.\n"
        ),
    )

    reg = load_skills(cwd=cwd, home=home)
    skills = reg.list()
    assert len(skills) == 1
    s = skills[0]
    # ``name`` frontmatter override wins over dir name.
    assert s.name == "my-skill"
    assert s.description == "The full enchilada."
    assert s.when_to_use == "When testing every field at once."
    assert s.allowed_tools == frozenset({"bash", "read_file"})
    assert s.arguments == ("target", "mode")
    assert s.argument_hint == "<target> <mode>"
    assert s.version == "1.2.3"
    assert s.user_invocable is True
    assert s.disable_model_invocation is False
    assert s.paths == frozenset()
    assert s.is_conditional() is False


def test_scalar_arguments_split_on_whitespace(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "x" / "SKILL.md",
        "---\ndescription: x\narguments: foo bar baz\n---\nbody\n",
    )

    reg = load_skills(cwd=cwd, home=home)
    skill = reg.list()[0]
    # Matches claude-code's ``parseArgumentNames`` scalar-form behaviour.
    assert skill.arguments == ("foo", "bar", "baz")


def test_allowed_tools_scalar_form(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "y" / "SKILL.md",
        "---\ndescription: y\nallowed-tools: bash read_file edit_file\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.list()[0]
    assert skill.allowed_tools == frozenset({"bash", "read_file", "edit_file"})


# ---------------------------------------------------------------------------
# Collisions & layering
# ---------------------------------------------------------------------------


def test_user_wins_on_name_collision(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    _write(
        home / ".aura" / "skills" / "dup" / "SKILL.md",
        "---\ndescription: USER-VERSION\n---\nuser-body\n",
    )
    _write(
        cwd / ".aura" / "skills" / "dup" / "SKILL.md",
        "---\ndescription: PROJECT-VERSION\n---\nproj-body\n",
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


def test_project_walks_up_to_home_exclusive(tmp_path: Path) -> None:
    """Project layer collects ``.aura/skills/`` from every dir cwd → home (exclusive)."""
    home = tmp_path / "home"
    home.mkdir()
    # Nested project structure: home/outer/mid/inner (cwd).
    outer = home / "outer"
    mid = outer / "mid"
    inner = mid / "inner"
    inner.mkdir(parents=True)
    _write(
        outer / ".aura" / "skills" / "outer-skill" / "SKILL.md",
        "---\ndescription: outer.\n---\nOUTER\n",
    )
    _write(
        mid / ".aura" / "skills" / "mid-skill" / "SKILL.md",
        "---\ndescription: mid.\n---\nMID\n",
    )
    _write(
        inner / ".aura" / "skills" / "inner-skill" / "SKILL.md",
        "---\ndescription: inner.\n---\nINNER\n",
    )
    # Home itself has a skill dir — should be user-layer, not project-layer
    # (walk-up is home-exclusive).
    _write(
        home / ".aura" / "skills" / "home-skill" / "SKILL.md",
        "---\ndescription: home.\n---\nHOME\n",
    )

    reg = load_skills(cwd=inner, home=home)
    names = {s.name: s.layer for s in reg.list()}
    assert names == {
        "home-skill": "user",
        "outer-skill": "project",
        "mid-skill": "project",
        "inner-skill": "project",
    }


def test_outer_project_wins_on_collision_against_inner(tmp_path: Path) -> None:
    """Outer-project skill survives when an inner dir declares the same name.

    Matches "first-seen wins" dedup order: outer layer is scanned first (the
    loader reverses the walk so outermost comes first), so the inner skill
    loses. The user can still override via ~/.aura/skills/.
    """
    home = tmp_path / "home"
    home.mkdir()
    outer = home / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    _write(
        outer / ".aura" / "skills" / "shared" / "SKILL.md",
        "---\ndescription: OUTER.\n---\nO\n",
    )
    _write(
        inner / ".aura" / "skills" / "shared" / "SKILL.md",
        "---\ndescription: INNER.\n---\nI\n",
    )
    reg = load_skills(cwd=inner, home=home)
    skill = reg.get("shared")
    assert skill is not None
    assert skill.description == "OUTER."


# ---------------------------------------------------------------------------
# Realpath dedup
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="symlink permissions flaky on Windows")
def test_realpath_dedup_via_symlink(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    # Real skill in the user layer.
    _write(
        home / ".aura" / "skills" / "real" / "SKILL.md",
        "---\ndescription: Real skill.\n---\nREAL-BODY\n",
    )
    # Symlink in the project layer pointing at the same dir.
    (cwd / ".aura" / "skills").mkdir(parents=True)
    link = cwd / ".aura" / "skills" / "alias"
    link.symlink_to(home / ".aura" / "skills" / "real")

    reg = load_skills(cwd=cwd, home=home)
    skills = reg.list()
    # Exactly one — the symlinked path collapsed to the same realpath.
    assert len(skills) == 1
    assert skills[0].description == "Real skill."


# ---------------------------------------------------------------------------
# Conditional skills
# ---------------------------------------------------------------------------


def test_conditional_skill_not_in_registry_at_load(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "pyhelp" / "SKILL.md",
        "---\ndescription: Python helper.\npaths:\n  - 'src/**'\n---\nPY-BODY\n",
    )

    reg = load_skills(cwd=cwd, home=home)
    # Conditional → not active yet.
    assert reg.list() == []
    assert [s.name for s in get_conditional_skills()] == ["pyhelp"]


def test_activate_conditional_skill_by_matching_path(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "pyhelp" / "SKILL.md",
        "---\ndescription: Python helper.\npaths:\n  - 'src/**'\n---\nPY\n",
    )
    # Need an actual file inside cwd/src so ``resolve()`` behaves.
    _write(cwd / "src" / "foo.py", "# stub\n")

    load_skills(cwd=cwd, home=home)
    activated = activate_conditional_skills_for_paths(["src/foo.py"], cwd=cwd)
    assert activated == ["pyhelp"]
    # After activation, a re-load of the same cwd pulls the skill into the
    # active registry (not the conditional bucket) — because the loader
    # consults ``_activated_conditional_names`` on the way in.
    reg = load_skills(cwd=cwd, home=home)
    assert [s.name for s in reg.list()] == ["pyhelp"]


def test_activate_no_match_keeps_skill_conditional(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "pyhelp" / "SKILL.md",
        "---\ndescription: Python helper.\npaths:\n  - 'src/**'\n---\nb\n",
    )
    _write(cwd / "tests" / "foo.py", "# stub\n")

    load_skills(cwd=cwd, home=home)
    activated = activate_conditional_skills_for_paths(["tests/foo.py"], cwd=cwd)
    assert activated == []
    assert [s.name for s in get_conditional_skills()] == ["pyhelp"]


# ---------------------------------------------------------------------------
# Legacy format migration
# ---------------------------------------------------------------------------


def test_plain_md_at_top_level_not_loaded_journals_event(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    legacy = cwd / ".aura" / "skills" / "legacy.md"
    _write(
        legacy,
        "---\nname: legacy\ndescription: Old flat format.\n---\nL-BODY\n",
    )
    # Valid directory-format skill alongside it — should still load.
    _write(
        cwd / ".aura" / "skills" / "new" / "SKILL.md",
        "---\ndescription: New format.\n---\nN-BODY\n",
    )

    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()

    # Legacy file silently ignored; new-format skill loads.
    assert [s.name for s in reg.list()] == ["new"]
    legacy_events = [
        e for e in _events(log) if e["event"] == "skill_legacy_format_detected"
    ]
    assert len(legacy_events) == 1
    assert legacy_events[0]["layer"] == "project"
    files = legacy_events[0]["files"]
    assert isinstance(files, list)
    assert any(str(legacy) in f for f in files)


# ---------------------------------------------------------------------------
# Rendering / variable substitution
# ---------------------------------------------------------------------------


def test_render_skill_body_substitutes_skill_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    skill_dir = cwd / ".aura" / "skills" / "example"
    _write(
        skill_dir / "SKILL.md",
        "---\ndescription: Example.\n---\nSee ${AURA_SKILL_DIR}/examples/foo.\n",
    )
    _write(skill_dir / "examples" / "foo", "data\n")

    reg = load_skills(cwd=cwd, home=home)
    skill = reg.list()[0]
    out = render_skill_body(skill, session_id="abc123")
    assert str(skill.base_dir) in out
    assert "${AURA_SKILL_DIR}" not in out


def test_render_skill_body_substitutes_arguments(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "greet" / "SKILL.md",
        "---\ndescription: Greet.\narguments:\n  - who\n  - how\n---\nHello ${who}, ${how}.\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.list()[0]
    out = render_skill_body(skill, session_id="sid", argument_values=["alice", "warmly"])
    assert "Hello alice, warmly." in out


def test_render_skill_body_substitutes_session_id(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "sid" / "SKILL.md",
        "---\ndescription: Sid.\n---\nSession: ${AURA_SESSION_ID}.\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.list()[0]
    out = render_skill_body(skill, session_id="my-session-abc")
    assert "Session: my-session-abc." in out


# ---------------------------------------------------------------------------
# Error handling — missing fields, broken YAML
# ---------------------------------------------------------------------------


def test_missing_description_silent_skip_journal(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    bad = cwd / ".aura" / "skills" / "nodesc" / "SKILL.md"
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


def test_broken_yaml_silent_skip_journal(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "bad" / "SKILL.md",
        "---\nname: [broken\ndescription: x\n---\nbody\n",
    )
    _write(
        cwd / ".aura" / "skills" / "ok" / "SKILL.md",
        "---\ndescription: ok desc\n---\nok-body\n",
    )

    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()

    names = {s.name for s in reg.list()}
    assert names == {"ok"}
    assert any(
        e["event"] == "skill_parse_failed" for e in _events(log)
    )


def test_empty_skill_dir_is_skipped_silently(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    # Skill dir with no SKILL.md — e.g. bare ``examples/`` left behind.
    (cwd / ".aura" / "skills" / "empty").mkdir(parents=True)
    reg = load_skills(cwd=cwd, home=home)
    assert reg.list() == []


def test_home_defaults_to_path_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_home = tmp_path / "fake_home"
    _write(
        fake_home / ".aura" / "skills" / "from-home" / "SKILL.md",
        "---\ndescription: d\n---\nb\n",
    )
    cwd = tmp_path / "proj"
    cwd.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    reg = load_skills(cwd=cwd)
    names = {s.name for s in reg.list()}
    assert names == {"from-home"}


# ---------------------------------------------------------------------------
# Claude-code compat (V12-G) — load ``~/.claude/skills/<name>/SKILL.md``
# verbatim, accept both ``${CLAUDE_*}`` and ``${AURA_*}`` placeholders, and
# surface a journal warning on inline ``!`cmd` `` usage (unsupported here).
# ---------------------------------------------------------------------------


def test_claude_code_skills_dir_loads(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    # Drop a vanilla claude-code skill under ~/.claude/skills/ — same
    # dir-per-skill convention, frontmatter identical to what claude-code
    # authors write.
    _write(
        home / ".claude" / "skills" / "imported-skill" / "SKILL.md",
        "---\ndescription: imported from claude-code\n---\nbody here\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    names = {s.name for s in reg.list()}
    assert "imported-skill" in names


def test_claude_and_aura_skill_dirs_merge_without_double_load(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "aura-native" / "SKILL.md",
        "---\ndescription: a\n---\nb\n",
    )
    _write(
        home / ".claude" / "skills" / "claude-native" / "SKILL.md",
        "---\ndescription: c\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    names = {s.name for s in reg.list()}
    assert names == {"aura-native", "claude-native"}


def test_aura_wins_over_claude_on_name_collision(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "shared" / "SKILL.md",
        "---\ndescription: aura version\n---\nb\n",
    )
    _write(
        home / ".claude" / "skills" / "shared" / "SKILL.md",
        "---\ndescription: claude version\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    winner = reg.get("shared")
    assert winner is not None
    # Aura-native layer is scanned first → first-seen-wins keeps it.
    assert winner.description == "aura version"


def test_render_skill_body_substitutes_claude_skill_dir_namespace(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "SKILL.md",
        "---\ndescription: d\n---\n"
        "aura: ${AURA_SKILL_DIR}/x\n"
        "claude: ${CLAUDE_SKILL_DIR}/y\n",
    )
    from aura.core.skills.types import Skill
    skill = Skill(
        name="n",
        description="d",
        body=(
            "aura: ${AURA_SKILL_DIR}/x\n"
            "claude: ${CLAUDE_SKILL_DIR}/y\n"
        ),
        source_path=tmp_path / "SKILL.md",
        layer="user",
    )
    out = render_skill_body(skill, session_id="sess")
    assert f"aura: {tmp_path}/x" in out
    assert f"claude: {tmp_path}/y" in out


def test_render_skill_body_substitutes_claude_session_id_namespace(
    tmp_path: Path,
) -> None:
    from aura.core.skills.types import Skill
    skill = Skill(
        name="n",
        description="d",
        body=(
            "aura: ${AURA_SESSION_ID}\n"
            "claude: ${CLAUDE_SESSION_ID}\n"
        ),
        source_path=tmp_path / "SKILL.md",
        layer="user",
    )
    out = render_skill_body(skill, session_id="session-abc")
    assert "aura: session-abc" in out
    assert "claude: session-abc" in out


def test_inline_cmd_in_body_emits_journal_warning(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        _write(
            home / ".aura" / "skills" / "uses-inline-cmd" / "SKILL.md",
            # Body references inline shell-exec syntax — claude-code would
            # expand it, Aura renders literally + journals the mismatch.
            "---\ndescription: uses !`date`\n---\n"
            "Today's date: !`date +%Y-%m-%d`\n",
        )
        load_skills(cwd=cwd, home=home)
        events = _events(log)
        warnings = [
            e for e in events if e["event"] == "skill_inline_cmd_unsupported"
        ]
        assert len(warnings) == 1
        assert warnings[0]["name"] == "uses-inline-cmd"
        assert warnings[0]["layer"] == "user"
    finally:
        journal_module.reset()
