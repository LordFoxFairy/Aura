"""Skill loader: dir-per-skill discovery, frontmatter, layering, walk-up, dedup."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from aura.domain.skill import Skill
from aura.infrastructure.persistence import journal as journal_module
from aura.infrastructure.skills.loader import (
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
    # Scalar form: whitespace-split into ordered argument names.
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


def test_claude_code_skills_dir_loads(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    # ~/.claude/skills/ uses the same dir-per-skill + frontmatter convention.
    _write(
        home / ".claude" / "skills" / "imported-skill" / "SKILL.md",
        "---\ndescription: imported skill\n---\nbody here\n",
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
    from aura.domain.skill import Skill
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
    from aura.domain.skill import Skill
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
            # Inline ``!`cmd``` shell-exec syntax: Aura renders literally and journals.
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


def _make_skill_with_body(body: str, tmp_path: Path) -> Skill:
    """Construct a Skill dataclass directly so render tests don't have to round-trip yaml."""
    return Skill(
        name="t",
        description="d",
        body=body,
        source_path=tmp_path / "SKILL.md",
        layer="user",
    )


def test_render_inline_cmd_replaced_with_inert_placeholder(tmp_path: Path) -> None:
    """A1: ``!`date` `` in body is wrapped in the inert ``[Aura: ...]`` placeholder.

    The original command text is preserved INSIDE the placeholder for context,
    but no longer reads as a standalone exec contract — model sees ``[Aura:
    inline shell not supported — original: !`date`]`` instead of bare
    ``!`date` ``.
    """
    body = "Today is !`date +%Y-%m-%d`. Done."
    skill = _make_skill_with_body(body, tmp_path)
    out = render_skill_body(skill, session_id="s")
    # Placeholder present — wraps the original syntax explicitly.
    assert "[Aura: inline shell not supported — original: !`date +%Y-%m-%d`]" in out
    # Standalone exec syntax (not enclosed by the placeholder) is gone.
    # We strip out the placeholder text and assert the residue has no leftover ``!`...```.
    residue = out.replace(
        "[Aura: inline shell not supported — original: !`date +%Y-%m-%d`]", ""
    )
    assert "!`" not in residue


def test_render_multiple_inline_cmds_each_replaced(tmp_path: Path) -> None:
    """A1: each ``!`cmd` `` occurrence is independently sanitised."""
    skill = _make_skill_with_body("a !`pwd` && !`whoami` end.", tmp_path)
    out = render_skill_body(skill, session_id="s")
    # Both placeholders present.
    assert "[Aura: inline shell not supported — original: !`pwd`]" in out
    assert "[Aura: inline shell not supported — original: !`whoami`]" in out
    # No standalone exec syntax outside the placeholders.
    residue = (
        out.replace("[Aura: inline shell not supported — original: !`pwd`]", "")
        .replace("[Aura: inline shell not supported — original: !`whoami`]", "")
    )
    assert "!`" not in residue


def test_render_no_inline_cmd_passes_through(tmp_path: Path) -> None:
    """A1: bodies without the syntax are byte-identical (modulo other substitutions)."""
    body = "No shell here. Just prose with `inline code` and ${AURA_SESSION_ID}."
    skill = _make_skill_with_body(body, tmp_path)
    out = render_skill_body(skill, session_id="abc")
    assert "No shell here" in out
    assert "`inline code`" in out
    assert "abc" in out
    # Sanity: no spurious placeholders inserted.
    assert "[Aura:" not in out


def test_render_inline_cmd_inside_fenced_block_preserved(tmp_path: Path) -> None:
    """A1: ``!`cmd` `` inside a ``` fenced block is documentation — leave it alone."""
    body = (
        "Outside !`date` here.\n"
        "```\n"
        "Example: !`date` should NOT be touched in docs.\n"
        "```\n"
        "Outside again !`pwd` here.\n"
    )
    skill = _make_skill_with_body(body, tmp_path)
    out = render_skill_body(skill, session_id="s")
    # Outside-fence: substituted.
    assert out.count("[Aura: inline shell not supported") == 2
    assert "Outside !`date` here" not in out
    assert "Outside again !`pwd` here" not in out
    # Inside-fence: preserved verbatim.
    assert "Example: !`date` should NOT be touched in docs." in out


def test_render_skill_body_returns_helper_command_list(tmp_path: Path) -> None:
    """A1 helper: ``_sanitize_inline_cmds`` exposes the original commands for callers."""
    from aura.infrastructure.skills.loader import _sanitize_inline_cmds
    sanitized, originals = _sanitize_inline_cmds("a !`x` b !`y` c")
    # Placeholders inserted; originals captured in order.
    assert "[Aura: inline shell not supported — original: !`x`]" in sanitized
    assert "[Aura: inline shell not supported — original: !`y`]" in sanitized
    assert originals == ["x", "y"]


def test_unsupported_frontmatter_single_field_journals(tmp_path: Path) -> None:
    """A2: ``model:`` is parsed but not honored — one ``skill_unsupported_frontmatter`` event."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        _write(
            home / ".aura" / "skills" / "uses-model" / "SKILL.md",
            "---\ndescription: uses model\nmodel: claude-opus-4-7\n---\nbody\n",
        )
        load_skills(cwd=cwd, home=home)
        events = [
            e for e in _events(log)
            if e["event"] == "skill_unsupported_frontmatter"
        ]
        assert len(events) == 1
        assert events[0]["name"] == "uses-model"
        assert events[0]["layer"] == "user"
        assert events[0]["fields"] == ["model"]
    finally:
        journal_module.reset()


def test_unsupported_frontmatter_lists_all_dropped_fields(tmp_path: Path) -> None:
    """A2: multiple unsupported fields → single event whose ``fields`` lists them all."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        _write(
            home / ".aura" / "skills" / "kitchen-sink" / "SKILL.md",
            (
                "---\n"
                "description: kitchen sink\n"
                "model: opus\n"
                "context: fork\n"
                "effort: high\n"
                "---\nbody\n"
            ),
        )
        load_skills(cwd=cwd, home=home)
        events = [
            e for e in _events(log)
            if e["event"] == "skill_unsupported_frontmatter"
        ]
        assert len(events) == 1
        fields = events[0]["fields"]
        assert isinstance(fields, list)
        assert sorted(fields) == ["context", "effort", "model"]
    finally:
        journal_module.reset()


def test_unsupported_frontmatter_silent_when_only_recognized_fields(
    tmp_path: Path,
) -> None:
    """A2: skills using only known fields produce ZERO ``skill_unsupported_frontmatter`` events."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        _write(
            home / ".aura" / "skills" / "clean" / "SKILL.md",
            (
                "---\n"
                "name: clean\n"
                "description: only known fields\n"
                "when_to_use: always\n"
                "allowed-tools: [bash]\n"
                "version: 1.0\n"
                "---\nbody\n"
            ),
        )
        load_skills(cwd=cwd, home=home)
        events = [
            e for e in _events(log)
            if e["event"] == "skill_unsupported_frontmatter"
        ]
        assert events == []
    finally:
        journal_module.reset()


def test_integration_claude_code_skill_full_bad_shape(tmp_path: Path) -> None:
    """End-to-end load with every unsupported frontmatter field + inline shell-exec."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        _write(
            home / ".claude" / "skills" / "imported" / "SKILL.md",
            (
                "---\n"
                "description: imported skill\n"
                "model: claude-opus-4-7\n"
                "context: fork\n"
                "agent: subagent-delegate\n"
                "effort: high\n"
                "shell: bash\n"
                "hooks:\n"
                "  pre: notify\n"
                "allowed-tools: [read_file]\n"
                "---\n"
                "Today is !`date +%Y-%m-%d`. Try !`pwd`.\n"
            ),
        )
        reg = load_skills(cwd=cwd, home=home)
        events = _events(log)

        # 1. Skill loaded.
        skill = reg.get("imported")
        assert skill is not None
        assert skill.description == "imported skill"
        assert skill.allowed_tools == frozenset({"read_file"})

        # 2. ``skill_unsupported_frontmatter`` event fired with all 6 fields.
        unsupported_events = [
            e for e in events if e["event"] == "skill_unsupported_frontmatter"
        ]
        assert len(unsupported_events) == 1
        fields = unsupported_events[0]["fields"]
        assert isinstance(fields, list)
        assert sorted(fields) == [
            "agent", "context", "effort", "hooks", "model", "shell",
        ]
        assert unsupported_events[0]["name"] == "imported"
        assert unsupported_events[0]["layer"] == "user"

        # 3. ``skill_inline_cmd_unsupported`` still fires.
        inline_events = [
            e for e in events if e["event"] == "skill_inline_cmd_unsupported"
        ]
        assert len(inline_events) == 1
        assert inline_events[0]["name"] == "imported"

        # 4. Rendered body: inline shell-exec replaced with inert placeholder.
        rendered = render_skill_body(skill, session_id="test-session")
        assert (
            "[Aura: inline shell not supported — original: !`date +%Y-%m-%d`]"
            in rendered
        )
        assert "[Aura: inline shell not supported — original: !`pwd`]" in rendered
        # No standalone ``!``...`` outside the placeholders. We strip the
        # placeholders and assert the residue has no exec syntax left.
        residue = (
            rendered
            .replace(
                "[Aura: inline shell not supported — original: !`date +%Y-%m-%d`]",
                "",
            )
            .replace("[Aura: inline shell not supported — original: !`pwd`]", "")
        )
        assert "!`" not in residue
    finally:
        journal_module.reset()


# --- Bundled (managed) layer ------------------------------------------------


def test_include_bundled_materializes_managed_skills(tmp_path: Path) -> None:
    """Managed layer ships code-defined skills so a fresh install has a baseline catalogue."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    reg = load_skills(cwd=cwd, home=home, include_bundled=True)
    by_name = {s.name: s for s in reg.list()}
    # Code-defined bundled skills land in the registry at the managed layer.
    assert {"verify", "simplify", "code-review"} <= set(by_name)
    assert by_name["verify"].layer == "managed"


def test_bundled_user_collision_keeps_managed_first_seen(tmp_path: Path) -> None:
    """Managed loads first, so a user skill sharing its name loses (managed wins collisions)."""
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    cwd.mkdir()
    # User skill deliberately shadowing the bundled ``verify`` name.
    _write(
        home / ".aura" / "skills" / "verify" / "SKILL.md",
        "---\ndescription: USER-OVERRIDE-VERIFY\n---\nuser-body\n",
    )
    reg = load_skills(cwd=cwd, home=home, include_bundled=True)
    kept = reg.get("verify")
    assert kept is not None
    # Managed scanned first → first-seen-wins keeps the bundled description.
    assert kept.description != "USER-OVERRIDE-VERIFY"
    assert kept.layer == "managed"


def test_exclude_bundled_omits_managed_skills(tmp_path: Path) -> None:
    """Default (``include_bundled=False``) keeps the managed catalogue out of the registry."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    reg = load_skills(cwd=cwd, home=home)
    names = {s.name for s in reg.list()}
    # None of the bundled names leak in when bundling is off.
    assert names.isdisjoint({"verify", "simplify", "code-review"})


# --- restrict-tools ---------------------------------------------------------


def test_restrict_tools_list_form_parsed(tmp_path: Path) -> None:
    """Lease-scoped whitelist must survive load so a skill can clamp its tool surface."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "locked" / "SKILL.md",
        "---\ndescription: locked\nrestrict-tools:\n  - read_file\n  - grep\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("locked")
    assert skill is not None
    assert skill.restrict_tools == frozenset({"read_file", "grep"})


def test_restrict_tools_scalar_form_splits_on_whitespace(tmp_path: Path) -> None:
    """Scalar whitelist string splits like ``allowed-tools`` so authors can use either shape."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "locked2" / "SKILL.md",
        "---\ndescription: locked2\nrestrict-tools: read_file grep edit_file\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("locked2")
    assert skill is not None
    assert skill.restrict_tools == frozenset({"read_file", "grep", "edit_file"})
    # ``restrict-tools`` is recognized → no unsupported-frontmatter false positive.
    assert "restrict-tools" not in str(skill)


def test_restrict_tools_absent_defaults_empty(tmp_path: Path) -> None:
    """Absent whitelist means 'no restriction' — empty frozenset, not an implicit clamp."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "norestrict" / "SKILL.md",
        "---\ndescription: norestrict\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("norestrict")
    assert skill is not None
    assert skill.restrict_tools == frozenset()


# --- paths: '**' unconditional normalization --------------------------------


def test_paths_double_star_only_becomes_unconditional(tmp_path: Path) -> None:
    """A ``paths: ['**']`` skill matches everything, so it must register eagerly, not lazily."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "always" / "SKILL.md",
        "---\ndescription: always on\npaths:\n  - '**'\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("always")
    assert skill is not None
    # All-match collapses to unconditional → empty paths, eagerly registered.
    assert skill.paths == frozenset()
    assert skill.is_conditional() is False
    assert get_conditional_skills() == []


def test_paths_dir_glob_suffix_stripped_stays_conditional(tmp_path: Path) -> None:
    """``foo/**`` equals ``foo`` under pathspec, but a non-universal glob stays conditional."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "scoped" / "SKILL.md",
        "---\ndescription: scoped\npaths:\n  - 'src/**'\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    # Conditional → stashed, not in the active registry.
    assert reg.list() == []
    conditional = {s.name: s for s in get_conditional_skills()}
    assert "scoped" in conditional
    # ``/**`` suffix stripped to the bare directory.
    assert conditional["scoped"].paths == frozenset({"src"})


def test_paths_empty_after_filtering_stays_unconditional(tmp_path: Path) -> None:
    """A blank ``paths`` entry must not turn a skill conditional on noise alone."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "blankpath" / "SKILL.md",
        "---\ndescription: blankpath\npaths:\n  - ''\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("blankpath")
    assert skill is not None
    assert skill.paths == frozenset()
    assert skill.is_conditional() is False


# --- _split_frontmatter edge cases ------------------------------------------


def test_no_frontmatter_fence_skips_with_journal(tmp_path: Path) -> None:
    """A body with no leading ``---`` fence is not a skill — skip + journal, never crash."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "nofence" / "SKILL.md",
        "# Just a heading\nNo frontmatter at all.\n",
    )
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()
    assert reg.list() == []
    failed = [e for e in _events(log) if e["event"] == "skill_parse_failed"]
    assert len(failed) == 1
    assert failed[0]["error"] == "missing frontmatter"


def test_unterminated_frontmatter_fence_skips_with_journal(tmp_path: Path) -> None:
    """An opened-but-never-closed ``---`` fence is malformed — skip, don't eat the whole body."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "unterminated" / "SKILL.md",
        "---\ndescription: never closed\nbody continues forever\n",
    )
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()
    assert reg.list() == []
    failed = [e for e in _events(log) if e["event"] == "skill_parse_failed"]
    assert len(failed) == 1
    assert failed[0]["error"] == "missing frontmatter"


def test_empty_file_skips_with_journal(tmp_path: Path) -> None:
    """A zero-byte SKILL.md has no fence — skip silently, never index garbage."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(cwd / ".aura" / "skills" / "blankfile" / "SKILL.md", "")
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()
    assert reg.list() == []
    assert any(e["event"] == "skill_parse_failed" for e in _events(log))


def test_dots_terminator_closes_frontmatter(tmp_path: Path) -> None:
    """YAML's ``...`` document terminator must also close frontmatter, per the spec."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "dotsclose" / "SKILL.md",
        "---\ndescription: closed with dots\n...\nDOTS-BODY\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("dotsclose")
    assert skill is not None
    assert skill.description == "closed with dots"
    assert "DOTS-BODY" in skill.body


# --- frontmatter mapping / type guards --------------------------------------


def test_scalar_frontmatter_is_not_a_mapping_skips(tmp_path: Path) -> None:
    """A frontmatter that parses to a scalar (not a dict) is invalid — skip with journal."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "scalarfm" / "SKILL.md",
        "---\njust a bare string\n---\nbody\n",
    )
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()
    assert reg.list() == []
    failed = [e for e in _events(log) if e["event"] == "skill_parse_failed"]
    assert len(failed) == 1
    assert failed[0]["error"] == "frontmatter is not a mapping"


def test_empty_frontmatter_block_is_null_not_mapping_skips(tmp_path: Path) -> None:
    """An empty ``---``/``---`` block yields YAML null → treated as non-mapping, skipped."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "nullfm" / "SKILL.md",
        "---\n---\nbody\n",
    )
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()
    assert reg.list() == []
    failed = [e for e in _events(log) if e["event"] == "skill_parse_failed"]
    assert len(failed) == 1
    assert failed[0]["error"] == "frontmatter is not a mapping"


def test_description_wrong_type_skips(tmp_path: Path) -> None:
    """A non-string ``description`` (e.g. a list) fails the contract — skip, don't coerce."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "listdesc" / "SKILL.md",
        "---\ndescription:\n  - not\n  - a\n  - string\n---\nbody\n",
    )
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()
    assert reg.list() == []
    failed = [e for e in _events(log) if e["event"] == "skill_parse_failed"]
    assert len(failed) == 1
    assert "description" in str(failed[0]["error"])


def test_whitespace_only_description_skips(tmp_path: Path) -> None:
    """A whitespace-only description is effectively empty — must fail the non-empty contract."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "wsdesc" / "SKILL.md",
        '---\ndescription: "   "\n---\nbody\n',
    )
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=cwd, home=home)
    finally:
        journal_module.reset()
    assert reg.list() == []
    assert any(e["event"] == "skill_parse_failed" for e in _events(log))


# --- name override fallbacks ------------------------------------------------


def test_blank_name_override_falls_back_to_dir_name(tmp_path: Path) -> None:
    """An empty/whitespace ``name:`` must fall back to the dir name, not produce a blank id."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "dirname-skill" / "SKILL.md",
        '---\nname: "   "\ndescription: blank name override\n---\nb\n',
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("dirname-skill")
    assert skill is not None
    assert skill.name == "dirname-skill"


def test_non_string_name_override_falls_back_to_dir_name(tmp_path: Path) -> None:
    """A numeric ``name:`` is not a valid id — fall back to the directory name."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "numeric-name" / "SKILL.md",
        "---\nname: 12345\ndescription: numeric name\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("numeric-name")
    assert skill is not None
    assert skill.name == "numeric-name"


# --- _coerce_bool loose parsing ---------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("0", False),
        ("TRUE", True),
        ("No", False),
    ],
)
def test_user_invocable_string_booleans(
    tmp_path: Path, raw: str, expected: bool,
) -> None:
    """Authors hand-edit YAML; loose ``true/yes/1`` parsing keeps boolean intent intact."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "boolskill" / "SKILL.md",
        f'---\ndescription: bool\nuser-invocable: "{raw}"\n---\nb\n',
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("boolskill")
    assert skill is not None
    assert skill.user_invocable is expected


def test_unrecognized_bool_string_falls_back_to_default(tmp_path: Path) -> None:
    """An un-parseable bool string must use the documented default, not crash or guess."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "weirdbool" / "SKILL.md",
        "---\ndescription: weird\ndisable-model-invocation: maybe\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("weirdbool")
    assert skill is not None
    # ``disable-model-invocation`` default is False; ``maybe`` is unparseable.
    assert skill.disable_model_invocation is False


def test_numeric_bool_value_falls_back_to_default(tmp_path: Path) -> None:
    """A non-bool numeric value for a flag is not coerced — it falls to the default."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "numbool" / "SKILL.md",
        "---\ndescription: numbool\nuser-invocable: 42\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("numbool")
    assert skill is not None
    # ``42`` is neither a bool nor a recognized bool-string → default True.
    assert skill.user_invocable is True


# --- _coerce_str_list_field type guards -------------------------------------


def test_dict_valued_list_field_yields_empty(tmp_path: Path) -> None:
    """A mapping where a string-list is expected is junk — coerce to empty, don't explode."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "dicttools" / "SKILL.md",
        "---\ndescription: dicttools\nallowed-tools:\n  a: 1\n  b: 2\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("dicttools")
    assert skill is not None
    # Dict is neither str nor list → coerced to empty whitelist.
    assert skill.allowed_tools == frozenset()


def test_list_with_non_string_items_drops_them(tmp_path: Path) -> None:
    """Mixed YAML lists keep only the real string tool names; ints/None are dropped."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        cwd / ".aura" / "skills" / "mixedtools" / "SKILL.md",
        "---\ndescription: mixed\nallowed-tools:\n  - bash\n  - 7\n  - null\n  - grep\n---\nb\n",
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("mixedtools")
    assert skill is not None
    assert skill.allowed_tools == frozenset({"bash", "grep"})


# --- _read_text failure paths -----------------------------------------------


def test_directory_named_skill_md_is_unreadable_skips(tmp_path: Path) -> None:
    """If SKILL.md exists but is not a regular file, treat it as unreadable and skip."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    # A *directory* named SKILL.md fails ``is_file()`` → silent skip.
    skill_dir = cwd / ".aura" / "skills" / "broken"
    (skill_dir / "SKILL.md").mkdir(parents=True)
    reg = load_skills(cwd=cwd, home=home)
    assert reg.list() == []


def test_invalid_utf8_bytes_decode_with_replacement(tmp_path: Path) -> None:
    """Corrupt bytes must decode lossily (errors=replace), never raise mid-discovery."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    skill_file = cwd / ".aura" / "skills" / "binary" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    # Valid frontmatter then an invalid UTF-8 byte in the body.
    skill_file.write_bytes(
        b"---\ndescription: has bad bytes\n---\nbody \xff end\n"
    )
    reg = load_skills(cwd=cwd, home=home)
    skill = reg.get("binary")
    assert skill is not None
    # The undecodable byte is replaced, body still loads.
    assert "�" in skill.body


# --- iterdir OSError resilience ---------------------------------------------


def test_iterdir_oserror_returns_empty_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permission failure listing a skills root must degrade to 'no skills', not crash."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "present" / "SKILL.md",
        "---\ndescription: present\n---\nb\n",
    )
    real_iterdir = Path.iterdir
    target = (home / ".aura" / "skills").resolve()

    def fake_iterdir(self: Path) -> Iterator[Path]:
        if self.resolve() == target:
            raise PermissionError("denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    reg = load_skills(cwd=cwd, home=home)
    # The unreadable user layer yields nothing instead of propagating.
    assert reg.list() == []


# --- render base_dir fallback -----------------------------------------------


def test_render_uses_source_parent_when_base_dir_none(tmp_path: Path) -> None:
    """A Skill built without ``base_dir`` still resolves ``${AURA_SKILL_DIR}`` via source parent."""
    parent = tmp_path / "elsewhere"
    parent.mkdir()
    skill = Skill(
        name="n",
        description="d",
        body="dir is ${AURA_SKILL_DIR} here.",
        source_path=parent / "SKILL.md",
        layer="user",
    )
    # __post_init__ defaults base_dir to source_path.parent.
    out = render_skill_body(skill, session_id="s")
    assert f"dir is {parent} here." in out


def test_render_missing_argument_substitutes_empty_string(tmp_path: Path) -> None:
    """Fewer values than declared args must blank the unfilled slots, not raise IndexError."""
    skill = Skill(
        name="n",
        description="d",
        body="A=${a} B=${b}",
        source_path=tmp_path / "SKILL.md",
        layer="user",
        arguments=("a", "b"),
    )
    out = render_skill_body(skill, session_id="s", argument_values=["first"])
    # Second placeholder collapses to empty; first is filled.
    assert out == "A=first B="


# --- idempotency ------------------------------------------------------------


def test_cwd_equals_home_does_not_double_load_user_layer(tmp_path: Path) -> None:
    """When cwd IS $HOME the project walk-up hits the user root — skip it, never double-load."""
    home = tmp_path / "home"
    _write(
        home / ".aura" / "skills" / "homeonly" / "SKILL.md",
        "---\ndescription: homeonly\n---\nb\n",
    )
    # cwd == home: the only project_dir is home itself, whose skills root equals
    # the user root, so the project pass must skip it (no duplicate event).
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        reg = load_skills(cwd=home, home=home)
    finally:
        journal_module.reset()
    assert [s.name for s in reg.list()] == ["homeonly"]
    skill = reg.get("homeonly")
    assert skill is not None
    # Loaded exactly once as the user layer — no project-layer re-scan/dedup event.
    assert skill.layer == "user"
    assert not any(e["event"] == "skill_duplicate_skipped" for e in _events(log))


def test_load_skills_is_idempotent_across_repeated_calls(tmp_path: Path) -> None:
    """Discovery is read-only: loading twice yields the same registry, no double-register."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _write(
        home / ".aura" / "skills" / "stable" / "SKILL.md",
        "---\ndescription: stable\n---\nb\n",
    )
    first = {s.name for s in load_skills(cwd=cwd, home=home).list()}
    second = {s.name for s in load_skills(cwd=cwd, home=home).list()}
    assert first == second == {"stable"}


def test_activate_conditional_twice_is_idempotent(tmp_path: Path) -> None:
    """Re-activating an already-matched conditional skill must not duplicate or thrash state."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "proj"
    _write(
        cwd / ".aura" / "skills" / "pyhelp" / "SKILL.md",
        "---\ndescription: Python helper.\npaths:\n  - 'src/**'\n---\nPY\n",
    )
    _write(cwd / "src" / "foo.py", "# stub\n")
    load_skills(cwd=cwd, home=home)
    first = activate_conditional_skills_for_paths(["src/foo.py"], cwd=cwd)
    second = activate_conditional_skills_for_paths(["src/foo.py"], cwd=cwd)
    assert first == ["pyhelp"]
    # Second activation reports no *newly* activated skills (already active).
    assert second == []
    reg = load_skills(cwd=cwd, home=home)
    assert [s.name for s in reg.list()] == ["pyhelp"]
