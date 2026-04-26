"""Tests for aura.core.permissions.store — settings.json load/save/round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aura.config.schema import AuraConfigError
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet
from aura.core.permissions.store import (
    PermissionStoreError,
    ensure_local_settings,
    load,
    load_ruleset,
    save_rule,
)
from aura.schemas.permissions import PermissionsConfig


def test_load_on_nonexistent_file_returns_defaults(tmp_path: Path) -> None:
    cfg = load(tmp_path)
    assert isinstance(cfg, PermissionsConfig)
    assert cfg.mode == "default"
    assert cfg.allow == []
    assert cfg.safety_exempt == []
    # Findings A + B defaults: 5-minute prompt timeout, disable_bypass off.
    assert cfg.prompt_timeout_sec == 300.0
    assert cfg.disable_bypass is False


def test_permissions_config_default_fields() -> None:
    # Sanity check the schema defaults directly (without going through
    # the store loader) so a drift on either side surfaces in the
    # right test.
    cfg = PermissionsConfig()
    assert cfg.prompt_timeout_sec == 300.0
    assert cfg.disable_bypass is False


def test_permissions_config_accepts_none_timeout() -> None:
    # ``None`` → "wait forever" (legacy). Lock both that it's accepted
    # and that round-trip keeps the value.
    cfg = PermissionsConfig(prompt_timeout_sec=None)
    assert cfg.prompt_timeout_sec is None


def test_permissions_config_accepts_custom_timeout_and_disable_bypass() -> None:
    cfg = PermissionsConfig(prompt_timeout_sec=30.0, disable_bypass=True)
    assert cfg.prompt_timeout_sec == 30.0
    assert cfg.disable_bypass is True


def test_load_parses_disable_bypass_and_timeout(tmp_path: Path) -> None:
    # End-to-end round trip through the store loader. Confirms
    # settings.json -> PermissionsConfig for the new fields.
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "permissions": {
            "disable_bypass": True,
            "prompt_timeout_sec": 45.5,
        },
    }))
    cfg = load(tmp_path)
    assert cfg.disable_bypass is True
    assert cfg.prompt_timeout_sec == 45.5


def test_load_round_trips_three_rules(tmp_path: Path) -> None:
    r1 = Rule(tool="bash", content="npm test")
    r2 = Rule(tool="read_file", content=None)
    r3 = Rule(tool="write_file", content="/tmp/scratch")
    save_rule(tmp_path, r1)
    save_rule(tmp_path, r2)
    save_rule(tmp_path, r3)
    cfg = load(tmp_path)
    assert cfg.allow == [r1.to_string(), r2.to_string(), r3.to_string()]


def test_load_with_malformed_json_raises_aura_config_error(tmp_path: Path) -> None:
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text("{not json")
    with pytest.raises(AuraConfigError) as exc_info:
        load(tmp_path)
    assert str(settings) in str(exc_info.value) or str(settings) in exc_info.value.source


def test_load_with_unknown_permissions_key_raises(tmp_path: Path) -> None:
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": {"xyz": 1}}))
    with pytest.raises(AuraConfigError):
        load(tmp_path)


def test_load_without_permissions_key_returns_defaults(tmp_path: Path) -> None:
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"other_section": {"x": 1}}))
    cfg = load(tmp_path)
    assert cfg.mode == "default"
    assert cfg.allow == []


def test_save_rule_creates_aura_dir_if_missing(tmp_path: Path) -> None:
    assert not (tmp_path / ".aura").exists()
    save_rule(tmp_path, Rule(tool="bash", content=None))
    assert (tmp_path / ".aura").is_dir()
    assert (tmp_path / ".aura" / "settings.json").is_file()


def test_save_rule_atomic_write_failure_raises_permission_store_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Invariant tested: on OSError from Path.replace, save_rule raises
    # PermissionStoreError and does not claim success.
    def _boom(self: Path, target: Path | str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", _boom)
    with pytest.raises(PermissionStoreError) as exc_info:
        save_rule(tmp_path, Rule(tool="bash", content=None))
    # Detail should convey the failure cause; source should name the settings path.
    assert "disk full" in exc_info.value.detail
    assert "settings.json" in exc_info.value.source


def test_save_rule_de_dupes(tmp_path: Path) -> None:
    rule = Rule(tool="bash", content="npm test")
    save_rule(tmp_path, rule)
    save_rule(tmp_path, rule)
    cfg = load(tmp_path)
    assert cfg.allow == [rule.to_string()]


def test_save_rule_preserves_unrelated_top_level_keys(tmp_path: Path) -> None:
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps({"permissions": {"allow": []}, "other_section": {"x": 1}})
    )
    save_rule(tmp_path, Rule(tool="bash", content=None))
    reloaded = json.loads(settings.read_text())
    assert reloaded["other_section"] == {"x": 1}
    assert reloaded["permissions"]["allow"] == ["bash"]


def test_load_ruleset_parses_allow_strings_into_rules(tmp_path: Path) -> None:
    save_rule(tmp_path, Rule(tool="bash", content="npm test"))
    save_rule(tmp_path, Rule(tool="read_file", content=None))
    ruleset = load_ruleset(tmp_path)
    assert isinstance(ruleset, RuleSet)
    assert ruleset.rules == (
        Rule(tool="bash", content="npm test"),
        Rule(tool="read_file", content=None),
    )


def test_load_ruleset_invalid_rule_string_raises_aura_config_error(
    tmp_path: Path,
) -> None:
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps({"permissions": {"allow": ["bash(unclosed"]}})
    )
    with pytest.raises(AuraConfigError) as exc_info:
        load_ruleset(tmp_path)
    assert "bash(unclosed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# settings.local.json — machine-local overrides (gitignored by convention)
# ---------------------------------------------------------------------------


def _write_settings(tmp_path: Path, name: str, perms: dict[str, object]) -> Path:
    p = tmp_path / ".aura" / name
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"permissions": perms}))
    return p


def test_load_local_alone_returns_local_rules(tmp_path: Path) -> None:
    # settings.json absent, settings.local.json present.
    _write_settings(tmp_path, "settings.local.json", {"allow": ["bash"]})
    cfg = load(tmp_path)
    assert cfg.allow == ["bash"]


def test_load_concatenates_project_and_local_allow_lists(tmp_path: Path) -> None:
    _write_settings(tmp_path, "settings.json", {"allow": ["read_file(/shared)"]})
    _write_settings(tmp_path, "settings.local.json", {"allow": ["bash(ssh prod)"]})
    cfg = load(tmp_path)
    # Project rules come first (team's canonical ones), local appended.
    assert cfg.allow == ["read_file(/shared)", "bash(ssh prod)"]


def test_load_local_mode_overrides_project_mode(tmp_path: Path) -> None:
    _write_settings(tmp_path, "settings.json", {"mode": "default"})
    _write_settings(tmp_path, "settings.local.json", {"mode": "bypass"})
    cfg = load(tmp_path)
    assert cfg.mode == "bypass"


def test_load_project_mode_kept_when_local_omits_mode(tmp_path: Path) -> None:
    _write_settings(tmp_path, "settings.json", {"mode": "bypass"})
    _write_settings(tmp_path, "settings.local.json", {"allow": ["bash"]})
    cfg = load(tmp_path)
    assert cfg.mode == "bypass"


def test_load_local_unknown_key_raises_with_local_path(tmp_path: Path) -> None:
    _write_settings(tmp_path, "settings.local.json", {"nope": 1})
    with pytest.raises(AuraConfigError) as exc:
        load(tmp_path)
    # The error must name settings.local.json so the user knows which file
    # has the typo, not just "settings.json".
    assert "settings.local.json" in str(exc.value)


def test_load_concatenates_safety_exempt_lists(tmp_path: Path) -> None:
    _write_settings(tmp_path, "settings.json", {"safety_exempt": ["shared/"]})
    _write_settings(tmp_path, "settings.local.json", {"safety_exempt": ["local/"]})
    cfg = load(tmp_path)
    assert cfg.safety_exempt == ["shared/", "local/"]


def test_load_ruleset_merges_project_and_local_rules(tmp_path: Path) -> None:
    _write_settings(tmp_path, "settings.json", {"allow": ["read_file(/shared)"]})
    _write_settings(tmp_path, "settings.local.json", {"allow": ["bash(npm test)"]})
    rs = load_ruleset(tmp_path)
    assert len(rs.rules) == 2
    assert rs.rules[0].tool == "read_file"
    assert rs.rules[1].tool == "bash"


def test_save_rule_default_scope_writes_the_rule_into_settings_json(
    tmp_path: Path,
) -> None:
    # save_rule is the default "remember for this project" path: the RULE
    # goes into settings.json, never settings.local.json. (The local file
    # may also get auto-created as an empty template alongside — that's
    # the "first-run discoverability" side-effect, covered separately.)
    save_rule(tmp_path, Rule(tool="bash", content="npm test"))
    project_file = tmp_path / ".aura" / "settings.json"
    assert project_file.is_file()
    content = json.loads(project_file.read_text())
    assert content["permissions"]["allow"] == ["bash(npm test)"]
    # The local file, if present, is the empty template — no rules in it.
    local_file = tmp_path / ".aura" / "settings.local.json"
    if local_file.exists():
        local = json.loads(local_file.read_text())
        assert local["permissions"]["allow"] == []


def test_save_rule_scope_local_writes_to_settings_local_json(tmp_path: Path) -> None:
    rule = Rule(tool="bash", content="ssh prod")
    save_rule(tmp_path, rule, scope="local")
    # Round-trip via load: local-scope rule should appear in merged allow list.
    cfg = load(tmp_path)
    assert rule.to_string() in cfg.allow
    # settings.local.json created; settings.json must not exist.
    assert (tmp_path / ".aura" / "settings.local.json").is_file()
    assert not (tmp_path / ".aura" / "settings.json").exists()


def test_save_rule_scope_project_writes_to_settings_json(tmp_path: Path) -> None:
    rule = Rule(tool="bash", content="npm test")
    save_rule(tmp_path, rule, scope="project")
    assert (tmp_path / ".aura" / "settings.json").is_file()
    # Post-2026-04-21: project-scope save also drops the empty local template
    # for first-run discoverability. The rule is NOT in the local file.
    local_file = tmp_path / ".aura" / "settings.local.json"
    if local_file.exists():
        local = json.loads(local_file.read_text())
        assert local["permissions"]["allow"] == []
    cfg = load(tmp_path)
    assert cfg.allow == [rule.to_string()]


def test_save_rule_local_scope_de_dupes(tmp_path: Path) -> None:
    rule = Rule(tool="bash", content="npm test")
    save_rule(tmp_path, rule, scope="local")
    save_rule(tmp_path, rule, scope="local")
    cfg = load(tmp_path)
    assert cfg.allow == [rule.to_string()]


def test_save_rule_local_scope_atomic_write_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(self: Path, target: Path | str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", _boom)
    with pytest.raises(PermissionStoreError) as exc_info:
        save_rule(tmp_path, Rule(tool="bash", content=None), scope="local")
    assert "disk full" in exc_info.value.detail
    assert "settings.local.json" in exc_info.value.source


def test_save_rule_scope_invalid_raises(tmp_path: Path) -> None:
    with pytest.raises((TypeError, ValueError)):
        save_rule(tmp_path, Rule(tool="bash", content=None), scope="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ensure_local_settings — startup init of the machine-local override file
# ---------------------------------------------------------------------------


def test_ensure_local_creates_file_with_template(tmp_path: Path) -> None:
    (tmp_path / ".aura").mkdir()  # user has opted into aura in this dir
    path, created = ensure_local_settings(tmp_path)
    assert created is True
    assert path == tmp_path / ".aura" / "settings.local.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["permissions"] == {"allow": []}
    # The `"//"` key is a conventional JSON comment: it documents the
    # file for a user opening it cold. It's a top-level sibling of
    # "permissions", which ``load`` tolerates (it only reads the
    # "permissions" key) and ``save_rule`` preserves (unrelated top-level
    # keys round-trip).
    assert "//" in data
    assert "bash(" in data["//"]  # Content includes an example rule


def test_save_rule_project_scope_also_creates_local_template(
    tmp_path: Path,
) -> None:
    # First-run path: user picks "always in project" → save_rule creates
    # ``.aura/`` AND drops the local template alongside. If we waited for
    # the next CLI startup to run ``ensure_local_settings``, the user would
    # only see ``settings.local.json`` the second time they ran aura.
    assert not (tmp_path / ".aura").exists()
    save_rule(tmp_path, Rule(tool="bash", content="npm test"), scope="project")
    assert (tmp_path / ".aura" / "settings.json").is_file()
    assert (tmp_path / ".aura" / "settings.local.json").is_file()


def test_save_rule_local_scope_does_not_disturb_project_file(
    tmp_path: Path,
) -> None:
    # Saving to local scope should NEVER create/touch settings.json.
    assert not (tmp_path / ".aura").exists()
    save_rule(tmp_path, Rule(tool="bash", content="ssh prod"), scope="local")
    assert (tmp_path / ".aura" / "settings.local.json").is_file()
    assert not (tmp_path / ".aura" / "settings.json").exists()


def test_ensure_local_template_roundtrips_through_save_rule(tmp_path: Path) -> None:
    # The "//" comment key must NOT be clobbered when save_rule later
    # appends a rule — save_rule preserves unrelated top-level keys, and
    # this is a test for that specific round-trip.
    from aura.core.permissions.store import save_rule
    (tmp_path / ".aura").mkdir()
    ensure_local_settings(tmp_path)
    save_rule(tmp_path, Rule(tool="bash", content="npm test"), scope="local")
    reloaded = json.loads((tmp_path / ".aura" / "settings.local.json").read_text())
    assert "//" in reloaded
    assert "bash(npm test)" in reloaded["permissions"]["allow"]


def test_ensure_local_is_noop_when_file_exists(tmp_path: Path) -> None:
    path = tmp_path / ".aura" / "settings.local.json"
    path.parent.mkdir()
    existing = {"permissions": {"allow": ["bash(existing)"]}}
    path.write_text(json.dumps(existing))
    returned, created = ensure_local_settings(tmp_path)
    assert returned == path
    assert created is False
    # Content must be untouched, not reset to the empty template.
    assert json.loads(path.read_text()) == existing


def test_ensure_local_noop_when_aura_dir_absent(tmp_path: Path) -> None:
    # Fresh tmp dir; no .aura/ → user hasn't opted in here.
    # ensure_local must NOT create the dir or the file — that would
    # pollute any directory where the user happens to run ``aura``.
    assert not (tmp_path / ".aura").exists()
    _, created = ensure_local_settings(tmp_path)
    assert created is False
    assert not (tmp_path / ".aura").exists()


def test_ensure_local_creates_file_when_aura_dir_exists(tmp_path: Path) -> None:
    # User has set up aura in this dir (has .aura/, maybe config.json).
    # ensure_local writes the template alongside.
    (tmp_path / ".aura").mkdir()
    path, created = ensure_local_settings(tmp_path)
    assert created is True
    assert path.exists()


def test_ensure_local_output_roundtrips_through_load(tmp_path: Path) -> None:
    (tmp_path / ".aura").mkdir()
    ensure_local_settings(tmp_path)
    cfg = load(tmp_path)
    # Template's empty allow list means no rules — load should see defaults.
    assert cfg.allow == []
    assert cfg.mode == "default"


# ---------------------------------------------------------------------------
# F-04-007 — known-tool-name validation at RuleSet load time
# ---------------------------------------------------------------------------


def test_load_ruleset_known_tools_skipped_by_default(tmp_path: Path) -> None:
    # Backward compat: omitting ``known_tool_names`` keeps the old behaviour
    # where any tool name parses cleanly.
    save_rule(tmp_path, Rule(tool="totally_made_up_tool", content=None))
    rs = load_ruleset(tmp_path)
    assert rs.rules[0].tool == "totally_made_up_tool"


def test_load_ruleset_unknown_tool_raises_aura_config_error(tmp_path: Path) -> None:
    save_rule(tmp_path, Rule(tool="unknown_tool", content="read"))
    with pytest.raises(AuraConfigError) as exc:
        load_ruleset(
            tmp_path,
            known_tool_names={"bash", "read_file", "write_file", "edit_file"},
        )
    msg = str(exc.value)
    assert "unknown_tool" in msg
    assert "settings.json" in msg


def test_load_ruleset_unknown_tool_suggests_closest_match(tmp_path: Path) -> None:
    save_rule(tmp_path, Rule(tool="read_fil", content=None))
    with pytest.raises(AuraConfigError) as exc:
        load_ruleset(
            tmp_path,
            known_tool_names={"bash", "read_file", "write_file", "edit_file"},
        )
    msg = str(exc.value)
    assert "read_fil" in msg
    assert "read_file" in msg


def test_load_ruleset_known_tool_passes(tmp_path: Path) -> None:
    save_rule(tmp_path, Rule(tool="bash", content="npm test"))
    save_rule(tmp_path, Rule(tool="read_file", content=None))
    rs = load_ruleset(
        tmp_path,
        known_tool_names={"bash", "read_file", "write_file"},
    )
    assert len(rs.rules) == 2


def test_load_ruleset_wildcard_rule_skips_validation(tmp_path: Path) -> None:
    # Wildcards (e.g. mcp__github__*) cover server surfaces that may not be
    # populated yet at load time; treat them as known.
    save_rule(tmp_path, Rule(tool="mcp__github__*", content=None))
    rs = load_ruleset(
        tmp_path,
        known_tool_names={"bash", "read_file"},
    )
    assert rs.rules[0].tool == "mcp__github__*"


def test_load_ruleset_wildcard_in_known_covers_concrete_rule(tmp_path: Path) -> None:
    # A wildcard registered in known_tool_names (an MCP server's whole
    # surface) must also satisfy a non-wildcard rule that matches it.
    save_rule(tmp_path, Rule(tool="mcp__github__create_issue", content=None))
    rs = load_ruleset(
        tmp_path,
        known_tool_names={"bash", "mcp__github__*"},
    )
    assert rs.rules[0].tool == "mcp__github__create_issue"
