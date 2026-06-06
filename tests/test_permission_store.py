"""Tests for aura.infrastructure.permission_store — settings.json load/save/round-trip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aura.config.schema import AuraConfigError, PermissionsConfig
from aura.domain.permission.rule import Rule
from aura.domain.permission.session import RuleSet
from aura.infrastructure.permission_store import (
    PermissionStoreError,
    ensure_local_settings,
    load,
    load_ask_ruleset,
    load_deny_ruleset,
    load_ruleset,
    save_rule,
)


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
    bogus_scope: Any = "bogus"  # deliberately off-type to exercise path
    with pytest.raises((TypeError, ValueError)):
        save_rule(tmp_path, Rule(tool="bash", content=None), scope=bogus_scope)


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
    from aura.infrastructure.permission_store import save_rule
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


# --- malformed top-level shapes (Schema-crash boundary) -------------------


@pytest.mark.parametrize("payload", ["[1, 2, 3]", '"a string"', "42", "true", "null"])
def test_load_non_object_top_level_json_raises(tmp_path: Path, payload: str) -> None:
    # A settings file whose root is a JSON array/scalar is structurally
    # wrong; the loader must reject it (not silently coerce to {}), and
    # the error must name the offending file so the user can fix it.
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(payload)
    with pytest.raises(AuraConfigError) as exc:
        load(tmp_path)
    assert "object" in str(exc.value)
    assert "settings.json" in exc.value.source


@pytest.mark.parametrize("perms", ["[]", '"oops"', "5", "false"])
def test_load_permissions_not_object_raises(tmp_path: Path, perms: str) -> None:
    # ``permissions`` present but not an object is a typo we must surface,
    # not skip — otherwise rules silently vanish with no allow list.
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text('{"permissions": ' + perms + "}")
    with pytest.raises(AuraConfigError) as exc:
        load(tmp_path)
    assert "'permissions' must be an object" in str(exc.value)


def test_load_local_non_object_top_level_names_local_file(tmp_path: Path) -> None:
    # Same root-shape rejection on settings.local.json must point at the
    # LOCAL file, not the project one, so the typo is found in one read.
    local = tmp_path / ".aura" / "settings.local.json"
    local.parent.mkdir()
    local.write_text(json.dumps([1, 2]))
    with pytest.raises(AuraConfigError) as exc:
        load(tmp_path)
    assert "settings.local.json" in exc.value.source


# --- safety_exempt validation (cannot disarm a built-in protection) -------


@pytest.mark.parametrize("bad_glob", ["!", "\\"])
def test_load_invalid_safety_exempt_glob_raises(tmp_path: Path, bad_glob: str) -> None:
    # A safety_exempt entry that pathspec cannot parse as a gitignore glob
    # must surface as a config error naming the bad pattern, not crash deep
    # inside pathspec or be silently dropped.
    _write_settings(tmp_path, "settings.json", {"safety_exempt": [bad_glob]})
    with pytest.raises(AuraConfigError) as exc:
        load(tmp_path)
    assert "safety_exempt" in str(exc.value)
    assert "gitignore" in str(exc.value)


@pytest.mark.parametrize("greedy", ["**", "/etc/**", "**/.ssh/**"])
def test_load_safety_exempt_overlapping_builtin_protection_raises(
    tmp_path: Path, greedy: str,
) -> None:
    # The whole point of safety_exempt is to carve narrow holes; a pattern
    # that would re-expose a built-in protected path (ssh keys, /etc, .git)
    # must be refused so a config typo can't disarm a default safety entry.
    _write_settings(tmp_path, "settings.json", {"safety_exempt": [greedy]})
    with pytest.raises(AuraConfigError) as exc:
        load(tmp_path)
    msg = str(exc.value)
    assert "overlaps" in msg
    assert "refusing to" in msg


def test_load_narrow_safety_exempt_pattern_is_accepted(tmp_path: Path) -> None:
    # The negative control: a pattern that matches no protected sample must
    # pass cleanly, proving the overlap guard isn't a blanket rejection.
    _write_settings(tmp_path, "settings.json", {"safety_exempt": ["build/cache/"]})
    cfg = load(tmp_path)
    assert cfg.safety_exempt == ["build/cache/"]


def test_load_local_overlapping_safety_exempt_names_local_file(tmp_path: Path) -> None:
    # An overlapping pattern in the LOCAL file must blame the local file so
    # the user edits the right one.
    _write_settings(tmp_path, "settings.local.json", {"safety_exempt": ["**"]})
    with pytest.raises(AuraConfigError) as exc:
        load(tmp_path)
    assert "settings.local.json" in exc.value.source


# --- statusline + prompt_timeout merge precedence -------------------------


def test_load_statusline_from_project_is_merged(tmp_path: Path) -> None:
    # statusline lives in project settings; load must surface it on the
    # merged config (otherwise the configured status bar silently goes away).
    _write_settings(
        tmp_path, "settings.json",
        {"statusline": {"command": "echo P", "enabled": False, "timeout_ms": 250}},
    )
    cfg = load(tmp_path)
    assert cfg.statusline is not None
    assert cfg.statusline.command == "echo P"
    assert cfg.statusline.enabled is False
    assert cfg.statusline.timeout_ms == 250


def test_load_statusline_local_overrides_project(tmp_path: Path) -> None:
    # When both files define statusline, the machine-local one wins — a dev
    # can override the team's status bar without editing the shared file.
    _write_settings(
        tmp_path, "settings.json", {"statusline": {"command": "echo P"}},
    )
    _write_settings(
        tmp_path, "settings.local.json", {"statusline": {"command": "echo L"}},
    )
    cfg = load(tmp_path)
    assert cfg.statusline is not None
    assert cfg.statusline.command == "echo L"


def test_load_prompt_timeout_local_overrides_project(tmp_path: Path) -> None:
    # prompt_timeout_sec in the local file must take precedence over the
    # project value (line: local branch of the timeout merge).
    _write_settings(tmp_path, "settings.json", {"prompt_timeout_sec": 11.0})
    _write_settings(tmp_path, "settings.local.json", {"prompt_timeout_sec": 22.0})
    cfg = load(tmp_path)
    assert cfg.prompt_timeout_sec == 22.0


@pytest.mark.parametrize("zero_timeout", [0.0, -1.0])
def test_load_prompt_timeout_zero_and_negative_round_trip(
    tmp_path: Path, zero_timeout: float,
) -> None:
    # Numeric boundary: the loader passes the raw value through to the
    # schema; whatever the schema accepts for 0/-1 must survive the merge
    # unchanged rather than being coerced to the 300s default.
    _write_settings(
        tmp_path, "settings.local.json", {"prompt_timeout_sec": zero_timeout},
    )
    cfg = load(tmp_path)
    assert cfg.prompt_timeout_sec == zero_timeout


# --- deny / ask rulesets (malformed entries journal + skip, never raise) --


def test_load_deny_ruleset_parses_valid_entries(tmp_path: Path) -> None:
    # deny rules guard destructive ops; valid entries must parse into the
    # ruleset with the correct kind so the engine treats them as denials.
    _write_settings(
        tmp_path, "settings.json", {"deny": ["bash(rm -rf /)", "write_file(/etc)"]},
    )
    rs = load_deny_ruleset(tmp_path)
    assert isinstance(rs, RuleSet)
    assert [r.to_string() for r in rs.rules] == ["bash(rm -rf /)", "write_file(/etc)"]
    assert all(r.kind == "deny" for r in rs.rules)


def test_load_ask_ruleset_parses_valid_entries(tmp_path: Path) -> None:
    # ask rules force a confirmation prompt; the loader must tag them with
    # the ``ask`` kind, distinct from allow/deny.
    _write_settings(tmp_path, "settings.json", {"ask": ["bash(git push)"]})
    rs = load_ask_ruleset(tmp_path)
    assert [r.to_string() for r in rs.rules] == ["bash(git push)"]
    assert rs.rules[0].kind == "ask"


@pytest.mark.parametrize("field", ["deny", "ask"])
def test_load_kind_ruleset_skips_malformed_and_journals(
    tmp_path: Path, field: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A typo in ONE deny/ask string must not nuke the whole list (unlike
    # allow, which is strict): the bad entry is journalled for the user and
    # the valid ones still load. We capture the journal at the module seam.
    captured: list[dict[str, Any]] = []

    def _fake_write(event: str, **fields: Any) -> None:
        captured.append({"event": event, **fields})

    monkeypatch.setattr(
        "aura.infrastructure.permission_store.journal.write", _fake_write,
    )
    _write_settings(
        tmp_path, "settings.json", {field: ["bash(good)", "bash(unclosed"]},
    )
    loader = load_deny_ruleset if field == "deny" else load_ask_ruleset
    rs = loader(tmp_path)
    assert [r.to_string() for r in rs.rules] == ["bash(good)"]
    assert len(captured) == 1
    assert captured[0]["event"] == "permission_rule_parse_failed"
    assert captured[0]["kind"] == field
    assert captured[0]["rule"] == "bash(unclosed"


@pytest.mark.parametrize("field", ["deny", "ask"])
def test_load_kind_ruleset_swallows_journal_failure(
    tmp_path: Path, field: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even if the journal write itself blows up, a malformed rule must not
    # crash the load path — the rule is just dropped silently.
    def _boom(event: str, **fields: Any) -> None:
        raise RuntimeError("journal down")

    monkeypatch.setattr(
        "aura.infrastructure.permission_store.journal.write", _boom,
    )
    _write_settings(tmp_path, "settings.json", {field: ["bash(unclosed"]})
    loader = load_deny_ruleset if field == "deny" else load_ask_ruleset
    rs = loader(tmp_path)
    assert rs.rules == ()


@pytest.mark.parametrize("field", ["deny", "ask"])
def test_load_kind_ruleset_empty_when_absent(tmp_path: Path, field: str) -> None:
    # null/empty boundary: no settings file at all → an empty (not None)
    # ruleset, so callers can iterate without guarding.
    rs = (load_deny_ruleset if field == "deny" else load_ask_ruleset)(tmp_path)
    assert rs.rules == ()


# --- save_rule onto a file whose permissions value is non-dict ------------


def test_save_rule_recovers_when_permissions_is_non_object(tmp_path: Path) -> None:
    # A hand-edited settings.json with a junk ``permissions`` value (a list)
    # must not crash save_rule; it resets to a clean object and writes the
    # rule, keeping the tool usable after manual corruption.
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": ["junk"], "keep": 1}))
    save_rule(tmp_path, Rule(tool="bash", content="npm test"))
    reloaded = json.loads(settings.read_text())
    assert reloaded["permissions"]["allow"] == ["bash(npm test)"]
    assert reloaded["keep"] == 1


def test_save_rule_is_idempotent_after_corrupt_permissions(tmp_path: Path) -> None:
    # Idempotency boundary: calling save_rule twice on a recovered file must
    # not duplicate the rule.
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": 5}))
    rule = Rule(tool="bash", content="npm test")
    save_rule(tmp_path, rule)
    save_rule(tmp_path, rule)
    reloaded = json.loads(settings.read_text())
    assert reloaded["permissions"]["allow"] == [rule.to_string()]
