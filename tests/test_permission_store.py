"""Tests for aura.core.permissions.store — settings.json load/save/round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aura.config.schema import AuraConfigError
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet
from aura.core.permissions.store import (
    PermissionsConfig,
    PermissionStoreError,
    load,
    load_ruleset,
    save_rule,
)


def test_load_on_nonexistent_file_returns_defaults(tmp_path: Path) -> None:
    cfg = load(tmp_path)
    assert isinstance(cfg, PermissionsConfig)
    assert cfg.mode == "default"
    assert cfg.allow == []
    assert cfg.safety_exempt == []


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


def _write_settings(tmp_path: Path, name: str, perms: dict) -> Path:
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


def test_save_rule_writes_to_project_not_local_by_default(tmp_path: Path) -> None:
    # save_rule is the default "remember for this project" path — it must
    # never silently pollute settings.local.json.
    save_rule(tmp_path, Rule(tool="bash", content="npm test"))
    assert (tmp_path / ".aura" / "settings.json").exists()
    assert not (tmp_path / ".aura" / "settings.local.json").exists()


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
    assert not (tmp_path / ".aura" / "settings.local.json").exists()
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
