"""Permission persistence — ``./.aura/settings{,.local}.json`` load/save.

Merge invariants (load):

- scalars (``mode`` / ``prompt_timeout_sec`` / ``statusline``) — local wins when set
- lists (``allow`` / ``deny`` / ``ask`` / ``safety_exempt``) — concatenated, project first
- ``disable_bypass`` — OR (local cannot relax a project kill switch)
- unknown keys under ``permissions`` raise ``AuraConfigError`` naming the file
- non-``permissions`` top-level sections are preserved on write
"""

from __future__ import annotations

import contextlib
import difflib
import json
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

import pathspec
from pydantic import ValidationError

from aura.config.schema import AuraConfigError, PermissionsConfig
from aura.domain.errors import AuraError
from aura.domain.permission.rule import InvalidRuleError, Rule
from aura.domain.permission.safety import (
    DEFAULT_PROTECTED_READS,
    DEFAULT_PROTECTED_WRITES,
)
from aura.domain.permission.session import RuleSet


class PermissionStoreError(AuraError):
    def __init__(self, *, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.source = source
        self.detail = detail


_SETTINGS = "settings.json"
_SETTINGS_LOCAL = "settings.local.json"


def _settings_path(project_root: Path) -> Path:
    return project_root / ".aura" / _SETTINGS


def _settings_local_path(project_root: Path) -> Path:
    return project_root / ".aura" / _SETTINGS_LOCAL


def _read_top_level(settings: Path) -> dict[str, Any]:
    if not settings.exists():
        return {}
    try:
        raw = json.loads(settings.read_text())
    except json.JSONDecodeError as exc:
        raise AuraConfigError(
            source=str(settings), detail=f"invalid JSON: {exc}",
        ) from exc
    if not isinstance(raw, dict):
        raise AuraConfigError(
            source=str(settings), detail="top-level JSON must be an object",
        )
    return raw


def _load_permissions_raw(settings: Path) -> dict[str, Any]:
    top = _read_top_level(settings)
    if "permissions" not in top:
        return {}
    perms = top["permissions"]
    if not isinstance(perms, dict):
        raise AuraConfigError(
            source=str(settings),
            detail="'permissions' must be an object",
        )
    return perms


def _validate(raw: dict[str, Any], source: Path) -> PermissionsConfig:
    try:
        return PermissionsConfig.model_validate(raw)
    except ValidationError as exc:
        raise AuraConfigError(source=str(source), detail=str(exc)) from exc


# Reject safety_exempt patterns that match any of these — disarming a built-in.
_PROTECTED_OVERLAP_SAMPLES: tuple[str, ...] = (
    "~/.ssh/id_rsa",
    "~/.ssh/config",
    "~/project/.git/HEAD",
    "~/project/.git/config",
    "~/project/.aura/settings.json",
    "~/.bashrc",
    "~/.zshrc",
    "~/.profile",
    "~/.bash_profile",
    "~/.zprofile",
    "/etc/passwd",
    "/etc/hosts",
)


def _validate_safety_exempt(cfg: PermissionsConfig, source: Path) -> None:
    if not cfg.safety_exempt:
        return

    home = str(Path.home())
    samples = [
        s.replace("~", home, 1) if s.startswith("~") else s
        for s in _PROTECTED_OVERLAP_SAMPLES
    ]
    protected_names: list[tuple[str, pathspec.PathSpec]] = []
    for protected in (*DEFAULT_PROTECTED_WRITES, *DEFAULT_PROTECTED_READS):
        expanded = (
            protected.replace("~", home, 1)
            if protected.startswith("~")
            else protected
        )
        protected_names.append((
            protected,
            pathspec.PathSpec.from_lines("gitignore", [expanded]),
        ))

    for pattern in cfg.safety_exempt:
        expanded_pat = (
            pattern.replace("~", home, 1) if pattern.startswith("~") else pattern
        )
        try:
            spec = pathspec.PathSpec.from_lines("gitignore", [expanded_pat])
        except Exception as exc:  # noqa: BLE001  # any pathspec parser failure surfaces as config error
            raise AuraConfigError(
                source=str(source),
                detail=(
                    f"safety_exempt pattern {pattern!r} is not a valid "
                    f"gitignore-style glob: {exc}"
                ),
            ) from exc
        for sample in samples:
            if not spec.match_file(sample):
                continue
            for protected_pattern, protected_spec in protected_names:
                if protected_spec.match_file(sample):
                    raise AuraConfigError(
                        source=str(source),
                        detail=(
                            f"safety_exempt pattern {pattern!r} overlaps "
                            f"built-in protected pattern "
                            f"{protected_pattern!r} "
                            f"(sample path {sample!r}); refusing to "
                            "disarm a default safety entry"
                        ),
                    )


def load(project_root: Path) -> PermissionsConfig:
    project_path = _settings_path(project_root)
    local_path = _settings_local_path(project_root)

    project_raw = _load_permissions_raw(project_path)
    local_raw = _load_permissions_raw(local_path)

    # Validate each file standalone so errors point at the file with the typo.
    project_cfg = _validate(project_raw, project_path)
    local_cfg = _validate(local_raw, local_path)

    _validate_safety_exempt(project_cfg, project_path)
    _validate_safety_exempt(local_cfg, local_path)

    merged: dict[str, Any] = {
        "mode": local_raw.get("mode") or project_raw.get("mode") or "default",
        "allow": (
            list(project_raw.get("allow") or [])
            + list(local_raw.get("allow") or [])
        ),
        "deny": (
            list(project_raw.get("deny") or [])
            + list(local_raw.get("deny") or [])
        ),
        "ask": (
            list(project_raw.get("ask") or [])
            + list(local_raw.get("ask") or [])
        ),
        "safety_exempt": (
            list(project_raw.get("safety_exempt") or [])
            + list(local_raw.get("safety_exempt") or [])
        ),
    }
    statusline = local_raw.get("statusline") or project_raw.get("statusline")
    if statusline is not None:
        merged["statusline"] = statusline
    if "disable_bypass" in local_raw or "disable_bypass" in project_raw:
        merged["disable_bypass"] = bool(
            project_raw.get("disable_bypass") or local_raw.get("disable_bypass")
        )
    if "prompt_timeout_sec" in local_raw:
        merged["prompt_timeout_sec"] = local_raw["prompt_timeout_sec"]
    elif "prompt_timeout_sec" in project_raw:
        merged["prompt_timeout_sec"] = project_raw["prompt_timeout_sec"]
    return PermissionsConfig.model_validate(merged)


def _validate_known_tools(
    rules: Iterable[Rule],
    known_tool_names: Iterable[str],
    *,
    source: str,
) -> None:
    """Reject unknown tool names; wildcards pass (MCP server may register them later)."""
    known_set = set(known_tool_names)
    for rule in rules:
        if "*" in rule.tool:
            continue
        if rule.tool in known_set:
            continue
        if any("*" in known and fnmatchcase(rule.tool, known) for known in known_set):
            continue
        suggestions = difflib.get_close_matches(rule.tool, known_set, n=1)
        hint = f"; did you mean {suggestions[0]!r}?" if suggestions else ""
        raise AuraConfigError(
            source=source,
            detail=(
                f"unknown tool name in rule {rule.to_string()!r}: "
                f"{rule.tool!r}{hint}"
            ),
        )


def load_ruleset(
    project_root: Path,
    *,
    known_tool_names: Iterable[str] | None = None,
) -> RuleSet:
    cfg = load(project_root)
    parsed: list[Rule] = []
    for raw in cfg.allow:
        try:
            parsed.append(Rule.parse(raw, kind="allow"))
        except InvalidRuleError as exc:
            raise AuraConfigError(
                source=str(_settings_path(project_root)),
                detail=f"invalid rule string {raw!r}: {exc}",
            ) from exc
    if known_tool_names is not None:
        _validate_known_tools(
            parsed,
            known_tool_names,
            source=str(_settings_path(project_root)),
        )
    return RuleSet(rules=tuple(parsed))


def _load_kind_ruleset(
    project_root: Path,
    *,
    field: Literal["deny", "ask"],
) -> RuleSet:
    """Deny/ask loader; malformed entries journal + skip rather than raise."""
    from aura.core import journal as _j  # noqa: PLC0415  # defer to dodge import cycle

    cfg = load(project_root)
    raw_list = cfg.deny if field == "deny" else cfg.ask
    parsed: list[Rule] = []
    for raw in raw_list:
        try:
            parsed.append(Rule.parse(raw, kind=field))
        except InvalidRuleError as exc:
            with contextlib.suppress(Exception):
                _j.write(
                    "permission_rule_parse_failed",
                    kind=field,
                    rule=raw,
                    detail=str(exc),
                )
    return RuleSet(rules=tuple(parsed))


def load_deny_ruleset(project_root: Path) -> RuleSet:
    return _load_kind_ruleset(project_root, field="deny")


def load_ask_ruleset(project_root: Path) -> RuleSet:
    return _load_kind_ruleset(project_root, field="ask")


def _write_rule_to_file(settings: Path, rule: Rule) -> None:
    settings.parent.mkdir(parents=True, exist_ok=True)

    top = _read_top_level(settings)
    perms = top.get("permissions") or {}
    if not isinstance(perms, dict):
        perms = {}
    allow = list(perms.get("allow") or [])
    rule_str = rule.to_string()
    if rule_str not in allow:
        allow.append(rule_str)
    perms["allow"] = allow
    top["permissions"] = perms

    tmp = settings.with_suffix(settings.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(top, indent=2))
        tmp.replace(settings)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise PermissionStoreError(source=str(settings), detail=str(exc)) from exc


def save_rule(
    project_root: Path,
    rule: Rule,
    *,
    scope: Literal["project", "local"] = "project",
) -> None:
    """Atomic-write ``rule`` into the chosen scope; preserves unrelated top-level keys."""
    if scope == "project":
        settings = _settings_path(project_root)
    elif scope == "local":
        settings = _settings_local_path(project_root)
    else:
        raise ValueError(
            f"scope must be 'project' or 'local', got {scope!r}",
        )
    _write_rule_to_file(settings, rule)
    # First project save creates .aura/ — drop the local template now so it's discoverable.
    if scope == "project":
        ensure_local_settings(project_root)


def ensure_local_settings(project_root: Path) -> tuple[Path, bool]:
    """Drop an empty local-settings template when ``.aura/`` exists and the file does not."""
    settings = _settings_local_path(project_root)
    if settings.exists():
        return settings, False
    if not settings.parent.exists():
        return settings, False
    template: dict[str, Any] = {
        "//": (
            "Machine-local permission overrides. Add rule strings to "
            "permissions.allow to auto-approve tool calls without a "
            "prompt. Glob metachars (* and ?) are supported in rule "
            "content, so one rule covers a whole family. Examples: "
            "\"bash(npm test)\" (exact), \"bash(npm install *)\" (glob — "
            "covers every npm install variant), \"bash(ls *)\" (any ls), "
            "\"read_file(/tmp)\" (path prefix), \"grep\" (tool-wide)."
        ),
        "permissions": {
            "allow": [],
        },
    }
    settings.write_text(json.dumps(template, indent=2) + "\n")
    return settings, True
