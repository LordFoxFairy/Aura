"""Permission persistence — ``./.aura/settings{,.local}.json`` load/save.

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §7.

Two-file layout (both optional, either may be absent):

- ``.aura/settings.json``      — shared / committed-to-VCS project rules.
  The team's canonical "this project allows bash(npm test)" decisions.
- ``.aura/settings.local.json`` — machine-local overrides; should be
  gitignored. Your personal "always allow ssh prod" decisions that
  shouldn't leak to teammates.

Merge semantics (``load``):

- ``allow`` / ``safety_exempt`` — concatenated, project first, local
  appended. Both contribute rules (order is informational; first-match
  semantics in RuleSet fire on either).
- ``mode`` — local overrides project when local sets it explicitly.

On-disk schema (per file, validated by ``PermissionsConfig``)::

    {
      "permissions": {
        "mode": "default" | "bypass",
        "allow": ["bash", "bash(npm test)", ...],
        "safety_exempt": [".git/**", ...]
      }
    }

Unknown keys under ``permissions`` raise ``AuraConfigError`` naming the
offending file — silent typos in a security-relevant config are not
acceptable. Top-level unrelated sections (e.g. future ``ui``) are passed
through on write untouched; the permissions store owns only its own key.

``save_rule`` always writes to ``settings.json`` (project-scoped persist).
Machine-local rules are set by manually editing ``settings.local.json`` —
there is no CLI path that writes to it, by design (the common case is
team-shared, so the default should shoulder the common case).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aura.config.schema import AuraConfigError
from aura.core.permissions.rule import InvalidRuleError, Rule
from aura.core.permissions.session import RuleSet
from aura.errors import AuraError


class PermissionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["default", "bypass"] = "default"
    allow: list[str] = Field(default_factory=list)
    safety_exempt: list[str] = Field(default_factory=list)


class PermissionStoreError(AuraError):
    """Raised when writing settings.json fails (permissions, disk, etc)."""

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
    """Return the parsed settings.json top-level dict, or {} if absent.

    Raises ``AuraConfigError`` if the file is present but not valid JSON.
    """
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
    """Return the raw ``permissions`` dict from one settings file (or {} if
    file absent, no "permissions" key, or the key's value is not a dict —
    malformed non-dict values here raise ``AuraConfigError``).

    Running validation later per-file (in ``load``) gives clearer error
    messages that name the file that actually contains the typo.
    """
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


def load(project_root: Path) -> PermissionsConfig:
    """Load merged project + local permissions. See module docstring."""
    project_path = _settings_path(project_root)
    local_path = _settings_local_path(project_root)

    project_raw = _load_permissions_raw(project_path)
    local_raw = _load_permissions_raw(local_path)

    # Validate each file independently so error messages point at the
    # file that actually has the typo.
    _validate(project_raw, project_path)
    _validate(local_raw, local_path)

    # Merge:
    #   mode — local wins if it sets it; explicit "default" in local is
    #          still a wins-value (the user can downgrade project bypass
    #          on their own machine).
    #   allow / safety_exempt — concat, project first.
    mode = local_raw.get("mode") or project_raw.get("mode") or "default"
    allow = list(project_raw.get("allow") or []) + list(local_raw.get("allow") or [])
    safety_exempt = (
        list(project_raw.get("safety_exempt") or [])
        + list(local_raw.get("safety_exempt") or [])
    )

    return PermissionsConfig(mode=mode, allow=allow, safety_exempt=safety_exempt)


def load_ruleset(project_root: Path) -> RuleSet:
    cfg = load(project_root)
    parsed: list[Rule] = []
    for raw in cfg.allow:
        try:
            parsed.append(Rule.parse(raw))
        except InvalidRuleError as exc:
            raise AuraConfigError(
                source=str(_settings_path(project_root)),
                detail=f"invalid rule string {raw!r}: {exc}",
            ) from exc
    return RuleSet(rules=tuple(parsed))


def save_rule(project_root: Path, rule: Rule) -> None:
    """Persist ``rule`` to ``settings.json`` (project-scoped).

    Never writes to ``settings.local.json`` — the default CLI flow
    assumes shared persistence. Users who want machine-local rules edit
    that file manually.
    """
    settings = _settings_path(project_root)
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
    tmp.write_text(json.dumps(top, indent=2))
    try:
        tmp.replace(settings)
    except OSError as exc:
        # Leave the .tmp for the caller/user to inspect; atomic rename
        # either fully succeeded or did not happen at all.
        raise PermissionStoreError(source=str(settings), detail=str(exc)) from exc
