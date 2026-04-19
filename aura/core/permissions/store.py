"""Permission persistence — ``./.aura/settings.json`` load/save.

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §7.

On-disk schema (validated by ``PermissionsConfig``)::

    {
      "permissions": {
        "mode": "default" | "bypass",
        "allow": ["bash", "bash(npm test)", ...],
        "safety_exempt": [".git/**", ...]
      }
    }

Unknown keys under ``permissions`` raise ``AuraConfigError`` on load — silent
typos in a security-relevant config are not acceptable. Top-level unrelated
sections (e.g. future ``ui``) are passed through on write untouched; the
permissions store owns only its own key.
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


def _settings_path(project_root: Path) -> Path:
    return project_root / ".aura" / "settings.json"


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


def load(project_root: Path) -> PermissionsConfig:
    settings = _settings_path(project_root)
    top = _read_top_level(settings)
    perms_raw = top.get("permissions")
    if perms_raw is None:
        return PermissionsConfig()
    try:
        return PermissionsConfig.model_validate(perms_raw)
    except ValidationError as exc:
        raise AuraConfigError(source=str(settings), detail=str(exc)) from exc


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
