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

``save_rule`` defaults to writing ``settings.json`` (project-scoped) —
the common case is team-shared, so the default shoulders it. The
``scope="local"`` keyword opts into ``settings.local.json`` for
machine-local rules. Backend supports both scopes; the MVP CLI prompt
(§8.2) only exposes project scope — exposing a scope picker in the
prompt is Task 8+ work that no longer requires backend changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from aura.config.schema import AuraConfigError
from aura.core.permissions.rule import InvalidRuleError, Rule
from aura.core.permissions.session import RuleSet
from aura.errors import AuraError
from aura.schemas.permissions import PermissionsConfig


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
    #   statusline — local wins entirely when present (it's a single
    #          config object, not a list; local is the "per-machine
    #          override" surface, so it's the natural winner).
    # Route the merged dict through model_validate so pydantic narrows
    # ``mode`` from Any to the Literal it declares — mypy-clean without a cast.
    merged: dict[str, Any] = {
        "mode": local_raw.get("mode") or project_raw.get("mode") or "default",
        "allow": (
            list(project_raw.get("allow") or [])
            + list(local_raw.get("allow") or [])
        ),
        "safety_exempt": (
            list(project_raw.get("safety_exempt") or [])
            + list(local_raw.get("safety_exempt") or [])
        ),
    }
    statusline = local_raw.get("statusline") or project_raw.get("statusline")
    if statusline is not None:
        merged["statusline"] = statusline
    # disable_bypass: OR-together. Either layer setting it to ``true`` wins;
    # local CANNOT relax a project-level org kill switch. That asymmetry
    # is deliberate — ``settings.local.json`` is the per-machine override
    # surface, but for a compliance flag the project (committed) setting
    # must be the ceiling, not the floor.
    if "disable_bypass" in local_raw or "disable_bypass" in project_raw:
        merged["disable_bypass"] = bool(
            project_raw.get("disable_bypass") or local_raw.get("disable_bypass")
        )
    # prompt_timeout_sec: local wins when explicitly set (same "per-machine
    # override" semantics as ``mode``). An operator on a slow-to-respond
    # machine can bump the timeout up locally without editing the
    # committed project config.
    if "prompt_timeout_sec" in local_raw:
        merged["prompt_timeout_sec"] = local_raw["prompt_timeout_sec"]
    elif "prompt_timeout_sec" in project_raw:
        merged["prompt_timeout_sec"] = project_raw["prompt_timeout_sec"]
    return PermissionsConfig.model_validate(merged)


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


def _write_rule_to_file(settings: Path, rule: Rule) -> None:
    """Shared atomic-write body. ``settings`` is the resolved target file."""
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
    # Wrap BOTH the write and the replace. ``tmp.write_text`` can also fail
    # (disk full mid-write, permissions), and a raw OSError would propagate
    # up through the hook layer and crash the agent loop. Callers catch
    # ``PermissionStoreError`` specifically.
    try:
        tmp.write_text(json.dumps(top, indent=2))
        tmp.replace(settings)
    except OSError as exc:
        # Clean up the dangling tmp if the write half-succeeded but the
        # replace failed — stale .tmp files accumulate across retries
        # otherwise. ``missing_ok=True`` covers the case where the write
        # itself failed and no tmp exists.
        import contextlib
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise PermissionStoreError(source=str(settings), detail=str(exc)) from exc


def save_rule(
    project_root: Path,
    rule: Rule,
    *,
    scope: Literal["project", "local"] = "project",
) -> None:
    """Persist ``rule`` to the scope-selected settings file.

    - ``scope="project"`` (default) writes to ``./.aura/settings.json`` —
      shared / committed. This is the MVP CLI default (§8.2).
    - ``scope="local"`` writes to ``./.aura/settings.local.json`` —
      machine-local, gitignored by convention. Backend-only today; the
      CLI prompt doesn't expose the choice yet (§9 deferred item).

    Both scopes share the atomic-write pattern (tmp + ``Path.replace``),
    rule-string dedup, and preserve-unrelated-top-level-keys behavior.
    """
    if scope == "project":
        settings = _settings_path(project_root)
    elif scope == "local":
        settings = _settings_local_path(project_root)
    else:
        raise ValueError(
            f"scope must be 'project' or 'local', got {scope!r}",
        )
    _write_rule_to_file(settings, rule)
    # Side-effect: a project-scope save creates ``.aura/`` on first run,
    # which means ``ensure_local_settings`` (run once at CLI startup) has
    # already no-op'd. Drop the sibling local template in now so the user
    # discovers the machine-local file on THIS run, not the next one.
    # Idempotent — no-op if settings.local.json already exists.
    if scope == "project":
        ensure_local_settings(project_root)


def ensure_local_settings(project_root: Path) -> tuple[Path, bool]:
    """Create ``./.aura/settings.local.json`` with an empty template if absent.

    Returns ``(path, created)``: ``created=True`` iff we just wrote the file.

    Only fires when ``./.aura/`` already exists — i.e. the user has opted
    into aura in this directory (has a ``config.json`` or has been editing
    aura files here before). Running ``aura`` in a random dir where no
    ``.aura/`` exists is a no-op: we don't create clutter unprompted.

    Called at CLI startup so the machine-local override file is discoverable
    — the user can open it, see the empty allow list, edit manually, and
    expect it to be honoured on the next run.

    Content of a freshly-created file:

        {
          "permissions": {
            "allow": []
          }
        }

    Never overwrites — existing content (even ``{}``) short-circuits to
    ``created=False``. ``.gitignore`` already excludes the file.
    """
    settings = _settings_local_path(project_root)
    if settings.exists():
        return settings, False
    if not settings.parent.exists():
        # No .aura/ dir = user hasn't set up aura here. Don't pollute.
        return settings, False
    # The ``//`` key is a conventional JSON "comment" — harmless because our
    # top-level reader only consults ``permissions``, and ``save_rule``
    # preserves unrelated top-level keys on writes, so this hint survives
    # round-trips. Gives the user something to read when they open the file
    # for the first time.
    template: dict[str, Any] = {
        "//": (
            "Machine-local permission overrides. Add rule strings to "
            "permissions.allow to auto-approve tool calls without a "
            "prompt. Examples: \"bash(npm test)\", \"read_file(/tmp)\", "
            "\"grep\". Full spec: docs/specs/2026-04-19-aura-permission.md."
        ),
        "permissions": {
            "allow": [],
        },
    }
    settings.write_text(json.dumps(template, indent=2) + "\n")
    return settings, True
