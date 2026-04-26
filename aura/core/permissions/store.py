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

import difflib
import json
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from aura.config.schema import AuraConfigError
from aura.core.permissions.rule import InvalidRuleError, Rule
from aura.core.permissions.safety import (
    DEFAULT_PROTECTED_READS,
    DEFAULT_PROTECTED_WRITES,
)
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


# Concrete representative paths each protected pattern is *intended* to
# block. ``safety_exempt`` overlap with any of these is a configuration
# error: the user has weakened a built-in protection rather than added
# a project-specific carve-out. Expanded against ``Path.home()`` /
# absolutized at validation time so a user pattern like ``~/.ssh/**``
# is recognised as overlapping ``**/.ssh/**``.
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


def _validate_safety_exempt(
    cfg: PermissionsConfig, source: Path,
) -> None:
    """Reject ``safety_exempt`` patterns that overlap any built-in
    protected entry. F-04-020 — a user pattern such as ``~/.ssh/**`` or
    ``**/.git/**`` would otherwise silently disarm the safety floor.
    """
    if not cfg.safety_exempt:
        return
    import pathspec

    home = str(Path.home())
    samples = [
        s.replace("~", home, 1) if s.startswith("~") else s
        for s in _PROTECTED_OVERLAP_SAMPLES
    ]
    # Pre-compile both protected lists once so we can identify which
    # built-in entry the offending pattern overlapped (clear error).
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
        except Exception as exc:  # noqa: BLE001 — surface as config error
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
            # The exempt pattern matches a sample that the protected list
            # is meant to block. Find which built-in entry to name in
            # the error so the user knows what they tried to disarm.
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
    """Load merged project + local permissions. See module docstring."""
    project_path = _settings_path(project_root)
    local_path = _settings_local_path(project_root)

    project_raw = _load_permissions_raw(project_path)
    local_raw = _load_permissions_raw(local_path)

    # Validate each file independently so error messages point at the
    # file that actually has the typo.
    project_cfg = _validate(project_raw, project_path)
    local_cfg = _validate(local_raw, local_path)

    # F-04-020: per-file safety_exempt overlap checks BEFORE merge so
    # the error names the file that actually contains the offending
    # pattern. A merged-list check would lose that attribution.
    _validate_safety_exempt(project_cfg, project_path)
    _validate_safety_exempt(local_cfg, local_path)

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


def _validate_known_tools(
    rules: Iterable[Rule],
    known_tool_names: Iterable[str],
    *,
    source: str,
) -> None:
    """Reject rules whose ``tool`` is not in ``known_tool_names``.

    Wildcard tool patterns (``*`` anywhere in the tool field, e.g.
    ``mcp__github__*``) are treated as known by default — they're scoped
    to a server's surface that may not yet be discovered at load time, and
    a wildcard that matches NOTHING is harmless (the rule simply never
    fires). Exact-name rules with no match raise :class:`AuraConfigError`,
    naming the offending tool plus the closest known name (via
    :func:`difflib.get_close_matches`) so the operator can fix the typo.

    Failure modes are surfaced one-at-a-time; we raise on the first miss
    rather than aggregating, because subsequent errors are usually
    correlated typos and aggregating noise hides the actionable signal.
    """
    known_set = set(known_tool_names)
    for rule in rules:
        if "*" in rule.tool:
            continue
        if rule.tool in known_set:
            continue
        # Wildcards in known_tool_names (e.g. ``mcp__github__*`` registered
        # by an MCP server) should still cover non-wildcard rules that
        # match the wildcard's surface.
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
    """Load merged settings into a :class:`RuleSet`.

    When ``known_tool_names`` is supplied, every parsed rule's ``tool``
    field is validated against it via :func:`_validate_known_tools` —
    typos surface as :class:`AuraConfigError` at startup instead of as
    silent never-fires at runtime. Pass the union of registry tool names
    plus any internal names (``bash``, ``read_file``, ...). When
    ``known_tool_names`` is ``None`` the validation step is skipped to
    preserve backward compatibility for callers that haven't built a
    tool pool yet.
    """
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
    """Shared body for ``load_deny_ruleset`` / ``load_ask_ruleset``.

    Malformed entries journal a ``permission_rule_parse_failed`` event
    and are skipped — a typo in deny / ask must not crash startup.
    """
    import contextlib

    cfg = load(project_root)
    raw_list = cfg.deny if field == "deny" else cfg.ask
    parsed: list[Rule] = []
    from aura.core import journal as _j  # noqa: PLC0415
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
    """Load deny rules. Malformed entries journal + skip (no raise)."""
    return _load_kind_ruleset(project_root, field="deny")


def load_ask_ruleset(project_root: Path) -> RuleSet:
    """Load ask rules. Malformed entries journal + skip (no raise)."""
    return _load_kind_ruleset(project_root, field="ask")


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
