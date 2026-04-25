"""Discover + parse skill directories from user + project layers.

Claude-code v2.1.88 parity (see ``loadSkillsDir.ts``):

- **Layout**: one directory per skill, ``<name>/SKILL.md`` inside it. Plain
  ``.md`` files at the top of a ``.aura/skills/`` dir are no longer loaded —
  they emit a ``skill_legacy_format_detected`` journal event so the user can
  migrate explicitly. Mirrors ``loadSkillsFromSkillsDir`` lines 421-480.
- **Layers**:

  1. User   → ``~/.aura/skills/<name>/SKILL.md``
  2. Project → walk up from ``cwd`` to ``Path.home()`` (exclusive),
     scanning each ``<dir>/.aura/skills/<name>/SKILL.md``. Mirrors
     claude-code's ``getProjectDirsUpToHome('skills', cwd)``. Deeper
     (closer-to-cwd) dirs are visited last so they LOSE on name collisions —
     matches claude-code's "first-seen wins" dedup order in
     ``getSkillDirCommands`` where user layer is pushed before project.

- **Realpath dedup**: two discovered dirs that resolve to the same file
  (symlinks, overlapping parents) are collapsed to one. First-seen wins —
  same as the TS ``seenFileIds`` loop. Uses ``Path.resolve()`` which is the
  Python equivalent of ``realpath`` (the primitive claude-code uses).
- **Conditional skills**: ``paths:`` frontmatter → skill is STORED but not
  active at startup. ``activate_conditional_skills_for_paths`` activates
  matching skills lazily. Activation state is module-global so it survives
  across ``load_skills`` calls within one session (matches
  ``activatedConditionalSkillNames``).

Required frontmatter: ``description`` (str, non-empty). ``name`` is an
optional override — falls back to the directory name (dir name IS the
skill id in claude-code). Missing description silently skips the skill +
emits ``skill_parse_failed`` (same "silent-skip-with-audit" pattern as the
rest of Aura's loader code — one bad skill must not break the catalogue).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pathspec
import yaml

from aura.core.skills.registry import SkillRegistry
from aura.core.skills.types import Skill, SkillLayer

_AURA_DIR = ".aura"
_CLAUDE_DIR = ".claude"
_SKILLS_DIR = "skills"
_SKILL_FILE = "SKILL.md"

# Claude-code-compat namespace: skills written for claude-code reference
# ``${CLAUDE_SKILL_DIR}`` / ``${CLAUDE_SESSION_ID}`` in their body. We accept
# both namespaces at render time so a user can drop a claude-code skill into
# ``~/.aura/skills/`` or ``~/.claude/skills/`` and it renders correctly
# without edits. New Aura-native skills should still prefer ``${AURA_*}``.
_PLACEHOLDER_PAIRS: tuple[tuple[str, str], ...] = (
    ("${AURA_SKILL_DIR}", "base_dir"),
    ("${CLAUDE_SKILL_DIR}", "base_dir"),
    ("${AURA_SESSION_ID}", "session_id"),
    ("${CLAUDE_SESSION_ID}", "session_id"),
)

# Inline shell-execution syntax used by claude-code (``!`cmd` `` inside the
# skill body). Deliberately NOT supported in Aura — security surface we don't
# need in v1. Detected at load time so an imported skill that relies on it
# surfaces a journal warning instead of silently mis-rendering to literal
# text that confuses the model.
_INLINE_CMD_PATTERN = "!`"

# Module-global conditional-skill registry. Mirrors claude-code's
# ``conditionalSkills`` map + ``activatedConditionalSkillNames`` set. Lives
# at module scope so activation sticks across ``load_skills`` calls within
# a single process — matches the TS semantics where memoized discovery +
# activation state are both process-lifetime.
_conditional_skills: dict[str, Skill] = {}
_activated_conditional_names: set[str] = set()


def load_skills(cwd: Path, *, home: Path | None = None) -> SkillRegistry:
    """Scan user + project layers and return a populated SkillRegistry.

    Conditional skills (``paths:`` frontmatter) are stored in the module's
    conditional map instead of the returned registry — they only appear in
    the catalogue after ``activate_conditional_skills_for_paths`` moves
    them over.

    ``home`` defaults to ``Path.home()``; exposed as kwarg for tests.
    """
    home_dir = (home if home is not None else Path.home()).resolve()
    cwd_resolved = cwd.resolve()

    registry = SkillRegistry()
    seen_source_paths: set[Path] = set()

    # --- Layer 1a: user (Aura-native) ---
    # User layer goes first so it wins on name collisions — matches TS
    # ``allSkillsWithPaths = [...userSkills, ...projectSkills]`` ordering.
    user_skills_root = home_dir / _AURA_DIR / _SKILLS_DIR
    for skill in _load_layer(user_skills_root, layer="user"):
        _install_or_drop(skill, registry, seen_source_paths)

    # --- Layer 1b: user (claude-code-compat) ---
    # ``~/.claude/skills/`` holds skills a user already authored for
    # claude-code. Aura loads them verbatim — dir-per-skill, same frontmatter
    # schema, dual-namespace placeholders handled at render time. Realpath
    # dedup against the Aura-native layer (first-seen wins), so symlinking
    # ``~/.claude/skills/foo -> ~/.aura/skills/foo`` won't double-register.
    # ``user`` layer tag kept so collision semantics are uniform — both
    # locations are user-owned.
    claude_user_skills_root = home_dir / _CLAUDE_DIR / _SKILLS_DIR
    if claude_user_skills_root.resolve() != user_skills_root.resolve():
        for skill in _load_layer(claude_user_skills_root, layer="user"):
            _install_or_drop(skill, registry, seen_source_paths)

    # --- Layer 2: project (walk up from cwd to home exclusive) ---
    # Closer-to-cwd dirs are visited LAST so they lose on collisions against
    # already-registered skills (user + outer project). This is the
    # "outer-wins" convention: an inner child project inherits its parent's
    # skills but cannot silently shadow them with a same-named skill. The
    # user can still override by putting the skill in ~/.aura/skills/, which
    # is the one layer with real ownership semantics.
    for project_dir in _project_dirs_up_to_home(cwd_resolved, home_dir):
        project_skills_root = project_dir / _AURA_DIR / _SKILLS_DIR
        if project_skills_root.resolve() == user_skills_root.resolve():
            # If cwd is itself inside $HOME, the walk-up hits $HOME's own
            # .aura/skills/ which we already scanned as the user layer.
            continue
        for skill in _load_layer(project_skills_root, layer="project"):
            _install_or_drop(skill, registry, seen_source_paths)

    return registry


def activate_conditional_skills_for_paths(
    paths: list[str], cwd: Path,
) -> list[str]:
    """Activate stored conditional skills whose ``paths:`` match ``paths``.

    Mirrors ``activateConditionalSkillsForPaths`` (TS). Uses ``pathspec``
    with gitignore semantics — same library claude-code's ``ignore`` pkg
    is modelled on, and what Aura already uses for rules matching.
    Activated skills are registered into a fresh SkillRegistry returned to
    the caller; tracked in ``_activated_conditional_names`` so subsequent
    ``load_skills`` calls don't re-stash them as conditional.

    Returns: names of newly-activated skills (possibly empty).

    Contract note: the loop's file-touching code paths don't wire this yet
    — exposing the API is enough to unblock that integration work.
    """
    if not _conditional_skills:
        return []

    cwd_resolved = cwd.resolve()
    activated: list[str] = []

    for name in list(_conditional_skills.keys()):
        skill = _conditional_skills[name]
        if not skill.paths:
            # Shouldn't happen (conditional → is_conditional() → len(paths)>0),
            # but belt-and-suspenders: promote orphaned entries unconditionally.
            _activated_conditional_names.add(name)
            del _conditional_skills[name]
            activated.append(name)
            continue
        try:
            spec = pathspec.PathSpec.from_lines("gitignore", skill.paths)
        except Exception:  # noqa: BLE001 — pathspec raises many exc types
            # Same tolerance as rules.py: a bad glob pattern can't take down
            # the activation pipeline. Journal is lossy here because the
            # skill was already loaded — the user sees the miss via "skill
            # did not activate" rather than a parse error.
            continue

        for raw_path in paths:
            rel = _relative_to_cwd(raw_path, cwd_resolved)
            if rel is None:
                continue
            if spec.match_file(rel):
                _activated_conditional_names.add(name)
                del _conditional_skills[name]
                activated.append(name)
                break

    return activated


def get_conditional_skills() -> list[Skill]:
    """Return all skills stored as conditional (for integration surfaces)."""
    return list(_conditional_skills.values())


def activated_conditional_names() -> frozenset[str]:
    """Return the set of skills that have been activated this session."""
    return frozenset(_activated_conditional_names)


def clear_conditional_state() -> None:
    """Test hook — reset the module-global conditional registry."""
    _conditional_skills.clear()
    _activated_conditional_names.clear()


def render_skill_body(
    skill: Skill,
    session_id: str,
    argument_values: list[str] | None = None,
) -> str:
    """Substitute skill-dir / session-id / per-arg placeholders in the body.

    Dual-namespace: ``${AURA_SKILL_DIR}`` and ``${CLAUDE_SKILL_DIR}`` both
    substitute to the skill's base_dir; ``${AURA_SESSION_ID}`` and
    ``${CLAUDE_SESSION_ID}`` both substitute to the session id. This lets a
    claude-code skill (which references the ``CLAUDE_*`` names) drop into
    ``~/.aura/skills/`` or be picked up from ``~/.claude/skills/`` and
    render without edits. New Aura-native skills should still prefer the
    ``AURA_*`` names — the claude-code names stay as a compat escape hatch.

    Claude-code performs this substitution inside ``getPromptForCommand``
    (loadSkillsDir.ts:344-369). We do it at invocation time (tool / slash
    command) so the Context's ``<skill-invoked>`` render block can stay a
    dumb string-copier — no session-id leak into the memory module.
    """
    body = skill.body
    base_dir = skill.base_dir if skill.base_dir is not None else skill.source_path.parent
    replacements = {"base_dir": str(base_dir), "session_id": session_id}
    for placeholder, key in _PLACEHOLDER_PAIRS:
        body = body.replace(placeholder, replacements[key])
    values = argument_values or []
    for i, arg_name in enumerate(skill.arguments):
        placeholder = "${" + arg_name + "}"
        value = values[i] if i < len(values) else ""
        body = body.replace(placeholder, value)
    return body


# ---------------------------------------------------------------------------
# Internals — per-layer scan
# ---------------------------------------------------------------------------


def _install_or_drop(
    skill: Skill,
    registry: SkillRegistry,
    seen_source_paths: set[Path],
) -> None:
    """Add ``skill`` to the registry, stash under conditional, or drop on dup."""
    # Realpath dedup: a skill dir reached via two paths (symlink, overlapping
    # parent) is already-loaded. Silent skip + journal — matches TS
    # ``Skipping duplicate skill`` debug log.
    if skill.source_path in seen_source_paths:
        from aura.core.persistence import journal

        journal.write(
            "skill_duplicate_skipped",
            name=skill.name,
            source_path=str(skill.source_path),
        )
        return
    seen_source_paths.add(skill.source_path)

    # Conditional skills go into the lazy bucket unless already activated
    # this session. Activated conditionals land in the registry, but we
    # have to flip the ``activated`` flag on the Skill dataclass so
    # render-time filters (Context.build) know the conditional trigger
    # has already fired — otherwise ``is_conditional()`` stays True and
    # the skill is hidden from <skills-available> forever.
    if skill.is_conditional():
        if skill.name in _activated_conditional_names:
            from dataclasses import replace
            skill = replace(skill, activated=True)
        else:
            _conditional_skills[skill.name] = skill
            return

    # Name collision → first writer wins (user layer came first).
    if registry.get(skill.name) is not None:
        from aura.core.persistence import journal

        existing = registry.get(skill.name)
        journal.write(
            "skill_name_collision",
            name=skill.name,
            kept_path=str(existing.source_path) if existing else "",
            dropped_path=str(skill.source_path),
        )
        return
    registry.register(skill)


def _project_dirs_up_to_home(cwd: Path, home: Path) -> list[Path]:
    """Return cwd, cwd.parent, ..., up to (but NOT including) ``home``.

    If cwd is not under home (pathological test setups), return ``[cwd]`` —
    we still scan the project layer at cwd. Claude-code's
    ``getProjectDirsUpToHome`` uses the same semantics: the walk stops
    *before* entering home so a user's ``~/.aura/skills/`` isn't double-
    counted as both layers.
    """
    dirs: list[Path] = []
    current = cwd
    while True:
        dirs.append(current)
        parent = current.parent
        if parent == current:
            break  # Filesystem root — stop.
        if current == home:
            # We've reached home; the next iteration would be outside home.
            break
        if parent == home:
            break  # Don't include home itself in the project layer.
        current = parent
    # Visit outer dirs first so closer-to-cwd skills LOSE on collisions
    # (outer wins — matches "parent project can't be shadowed by child").
    # Reverse from the cwd-first order we built.
    dirs.reverse()
    return dirs


def _load_layer(skills_root: Path, *, layer: SkillLayer) -> list[Skill]:
    """Return parsed Skills from ``skills_root/<name>/SKILL.md`` entries."""
    if not skills_root.is_dir():
        return []

    out: list[Skill] = []
    try:
        entries = sorted(skills_root.iterdir())
    except OSError:
        return []

    # Legacy detection: plain .md at the top level → user still on the old
    # flat layout. Journal + skip so they migrate explicitly (wrap each
    # ``foo.md`` → ``foo/SKILL.md``).
    legacy_files = [e for e in entries if e.is_file() and e.suffix == ".md"]
    if legacy_files:
        from aura.core.persistence import journal

        journal.write(
            "skill_legacy_format_detected",
            layer=layer,
            root=str(skills_root),
            files=[str(f) for f in legacy_files],
        )

    for entry in entries:
        # Only dirs (and symlinks to dirs) are valid skill containers.
        if not entry.is_dir():
            continue
        skill_file = entry / _SKILL_FILE
        if not skill_file.is_file():
            # Empty skill dir / sidecar (e.g. ``examples/`` directory on
            # its own, misplaced) — skip silently.
            continue
        skill = _build_skill(skill_file, layer=layer)
        if skill is not None:
            out.append(skill)
    return out


def _build_skill(skill_file: Path, *, layer: SkillLayer) -> Skill | None:
    """Parse one ``SKILL.md`` file into a Skill, or silent-skip."""
    raw = _read_text(skill_file)
    if raw is None:
        _emit_parse_failed(skill_file, "unreadable")
        return None

    frontmatter_text, body = _split_frontmatter(raw)
    # Frontmatter is optional per claude-code (see the coerceDescriptionToString
    # fallback); but we require ``description`` so we need SOME frontmatter.
    if frontmatter_text is None:
        _emit_parse_failed(skill_file, "missing frontmatter")
        return None

    try:
        parsed = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        _emit_parse_failed(skill_file, f"yaml error: {exc}")
        return None

    if not isinstance(parsed, dict):
        _emit_parse_failed(skill_file, "frontmatter is not a mapping")
        return None

    # Directory name IS the skill id; ``name`` in frontmatter is a display
    # override. Match claude-code's ``displayName`` + ``userFacingName()``
    # split where ``name`` = dir name but ``userFacingName()`` uses the
    # override. Aura collapses: the override replaces ``name`` outright.
    dir_name = skill_file.parent.name
    name_override = parsed.get("name")
    if isinstance(name_override, str) and name_override.strip():
        resolved_name = name_override.strip()
    else:
        resolved_name = dir_name

    description = parsed.get("description")
    if not isinstance(description, str) or not description.strip():
        _emit_parse_failed(skill_file, "missing or non-string 'description'")
        return None

    try:
        source = skill_file.resolve()
    except OSError:
        _emit_parse_failed(skill_file, "resolve failed")
        return None

    try:
        base_dir = skill_file.parent.resolve()
    except OSError:
        base_dir = skill_file.parent

    when_to_use = _coerce_optional_str(parsed.get("when_to_use"))
    version = _coerce_optional_str(parsed.get("version"))
    argument_hint = _coerce_optional_str(parsed.get("argument-hint"))

    allowed_tools = _coerce_str_list_field(parsed.get("allowed-tools"))
    # V14: ``restrict-tools`` is a strict whitelist — separate field from
    # the permissive ``allowed-tools`` (v0.13). Same parsing shape (list-or-
    # whitespace-string). Two fields, two semantics: combining them in one
    # field would silently break v0.12's claude-code-imported skill compat.
    restrict_tools = _coerce_str_list_field(parsed.get("restrict-tools"))
    arguments = tuple(_coerce_str_list_field(parsed.get("arguments")))
    paths_raw = _coerce_str_list_field(parsed.get("paths"))
    # Normalize "/**" suffix — pathspec treats "src" as matching both the
    # file and the directory contents, so "src/**" is redundant. Mirrors TS
    # ``parseSkillPaths`` lines 165-168.
    normalized_paths: list[str] = []
    for pattern in paths_raw:
        if pattern.endswith("/**"):
            pattern = pattern[:-3]
        if pattern:
            normalized_paths.append(pattern)
    # "All match-all" patterns → treat as unconditional (match TS lines 172-174).
    if normalized_paths and all(p == "**" for p in normalized_paths):
        normalized_paths = []

    user_invocable = _coerce_bool(parsed.get("user-invocable"), default=True)
    disable_model_invocation = _coerce_bool(
        parsed.get("disable-model-invocation"), default=False,
    )

    # Compat check: claude-code allows inline ``!`cmd`` `` shell execution
    # inside skill bodies (``executeShellCommandsInPrompt``). Aura does NOT
    # execute these — the body is handed to the model as literal text. If a
    # user imports a claude-code skill that relies on it, the model will see
    # the unexecuted command syntax and likely do the wrong thing. Emit a
    # journal warning so the mismatch is auditable rather than silent.
    if _INLINE_CMD_PATTERN in body:
        from aura.core.persistence import journal

        journal.write(
            "skill_inline_cmd_unsupported",
            name=resolved_name,
            source_path=str(source),
            layer=layer,
        )

    return Skill(
        name=resolved_name,
        description=description.strip(),
        body=body,
        source_path=source,
        base_dir=base_dir,
        layer=layer,
        when_to_use=when_to_use,
        allowed_tools=frozenset(allowed_tools),
        restrict_tools=frozenset(restrict_tools),
        arguments=arguments,
        argument_hint=argument_hint,
        version=version,
        paths=frozenset(normalized_paths),
        user_invocable=user_invocable,
        disable_model_invocation=disable_model_invocation,
    )


def _coerce_optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


def _coerce_str_list_field(value: Any) -> list[str]:
    """Accept ``list[str]`` or whitespace-separated str. Drops empty items."""
    if value is None:
        return []
    if isinstance(value, str):
        # Claude-code's ``parseArgumentNames`` splits on whitespace when
        # a scalar is given — same behaviour here so both YAML forms work
        # (``arguments: foo bar`` and ``arguments: [foo, bar]``).
        return [p for p in value.split() if p]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Parse booleans loosely — matches claude-code's ``parseBooleanFrontmatter``."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
    return default


def _relative_to_cwd(raw_path: str, cwd: Path) -> str | None:
    """Return ``raw_path`` as a cwd-relative POSIX string, or None if outside.

    Matches TS ``activateConditionalSkillsForPaths`` lines 1014-1027: paths
    outside cwd can't match cwd-relative patterns, and ``ignore`` raises on
    ``..``/absolute — guard the same way.
    """
    p = Path(raw_path)
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(cwd)
            return rel.as_posix()
        except (ValueError, OSError):
            return None
    # Already relative — just ensure it stays inside cwd.
    joined = (cwd / p).resolve()
    try:
        rel = joined.relative_to(cwd)
        return rel.as_posix()
    except ValueError:
        return None


def _emit_parse_failed(skill_file: Path, error: str) -> None:
    from aura.core.persistence import journal

    try:
        path_str = str(skill_file.resolve())
    except OSError:
        path_str = str(skill_file)
    journal.write("skill_parse_failed", path=path_str, error=error)


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")


def _split_frontmatter(raw: str) -> tuple[str | None, str]:
    """Split out the leading ``---``-fenced YAML frontmatter."""
    lines = raw.splitlines(keepends=True)
    if not lines:
        return None, raw

    first = lines[0].rstrip("\r\n").rstrip()
    if first != "---":
        return None, raw

    for idx in range(1, len(lines)):
        stripped = lines[idx].rstrip("\r\n").rstrip()
        if stripped in {"---", "..."}:
            frontmatter = "".join(lines[1:idx])
            body = "".join(lines[idx + 1 :])
            return frontmatter, body

    return None, raw
