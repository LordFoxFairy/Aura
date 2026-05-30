"""Discover + parse skill directories from user + project layers.

Layout: one directory per skill, ``<name>/SKILL.md`` inside it. Layers in
load order (user wins on name collisions):

1. Managed (bundled) — opt-in via ``include_bundled=True`` (Agent.__init__).
2. User — ``~/.aura/skills/<name>/SKILL.md`` and ``~/.claude/skills/``.
3. Project — walk up from ``cwd`` to ``Path.home()`` exclusive.

Conditional skills (``paths:`` frontmatter) are stashed; activation moves
them into the live registry on a matching file touch.

Required frontmatter: ``description``. ``name`` overrides the dir name.
Missing description → silent skip + ``skill_parse_failed`` journal event.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pathspec
import yaml

from aura.domain.skill import Skill, SkillLayer
from aura.infrastructure.skills.registry import SkillRegistry

_AURA_DIR = ".aura"
_CLAUDE_DIR = ".claude"
_SKILLS_DIR = "skills"
_SKILL_FILE = "SKILL.md"

# Bundled skills materialize from code-defined content (below) into a
# hidden runtime root under ``~/.aura/plugins`` so the active catalogue is
# detached from the Python package layout.
_BUNDLED_SKILLS_EXTRACTED_ROOT_NAME = "skills"
_BUNDLED_CACHE_KEY = "aura-bundled-skills"
_bundled_skills_extraction: tuple[str, Path] | None = None
_BUNDLED_SKILL_FILES: dict[str, str] = {
    "verify": """---
description: Verify the most recent change works end-to-end before claiming done.
when_to_use: Before responding \"done\" / \"fixed\" / \"passing\" — run real checks.
---
# Verify

Before claiming a task is complete:

1. Run the project's tests (`make check`, `pytest`, `npm test`, etc.) and confirm
   they pass.
2. Re-run the specific failing case from the bug report — don't assume related
   tests cover it.
3. Read back the changed files to confirm the diff is what you intended.
4. If the change touches a CLI / API surface, exercise it end-to-end at least
   once instead of trusting unit tests alone.

Evidence before assertions: paste the actual command output that proves the
verification, not a paraphrase.
""",
    "simplify": """---
description: Review the diff for reuse, dead code, and over-engineering before commit.
when_to_use: After implementing a change, before committing — pause to simplify.
---
# Simplify

Pre-commit pass over the current diff:

1. Is there an existing helper / utility that already does this? Reuse it
   instead of duplicating.
2. Did you add a flag, knob, or abstraction that no caller currently exercises?
   Drop it — half-wired extensibility rots.
3. Are comments explaining \"what\" instead of \"why\"? Strip the \"what\"; the
   code shows what.
4. Is there dead code (unreachable branches, unused imports, stale docstrings
   referencing removed behavior)? Delete it.
5. Could the same outcome be expressed with fewer lines, fewer types, or one
   less indirection? Do it.

The bar: would a staff engineer approve this diff as-is, or would they ask
for one more pass? If the latter, do the pass now.
""",
    "code-review": """---
description: Code review the pending diff with explicit pass/fail criteria.
when_to_use: Before opening a PR or merging — surface real issues, not nits.
---
# Code review

Walk the diff with these checks. Surface only real issues; suppress nits.

## Correctness
- Does the code do what the description / spec / failing test says it should?
- Are edge cases handled (empty input, None, concurrent access, partial
  failure)?
- Are error paths tested or at least exercised by the new code?

## Safety
- New `subprocess`, `eval`, `pickle.loads`, raw SQL string concat, or shell
  interpolation? Check for injection.
- New file writes / deletes outside an obviously-bounded path?
- Secrets, tokens, internal hostnames in code or test fixtures?

## Maintainability
- Is the change minimal — only the lines that needed to change, changed?
- Public API additions: is each one used by a caller in this same diff? If
  not, defer them.
- New abstraction layers: is there a second concrete user, or is this YAGNI?

## Test quality
- New behavior has at least one test that would fail against `main`.
- Tests assert on observable behavior, not internal implementation details.
- No `# type: ignore`, `# noqa`, or `pytest.skip` added without a reason in
  the same line.

Pass criteria: every check above is satisfied. If any check fails, file the
issue against the diff before approving.
""",
}

# Dual-namespace placeholders; new skills should prefer ``${AURA_*}``.
_PLACEHOLDER_PAIRS: tuple[tuple[str, str], ...] = (
    ("${AURA_SKILL_DIR}", "base_dir"),
    ("${CLAUDE_SKILL_DIR}", "base_dir"),
    ("${AURA_SESSION_ID}", "session_id"),
    ("${CLAUDE_SESSION_ID}", "session_id"),
)

# Inline ``!`cmd` `` shell-exec syntax. Aura does not execute these; detected
# at load time so imported skills get a journal warning, then neutralised at
# render time so the model doesn't see them as live exec directives.
_INLINE_CMD_PATTERN = "!`"
_INLINE_CMD_REGEX = re.compile(r"!`([^`]+)`")

# Frontmatter keys ``_build_skill`` interprets. Anything else gets a
# ``skill_unsupported_frontmatter`` journal event so importing a skill with
# (e.g.) ``model:`` / ``hooks:`` has an audit trail.
_RECOGNIZED_FRONTMATTER_FIELDS: frozenset[str] = frozenset({
    "name", "description", "when_to_use", "when-to-use",
    "allowed-tools", "restrict-tools", "argument-hint", "arguments",
    "version", "paths", "user-invocable", "disable-model-invocation",
})

# Module-global conditional-skill state — lifetime-scoped so activation
# sticks across ``load_skills`` calls within one process.
_conditional_skills: dict[str, Skill] = {}
_activated_conditional_names: set[str] = set()


@contextmanager
def _bundled_skills_root(*, home_dir: Path | None = None) -> Iterator[Path | None]:
    """Materialize bundled skills into a hidden runtime root; yield its path.

    Root: ``<home>/.aura/plugins/bundled-skills/<cache-key>/skills``. Cached
    across calls within one process; rebuilt on cache-key mismatch.
    """
    global _bundled_skills_extraction
    resolved_home = (home_dir or Path.home()).resolve()

    if _bundled_skills_extraction is not None:
        cached_key, cached_root = _bundled_skills_extraction
        if cached_key == _BUNDLED_CACHE_KEY and cached_root.is_dir():
            yield cached_root
            return
        if cached_root.exists():
            shutil.rmtree(cached_root)
        _bundled_skills_extraction = None

    extracted_root = (
        resolved_home / ".aura" / "plugins" / "bundled-skills"
        / _BUNDLED_CACHE_KEY / _BUNDLED_SKILLS_EXTRACTED_ROOT_NAME
    )
    if extracted_root.exists():
        shutil.rmtree(extracted_root)
    extracted_root.parent.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)
    for skill_name, body in _BUNDLED_SKILL_FILES.items():
        skill_dir = extracted_root / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / _SKILL_FILE).write_text(body, encoding="utf-8")
    _bundled_skills_extraction = (_BUNDLED_CACHE_KEY, extracted_root)
    yield extracted_root


def load_skills(
    cwd: Path,
    *,
    home: Path | None = None,
    include_bundled: bool = False,
) -> SkillRegistry:
    """Scan managed + user + project layers; return a populated SkillRegistry.

    Conditional skills (``paths:`` frontmatter) are held in the module's
    conditional map — they enter the registry only via
    :func:`activate_conditional_skills_for_paths`.

    ``include_bundled`` defaults to False so layer-precedence tests stay
    hermetic; production Agents pass True to opt into the verify / simplify
    / code-review bundles.
    """
    home_dir = (home or Path.home()).resolve()
    cwd_resolved = cwd.resolve()

    registry = SkillRegistry()
    seen_source_paths: set[Path] = set()

    # Layer 0: managed (bundled). Loaded first so managed wins on collisions.
    if include_bundled:
        with _bundled_skills_root(home_dir=home_dir) as bundled_root:
            if bundled_root is not None:
                for skill in _load_layer(bundled_root, layer="managed"):
                    _install_or_drop(skill, registry, seen_source_paths)

    # Layer 1a: user (Aura-native) — wins on name collisions vs project.
    user_skills_root = home_dir / _AURA_DIR / _SKILLS_DIR
    for skill in _load_layer(user_skills_root, layer="user"):
        _install_or_drop(skill, registry, seen_source_paths)

    # Layer 1b: user (``~/.claude/skills``). Realpath-dedup against 1a.
    claude_user_skills_root = home_dir / _CLAUDE_DIR / _SKILLS_DIR
    if claude_user_skills_root.resolve() != user_skills_root.resolve():
        for skill in _load_layer(claude_user_skills_root, layer="user"):
            _install_or_drop(skill, registry, seen_source_paths)

    # Layer 2: project — outer-first so closer-to-cwd dirs lose on collisions.
    for project_dir in _project_dirs_up_to_home(cwd_resolved, home_dir):
        project_skills_root = project_dir / _AURA_DIR / _SKILLS_DIR
        # When cwd is inside $HOME the walk-up hits $HOME's user layer; skip.
        if project_skills_root.resolve() == user_skills_root.resolve():
            continue
        for skill in _load_layer(project_skills_root, layer="project"):
            _install_or_drop(skill, registry, seen_source_paths)

    return registry


def activate_conditional_skills_for_paths(
    paths: list[str], cwd: Path,
) -> list[str]:
    """Activate stored conditional skills whose ``paths:`` match ``paths``.

    Uses ``pathspec`` gitignore semantics. Activated skills are tracked in
    ``_activated_conditional_names`` so subsequent ``load_skills`` calls
    don't re-stash them as conditional. Returns the names newly activated.
    """
    if not _conditional_skills:
        return []
    cwd_resolved = cwd.resolve()
    activated: list[str] = []
    for name in list(_conditional_skills.keys()):
        skill = _conditional_skills[name]
        if not skill.paths:
            _activated_conditional_names.add(name)
            del _conditional_skills[name]
            activated.append(name)
            continue
        try:
            spec = pathspec.PathSpec.from_lines("gitignore", skill.paths)
        except Exception:  # noqa: BLE001 — pathspec raises many exc types
            continue
        for raw_path in paths:
            rel = _relative_to_cwd(raw_path, cwd_resolved)
            if rel is not None and spec.match_file(rel):
                _activated_conditional_names.add(name)
                del _conditional_skills[name]
                activated.append(name)
                break
    return activated


def get_conditional_skills() -> list[Skill]:
    """Return all skills currently stashed as conditional."""
    return list(_conditional_skills.values())


def activated_conditional_names() -> frozenset[str]:
    """Return the set of skills activated this session."""
    return frozenset(_activated_conditional_names)


def clear_conditional_state() -> None:
    """Reset the module-global conditional state (test hook)."""
    _conditional_skills.clear()
    _activated_conditional_names.clear()


def render_skill_body(
    skill: Skill,
    session_id: str,
    argument_values: list[str] | None = None,
) -> str:
    """Substitute skill-dir / session-id / per-arg placeholders in the body.

    Dual-namespace: ``${AURA_*}`` and ``${CLAUDE_*}`` both substitute. Inline
    ``!`cmd` `` syntax is neutralised first (Aura does not execute these);
    fenced code blocks are preserved verbatim so example docs stay intact.
    """
    body, _ = _sanitize_inline_cmds(skill.body)
    base_dir = skill.base_dir if skill.base_dir is not None else skill.source_path.parent
    replacements = {"base_dir": str(base_dir), "session_id": session_id}
    for placeholder, key in _PLACEHOLDER_PAIRS:
        body = body.replace(placeholder, replacements[key])
    values = argument_values or []
    for i, arg_name in enumerate(skill.arguments):
        body = body.replace("${" + arg_name + "}", values[i] if i < len(values) else "")
    return body


def _sanitize_inline_cmds(body: str) -> tuple[str, list[str]]:
    """Replace ``!`cmd` `` (outside fenced code blocks) with an inert note.

    Returns ``(sanitized_body, original_commands)``. Stateful line-by-line
    scan so the in-fence guard is well-defined.
    """
    lines = body.split("\n")
    in_fence = False
    fence_marker = "```"
    out_lines: list[str] = []
    originals: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(fence_marker):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue

        def _replace(m: re.Match[str]) -> str:
            cmd = m.group(1)
            originals.append(cmd)
            return f"[Aura: inline shell not supported — original: !`{cmd}`]"

        out_lines.append(_INLINE_CMD_REGEX.sub(_replace, line))
    return "\n".join(out_lines), originals


def _install_or_drop(
    skill: Skill,
    registry: SkillRegistry,
    seen_source_paths: set[Path],
) -> None:
    """Register ``skill``, stash as conditional, or drop on dup."""
    from aura.infrastructure.persistence import journal

    # Realpath dedup: same source reached via two paths (symlink overlap).
    if skill.source_path in seen_source_paths:
        journal.write(
            "skill_duplicate_skipped",
            name=skill.name, source_path=str(skill.source_path),
        )
        return
    seen_source_paths.add(skill.source_path)

    # Conditional skills go to the lazy bucket unless already activated;
    # flip ``activated`` so render-time filters see them as visible.
    if skill.is_conditional():
        if skill.name in _activated_conditional_names:
            from dataclasses import replace
            skill = replace(skill, activated=True)
        else:
            _conditional_skills[skill.name] = skill
            return

    if registry.get(skill.name) is not None:
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

    Outer-first order so closer-to-cwd skills lose on name collisions.
    """
    dirs: list[Path] = []
    current = cwd
    while True:
        dirs.append(current)
        parent = current.parent
        if parent in (current, home) or current == home:
            break
        current = parent
    dirs.reverse()
    return dirs


def _load_layer(skills_root: Path, *, layer: SkillLayer) -> list[Skill]:
    """Return parsed Skills from ``skills_root/<name>/SKILL.md`` entries."""
    if not skills_root.is_dir():
        return []
    try:
        entries = sorted(skills_root.iterdir())
    except OSError:
        return []

    legacy_files = [e for e in entries if e.is_file() and e.suffix == ".md"]
    if legacy_files:
        from aura.infrastructure.persistence import journal
        journal.write(
            "skill_legacy_format_detected",
            layer=layer, root=str(skills_root),
            files=[str(f) for f in legacy_files],
        )

    out: list[Skill] = []
    for entry in entries:
        if not entry.is_dir():
            continue
        skill_file = entry / _SKILL_FILE
        if not skill_file.is_file():
            continue
        skill = _build_skill(skill_file, layer=layer)
        if skill is not None:
            out.append(skill)
    return out


def _build_skill(skill_file: Path, *, layer: SkillLayer) -> Skill | None:
    """Parse one ``SKILL.md`` file into a Skill, or silent-skip on failure."""
    from aura.infrastructure.persistence import journal

    raw = _read_text(skill_file)
    if raw is None:
        _emit_parse_failed(skill_file, "unreadable")
        return None
    frontmatter_text, body = _split_frontmatter(raw)
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

    # Unsupported fields (e.g. ``model`` / ``hooks``) get an audit event.
    present_fields = {k for k in parsed if isinstance(k, str)}
    unsupported_fields = sorted(present_fields - _RECOGNIZED_FRONTMATTER_FIELDS)
    if unsupported_fields:
        try:
            unsupported_source = str(skill_file.resolve())
        except OSError:
            unsupported_source = str(skill_file)
        name_field = parsed.get("name")
        unsupported_name = (
            name_field if isinstance(name_field, str) else skill_file.parent.name
        )
        journal.write(
            "skill_unsupported_frontmatter",
            name=unsupported_name, source_path=unsupported_source,
            layer=layer, fields=unsupported_fields,
        )

    name_override = parsed.get("name")
    resolved_name = (
        name_override.strip()
        if isinstance(name_override, str) and name_override.strip()
        else skill_file.parent.name
    )
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

    # ``restrict-tools`` is a strict whitelist alongside the permissive
    # ``allowed-tools``; two fields, two semantics.
    paths_raw = _coerce_str_list_field(parsed.get("paths"))
    # ``foo/**`` is equivalent to ``foo`` under pathspec; collapse the
    # suffix. All-match patterns become unconditional.
    normalized_paths = [p[:-3] if p.endswith("/**") else p for p in paths_raw]
    normalized_paths = [p for p in normalized_paths if p]
    if normalized_paths and all(p == "**" for p in normalized_paths):
        normalized_paths = []

    if _INLINE_CMD_PATTERN in body:
        journal.write(
            "skill_inline_cmd_unsupported",
            name=resolved_name, source_path=str(source), layer=layer,
        )

    return Skill(
        name=resolved_name,
        description=description.strip(),
        body=body,
        source_path=source,
        base_dir=base_dir,
        layer=layer,
        when_to_use=_coerce_optional_str(parsed.get("when_to_use")),
        allowed_tools=frozenset(_coerce_str_list_field(parsed.get("allowed-tools"))),
        restrict_tools=frozenset(_coerce_str_list_field(parsed.get("restrict-tools"))),
        arguments=tuple(_coerce_str_list_field(parsed.get("arguments"))),
        argument_hint=_coerce_optional_str(parsed.get("argument-hint")),
        version=_coerce_optional_str(parsed.get("version")),
        paths=frozenset(normalized_paths),
        user_invocable=_coerce_bool(parsed.get("user-invocable"), default=True),
        disable_model_invocation=_coerce_bool(
            parsed.get("disable-model-invocation"), default=False,
        ),
    )


def _coerce_optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


def _coerce_str_list_field(value: Any) -> list[str]:
    """Accept ``list[str]`` or whitespace-separated str; drop empty items."""
    if value is None:
        return []
    if isinstance(value, str):
        return [p for p in value.split() if p]
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Parse booleans loosely (accepts ``true/yes/1`` / ``false/no/0``)."""
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
    """Return ``raw_path`` as a cwd-relative POSIX string, or None if outside."""
    p = Path(raw_path)
    if p.is_absolute():
        try:
            return p.resolve().relative_to(cwd).as_posix()
        except (ValueError, OSError):
            return None
    try:
        return (cwd / p).resolve().relative_to(cwd).as_posix()
    except ValueError:
        return None


def _emit_parse_failed(skill_file: Path, error: str) -> None:
    from aura.infrastructure.persistence import journal
    try:
        path_str = str(skill_file.resolve())
    except OSError:
        path_str = str(skill_file)
    journal.write("skill_parse_failed", path=path_str, error=error)


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None


def _split_frontmatter(raw: str) -> tuple[str | None, str]:
    """Split out the leading ``---``-fenced YAML frontmatter."""
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n").rstrip() != "---":
        return None, raw
    for idx in range(1, len(lines)):
        if lines[idx].rstrip("\r\n").rstrip() in {"---", "..."}:
            return "".join(lines[1:idx]), "".join(lines[idx + 1:])
    return None, raw
