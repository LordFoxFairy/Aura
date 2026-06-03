"""Discover + parse skill directories from user + project layers.

Layout: one directory per skill, ``<name>/SKILL.md`` inside it. Layers in
load order (user wins on name collisions):

1. Managed (bundled) — opt-in via ``include_bundled=True`` (AgentSession.__init__).
2. User — ``~/.aura/skills/<name>/SKILL.md`` and ``~/.claude/skills/``.
3. Project — walk up from ``cwd`` to ``Path.home()`` exclusive.

Conditional skills (``paths:`` frontmatter) are stashed; activation moves
them into the live registry on a matching file touch.

Required frontmatter: ``description``. ``name`` overrides the dir name.
Missing description → silent skip + ``skill_parse_failed`` journal event.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from aura.domain.skill import Skill, SkillLayer
from aura.infrastructure.persistence import journal
from aura.infrastructure.skills.registry import SkillRegistry
from aura.infrastructure.skills.skills_bundled import _bundled_skills_root
from aura.infrastructure.skills.skills_conditional import (
    activate_conditional_skills_for_paths,
    activated_conditional_names,
    clear_conditional_state,
    get_conditional_skills,
    is_activated,
    stash_conditional,
)

__all__ = [
    "activate_conditional_skills_for_paths",
    "activated_conditional_names",
    "clear_conditional_state",
    "get_conditional_skills",
    "load_skills",
    "render_skill_body",
]

_AURA_DIR = ".aura"
_CLAUDE_DIR = ".claude"
_SKILLS_DIR = "skills"
_SKILL_FILE = "SKILL.md"

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
        if is_activated(skill.name):
            skill = replace(skill, activated=True)
        else:
            stash_conditional(skill)
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


def _emit_parse_failed(skill_file: Path, error: str) -> None:
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
