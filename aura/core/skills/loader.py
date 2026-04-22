"""Discover + parse skill files from user + project layers.

Two layers, scanned flat (no recursion, no walk-up):

1. ``~/.aura/skills/*.md`` — user, wins on name collisions.
2. ``<cwd>/.aura/skills/*.md`` — project.

Required frontmatter: ``name`` (str) and ``description`` (str). Any violation
(missing field, wrong type, malformed YAML, non-UTF-8 body) causes the skill
to be silently skipped and a ``skill_parse_failed`` journal event emitted.

Name collisions: user-layer wins. The losing project-layer skill is dropped
and a ``skill_name_collision`` journal event is emitted.

Frontmatter parsing mirrors :mod:`aura.core.memory.rules` — the splitter is
copy-pasted rather than imported so the two subsystems don't grow a cross-cut
dependency on shared scanning internals.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from aura.core.skills.registry import SkillRegistry
from aura.core.skills.types import Skill, SkillLayer

_AURA_DIR = ".aura"
_SKILLS_DIR = "skills"
_MD_SUFFIX = ".md"


def load_skills(cwd: Path, *, home: Path | None = None) -> SkillRegistry:
    """Scan user + project layers and return a populated SkillRegistry.

    ``home`` defaults to ``Path.home()``; exposed as kwarg for tests.
    """
    home_dir = (home if home is not None else Path.home())
    registry = SkillRegistry()

    # User layer first — its name wins on collisions.
    for skill in _scan_layer(home_dir / _AURA_DIR / _SKILLS_DIR, layer="user"):
        # First writer wins; user layer has no internal collisions because
        # filesystem names are unique per directory. If two user skills share
        # frontmatter name, the second one emits a journal event and is
        # dropped — same machinery as the cross-layer collision below.
        if registry.get(skill.name) is not None:
            from aura.core import journal

            journal.write(
                "skill_name_collision",
                name=skill.name,
                kept_path=str(registry.get(skill.name).source_path),  # type: ignore[union-attr]
                dropped_path=str(skill.source_path),
            )
            continue
        registry.register(skill)

    for skill in _scan_layer(
        cwd.resolve() / _AURA_DIR / _SKILLS_DIR, layer="project"
    ):
        existing = registry.get(skill.name)
        if existing is not None:
            from aura.core import journal

            journal.write(
                "skill_name_collision",
                name=skill.name,
                kept_path=str(existing.source_path),
                dropped_path=str(skill.source_path),
            )
            continue
        registry.register(skill)

    return registry


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _scan_layer(skills_root: Path, *, layer: SkillLayer) -> list[Skill]:
    """Return parsed skills from ``skills_root/*.md`` (flat, no recursion)."""
    if not skills_root.is_dir():
        return []
    try:
        md_files = sorted(skills_root.glob(f"*{_MD_SUFFIX}"))
    except OSError:
        return []

    out: list[Skill] = []
    for md_path in md_files:
        if not md_path.is_file():
            continue
        skill = _build_skill(md_path, layer=layer)
        if skill is not None:
            out.append(skill)
    return out


def _build_skill(md_path: Path, *, layer: SkillLayer) -> Skill | None:
    """Parse one ``.md`` file into a Skill, or silent-skip with a journal event."""
    raw = _read_text(md_path)
    if raw is None:
        _emit_parse_failed(md_path, "unreadable")
        return None

    frontmatter_text, body = _split_frontmatter(raw)
    if frontmatter_text is None:
        _emit_parse_failed(md_path, "missing frontmatter")
        return None

    try:
        parsed = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        _emit_parse_failed(md_path, f"yaml error: {exc}")
        return None

    if not isinstance(parsed, dict):
        _emit_parse_failed(md_path, "frontmatter is not a mapping")
        return None

    name = parsed.get("name")
    description = parsed.get("description")
    if not isinstance(name, str) or not name.strip():
        _emit_parse_failed(md_path, "missing or non-string 'name'")
        return None
    if not isinstance(description, str) or not description.strip():
        _emit_parse_failed(md_path, "missing or non-string 'description'")
        return None

    try:
        source = md_path.resolve()
    except OSError:
        _emit_parse_failed(md_path, "resolve failed")
        return None

    return Skill(
        name=name.strip(),
        description=description.strip(),
        body=body,
        source_path=source,
        layer=layer,
    )


def _emit_parse_failed(md_path: Path, error: str) -> None:
    from aura.core import journal

    try:
        path_str = str(md_path.resolve())
    except OSError:
        path_str = str(md_path)
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
    """Split out the leading ``---``-fenced YAML frontmatter.

    Copy of :func:`aura.core.memory.rules._split_frontmatter` (kept local to
    avoid an inter-subsystem dependency on rules' private helpers).
    """
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
