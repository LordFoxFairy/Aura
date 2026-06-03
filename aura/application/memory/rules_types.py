"""Value types for `.aura/rules/*.md` — leaf, no loader/parse deps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    source_path: Path
    base_dir: Path
    # Empty tuple = unconditional (no `paths` in frontmatter).
    globs: tuple[str, ...]
    content: str


@dataclass
class RulesBundle:
    unconditional: list[Rule] = field(default_factory=list)
    conditional: list[Rule] = field(default_factory=list)
