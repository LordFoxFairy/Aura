"""`.aura/rules/*.md` discovery + frontmatter parse + glob match; broken rules drop silently."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pathspec
import yaml

_AURA_DIR = ".aura"
_RULES_DIR = "rules"
_MD_SUFFIX = ".md"

_MAX_LINES = 200
_DEFAULT_BYTE_CAP = 25_000


def _truncation_warning(actual_bytes: int, limit: int) -> str:
    return (
        f"\nWARNING: this file is {actual_bytes} bytes (limit: {limit}). "
        "Keep memory files under 25 KB; split long content into separate files."
    )


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


# Single event-loop — no concurrent writes, no lock needed.
_rules_cache: dict[Path, RulesBundle] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_rules(cwd: Path, *, force_reload: bool = False) -> RulesBundle:
    resolved_cwd = cwd.resolve()
    if not force_reload and resolved_cwd in _rules_cache:
        return _rules_cache[resolved_cwd]

    bundle = RulesBundle()

    home = Path.home()
    _scan_layer(home / _AURA_DIR / _RULES_DIR, base_dir=home, bundle=bundle)

    # Project layer is cwd-only: rules are project-level, not path-level —
    # walk-up would leak ancestor project's rules into child projects.
    _scan_layer(
        resolved_cwd / _AURA_DIR / _RULES_DIR, base_dir=resolved_cwd, bundle=bundle
    )

    _warn_out_of_cwd_rules(bundle, resolved_cwd)

    _rules_cache[resolved_cwd] = bundle
    return bundle


def clear_cache(cwd: Path | None = None) -> None:
    if cwd is None:
        _rules_cache.clear()
        return
    _rules_cache.pop(cwd.resolve(), None)


def match(bundle: RulesBundle, path: Path) -> list[Rule]:
    try:
        resolved_path = path.resolve()
    except OSError:
        resolved_path = path

    seen: set[Path] = set()
    matched: list[Rule] = []
    for rule in bundle.conditional:
        if rule.source_path in seen:
            continue
        if _rule_matches_path(rule, resolved_path):
            matched.append(rule)
            seen.add(rule.source_path)

    matched.sort(key=lambda r: r.source_path)
    return matched


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scan_layer(rules_root: Path, *, base_dir: Path, bundle: RulesBundle) -> None:
    if not rules_root.is_dir():
        return
    try:
        md_files = sorted(rules_root.rglob(f"*{_MD_SUFFIX}"))
    except OSError:
        return

    for md_path in md_files:
        # rglob can return directories named "x.md".
        if not md_path.is_file():
            continue
        rule = _build_rule(md_path, base_dir=base_dir)
        if rule is None:
            continue
        if rule.globs:
            bundle.conditional.append(rule)
        else:
            bundle.unconditional.append(rule)


def _build_rule(md_path: Path, *, base_dir: Path) -> Rule | None:
    raw = _read_text(md_path)
    if raw is None:
        return None

    frontmatter_text, body = _split_frontmatter(raw)

    if frontmatter_text is None:
        globs: tuple[str, ...] = ()
    else:
        try:
            parsed = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as exc:
            from aura.core import journal

            journal.write(
                "rule_yaml_parse_failed", path=str(md_path), error=str(exc)
            )
            return None
        globs_or_skip = _extract_globs(parsed)
        if globs_or_skip is _SKIP:
            from aura.core import journal

            actual_type = type(parsed["paths"]).__name__ if isinstance(
                parsed, dict
            ) and "paths" in parsed else type(parsed).__name__
            journal.write(
                "rule_paths_invalid_type",
                path=str(md_path),
                actual_type=actual_type,
            )
            return None
        globs = globs_or_skip  # type: ignore[assignment]  # narrowing branch mypy doesn't track

    try:
        source = md_path.resolve()
        base = base_dir.resolve()
    except OSError:
        return None

    return Rule(
        source_path=source,
        base_dir=base,
        globs=globs,
        content=_truncate(body),
    )


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")


def _split_frontmatter(raw: str) -> tuple[str | None, str]:
    """Split `---\\n…\\n---` (or `...`) head from body; missing close = no frontmatter."""
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


# Distinguishes "no `paths` field (→ unconditional)" from "unknown type (→ skip)".
_SKIP = object()


def _extract_globs(parsed: Any) -> tuple[str, ...] | object:
    if not isinstance(parsed, dict):
        return ()
    if "paths" not in parsed:
        return ()
    value = parsed["paths"]
    if isinstance(value, str):
        parts = tuple(p.strip() for p in value.split(",") if p.strip())
        return _normalize_universal(parts)
    if isinstance(value, list):
        parts = tuple(str(item) for item in value)
        return _normalize_universal(parts)
    return _SKIP


_UNIVERSAL_GLOBS = frozenset({"**", "**/*"})


def _normalize_universal(parts: tuple[str, ...]) -> tuple[str, ...]:
    if parts and all(p in _UNIVERSAL_GLOBS for p in parts):
        return ()
    return parts


def _truncate(body: str, *, byte_cap: int = _DEFAULT_BYTE_CAP) -> str:
    """Truncate at 200 lines or `byte_cap` bytes; append WARNING to guide user to split."""
    original_bytes = len(body.encode("utf-8"))
    truncated = False
    limit_hit = byte_cap

    lines = body.splitlines(keepends=True)
    if len(lines) > _MAX_LINES:
        lines = lines[:_MAX_LINES]
        truncated = True

    trimmed = "".join(lines)
    encoded = trimmed.encode("utf-8")
    if len(encoded) > byte_cap:
        # errors='replace' guards against splitting a multi-byte char at byte_cap.
        trimmed = encoded[:byte_cap].decode("utf-8", errors="replace")
        truncated = True

    if truncated:
        return trimmed + _truncation_warning(original_bytes, limit_hit)
    return trimmed


def _rule_matches_path(rule: Rule, resolved_path: Path) -> bool:
    match_target = _relative_or_absolute(resolved_path, rule.base_dir)
    for glob in rule.globs:
        # pathspec raises various types (GitIgnorePatternError, re.error, …) on
        # malformed globs — all swallowed and journaled.
        try:
            spec = pathspec.PathSpec.from_lines("gitignore", [glob])
            if spec.match_file(match_target):
                return True
        except Exception as exc:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
            from aura.core import journal

            journal.write(
                "rule_glob_compile_failed",
                path=str(rule.source_path),
                glob=glob,
                error=str(exc),
            )
            continue
    return False


def _warn_out_of_cwd_rules(bundle: RulesBundle, cwd: Path) -> None:
    """Journal-warn rules whose absolute globs anchor outside `cwd` — they'd never trigger."""
    for rule in bundle.conditional:
        offenders: list[str] = []
        for glob in rule.globs:
            if not glob.startswith("/"):
                continue
            prefix = _glob_static_prefix(glob)
            try:
                if not Path(prefix).resolve().is_relative_to(cwd):
                    offenders.append(glob)
            except (OSError, ValueError):
                offenders.append(glob)
        if offenders:
            from aura.core import journal

            journal.write(
                "out_of_cwd_rule_warning",
                path=str(rule.source_path),
                patterns=offenders,
                cwd=str(cwd),
            )


def _glob_static_prefix(glob: str) -> str:
    """Strip wildcard tail: `/tmp/x/**/*.py` → `/tmp/x`; degenerate `/` stays `/`."""
    cut = len(glob)
    for i, ch in enumerate(glob):
        if ch in "*?[":
            cut = i
            break
    prefix = glob[:cut].rstrip("/")
    return prefix or "/"


def _relative_or_absolute(path: Path, base_dir: Path) -> str:
    try:
        rel = path.relative_to(base_dir)
        return rel.as_posix()
    except ValueError:
        return path.as_posix()
