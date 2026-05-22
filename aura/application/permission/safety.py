"""Safety check — direction-aware path matching with journaled failures.

Pure value objects (``SafetyPolicy``, ``DEFAULT_SAFETY``,
``DEFAULT_PROTECTED_*``) live in ``aura.domain.permission.safety``; this
module owns the I/O-bearing matcher.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pathspec

from aura.domain.permission.safety import SafetyPolicy
from aura.infrastructure.persistence import journal


def is_protected(
    path: Path | str,
    policy: SafetyPolicy,
    *,
    is_write: bool,
) -> bool:
    """True iff ``path`` is blocked by the direction-appropriate list.

    Matches against BOTH the absolute (symlink-preserving) path and the
    resolved (symlink-followed) path — catches a symlink into a
    protected dir AND macOS ``/etc`` vs ``/private/etc`` rewrites.

    ``policy.exempt`` overrides in both directions. Any exception during
    pathspec compile or path resolution returns False and journals the
    error: a broken safety check must never crash the agent.
    """
    try:
        candidates = _candidate_paths(path)
        if not candidates:
            return False

        active = policy.protected_writes if is_write else policy.protected_reads
        protected_spec = _compile_spec(active)
        exempt_spec = _compile_spec(policy.exempt)

        if any(exempt_spec.match_file(t) for t in candidates):
            return False
        return any(protected_spec.match_file(t) for t in candidates)
    except Exception as exc:  # noqa: BLE001 — safety must never crash the agent
        journal.write(
            "safety_check_error",
            path=str(path) if path is not None else None,
            is_write=is_write,
            detail=f"{type(exc).__name__}: {exc}",
        )
        return False


def _candidate_paths(path: Any) -> list[str]:
    """Up to two strings — absolute (no symlink resolve) and resolved.

    Returns ``[]`` for garbage input so the caller short-circuits.
    """
    if isinstance(path, Path):
        candidate = path
    elif isinstance(path, str):
        if not path:
            return []
        candidate = Path(path)
    else:
        return []

    try:
        expanded = candidate.expanduser()
    except (RuntimeError, OSError):
        return []

    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    absolute_str = os.path.normpath(str(absolute))

    try:
        resolved_str = str(expanded.resolve(strict=False))
    except (OSError, RuntimeError):
        resolved_str = absolute_str

    if resolved_str == absolute_str:
        return [absolute_str]
    return [absolute_str, resolved_str]


def _compile_spec(patterns: tuple[str, ...]) -> pathspec.PathSpec:
    home = str(Path.home())
    expanded = tuple(
        pat.replace("~", home, 1) if pat.startswith("~") else pat
        for pat in patterns
    )
    return pathspec.PathSpec.from_lines("gitignore", expanded)
