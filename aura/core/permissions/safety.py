"""Safety policy — paths that must never be written to, regardless of rules.

Spec §6. The default list targets "irreversible damage" (git metadata, ssh
keys, shell rc files, /etc) not "please don't". Users can override with
``permissions.safety_exempt`` in ``./.aura/settings.json``.

Glob syntax is gitignore-style via ``pathspec`` (already a project dep;
see ``aura/core/memory/rules.py`` for another use). The spec §6 entry
``.git/**`` is expressed here as ``**/.git/**`` so pathspec matches a
``.git/`` directory at any depth in the tree, which is what "any .git/
anywhere" in the spec intends.

``is_protected`` is pure + defensive: no exceptions escape, no filesystem
mutations — a broken safety check must never crash the agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pathspec

# NOTE: ``**/.git/**`` instead of the spec's bare ``.git/**``. pathspec's
# gitignore treats a leading ``**/`` as "match at any depth", which is
# what "any .git/ anywhere in the tree" means. Same for ``.aura/**`` and
# ``.ssh/**``. Home-dir rc files use ``~`` which is expanded to the
# caller's home at compile time inside ``is_protected``.
DEFAULT_PROTECTED: tuple[str, ...] = (
    "**/.git/**",
    "**/.aura/**",
    "**/.ssh/**",
    "~/.bashrc",
    "~/.zshrc",
    "~/.profile",
    "~/.bash_profile",
    "~/.zprofile",
    "/etc/**",
)


@dataclass(frozen=True)
class SafetyPolicy:
    """Immutable protect/exempt glob lists consulted before any prompt.

    ``protected`` entries block writes; ``exempt`` entries are explicit
    overrides — an exempt match suppresses a protected match.
    """

    protected: tuple[str, ...]
    exempt: tuple[str, ...]


DEFAULT_SAFETY: SafetyPolicy = SafetyPolicy(protected=DEFAULT_PROTECTED, exempt=())


def is_protected(path: Path | str, policy: SafetyPolicy) -> bool:
    """True iff ``path`` lies under ``policy.protected`` and not ``policy.exempt``.

    Input handling:

    - ``path`` may be a ``Path`` or ``str``. ``~`` in the input is expanded.
    - The path is absolutized (relative → cwd-joined) AND resolved (symlinks
      followed). Patterns are matched against BOTH — a symlink pointing into
      a protected dir is blocked (resolve catches it) AND a direct write to
      e.g. ``/etc/passwd`` is blocked even on macOS where ``/etc`` resolves
      to ``/private/etc`` (absolute-only form catches it).
    - Policy patterns containing a leading ``~`` are expanded against
      ``Path.home()`` before matching.

    Defensive contract: any unexpected input type, OS error during resolve,
    or pathspec compile error returns ``False`` rather than propagating.
    A broken safety check must never be the thing that crashes the agent.
    """
    try:
        candidates = _candidate_paths(path)
        if not candidates:
            return False

        protected_spec = _compile_spec(policy.protected)
        exempt_spec = _compile_spec(policy.exempt)

        if any(exempt_spec.match_file(t) for t in candidates):
            return False
        return any(protected_spec.match_file(t) for t in candidates)
    except Exception:  # noqa: BLE001 — safety must never crash the agent
        return False


def _candidate_paths(path: Any) -> list[str]:
    """Normalize ``path`` to the match-ready string forms.

    Returns up to two strings: the absolute path without symlink resolution,
    and the absolute path with symlinks followed. The dual match covers two
    real cases glob patterns must hit:

    - ``/etc/passwd`` vs ``/private/etc/passwd`` on macOS (resolve rewrites).
    - A symlink pointing INTO a protected dir (resolve catches the target).

    Returns ``[]`` for garbage input (``None``, empty string, non-str/Path)
    so the caller short-circuits to False.
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

    # Absolutize without following symlinks (Path.absolute preserves links).
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
    """Compile a tuple of globs, expanding ``~`` in each pattern first."""
    home = str(Path.home())
    expanded = tuple(
        pat.replace("~", home, 1) if pat.startswith("~") else pat
        for pat in patterns
    )
    return pathspec.PathSpec.from_lines("gitignore", expanded)
