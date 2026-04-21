"""Safety policy — paths that bypass rules entirely.

Spec §6. Two lists, two directions:

- **writes** (``DEFAULT_PROTECTED_WRITES``) — "do not clobber this".
  Git metadata, aura config, shell rc files, ``/etc``. The destructive
  blast-radius list; fires only on tools whose metadata declares
  ``is_destructive=True``.
- **reads** (``DEFAULT_PROTECTED_READS``) — "do not let the model see this".
  A narrower subset: ssh keys, shell rc files, ``/etc``. Fires on ANY tool
  with a resolvable path arg, destructive or not. The reads list closes
  the Plan B refactor hole where ``read_file("~/.ssh/id_rsa")`` used to
  auto-allow because the old ``is_read_only`` branch skipped safety.

Why ``.git/`` and ``.aura/`` are writes-only
---------------------------------------------

An agent legitimately *reads* ``.git/HEAD`` (to check branch state) and
``.aura/settings.json`` (self-introspection). These aren't secrets. What
matters is that writes to them are irreversible damage. So they stay on
the writes list and stay off the reads list — asymmetric on purpose.

Glob syntax is gitignore-style via ``pathspec``. The spec §6 entry
``.git/**`` is expressed as ``**/.git/**`` so pathspec matches a ``.git/``
directory at any depth in the tree — same convention for ``.aura/`` and
``.ssh/``.

``is_protected`` is pure + defensive: no exceptions escape, no filesystem
mutations. A broken safety check must never crash the agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pathspec

# Writes list — "do not clobber this".
# ``**/.git/**`` (not ``.git/**``) because pathspec's gitignore treats a
# leading ``**/`` as "match at any depth", which is what "any .git/
# anywhere in the tree" means. Same convention for ``.aura/`` and
# ``.ssh/``. Home-dir rc files use ``~``, expanded at compile time inside
# ``is_protected``.
DEFAULT_PROTECTED_WRITES: tuple[str, ...] = (
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

# Reads list — "do not let the model see this".
# Narrower than writes: ``.git/`` and ``.aura/`` are legitimate read
# targets for an agent (git log, self-config), so they're absent here on
# purpose. Only paths whose contents are secrets remain.
DEFAULT_PROTECTED_READS: tuple[str, ...] = (
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

    ``protected_writes`` entries block destructive tools; ``protected_reads``
    entries block any tool with a path arg (destructive or not). A single
    ``exempt`` list covers both directions — an explicit override that
    suppresses any matching protected entry.
    """

    protected_writes: tuple[str, ...]
    protected_reads: tuple[str, ...]
    exempt: tuple[str, ...]


DEFAULT_SAFETY: SafetyPolicy = SafetyPolicy(
    protected_writes=DEFAULT_PROTECTED_WRITES,
    protected_reads=DEFAULT_PROTECTED_READS,
    exempt=(),
)


def is_protected(
    path: Path | str,
    policy: SafetyPolicy,
    *,
    is_write: bool,
) -> bool:
    """True iff ``path`` is blocked by the direction-appropriate list.

    - ``is_write=True`` consults ``policy.protected_writes``.
    - ``is_write=False`` consults ``policy.protected_reads``.

    ``policy.exempt`` applies in both directions — an exempt match
    suppresses a protected match regardless of which list fired.

    Input handling:

    - ``path`` may be a ``Path`` or ``str``. ``~`` in the input is expanded.
    - The path is absolutized (relative → cwd-joined) AND resolved (symlinks
      followed). Patterns are matched against BOTH — a symlink pointing
      into a protected dir is blocked (resolve catches it) AND a direct
      write to e.g. ``/etc/passwd`` is blocked even on macOS where ``/etc``
      resolves to ``/private/etc`` (absolute-only form catches it).
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

        active = policy.protected_writes if is_write else policy.protected_reads
        protected_spec = _compile_spec(active)
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
