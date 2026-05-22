"""Safety policy value objects — paths that bypass rules entirely.

Two lists, two directions:

- **writes** (``DEFAULT_PROTECTED_WRITES``): blast-radius list. Fires on
  tools with ``is_destructive=True``. Includes ``.git/``, ``.aura/``,
  shell rc files, ``/etc``.
- **reads** (``DEFAULT_PROTECTED_READS``): narrower — paths whose
  contents are secrets. Fires on ANY tool with a resolvable path arg.
  ``.git/`` and ``.aura/`` are absent on purpose (legitimate reads).

``.git/`` paths use ``**/.git/**`` so pathspec matches a ``.git/``
directory at any depth. Both bare-dir (``**/.git``) and ``/**`` shapes
are required because pathspec's ``**/X/**`` matches FILES INSIDE X/,
not X itself.

The matching function ``is_protected`` lives in
``aura.application.permission.safety`` because it journals errors.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PROTECTED_WRITES: tuple[str, ...] = (
    "**/.git", "**/.git/**",
    "**/.aura", "**/.aura/**",
    "**/.ssh", "**/.ssh/**",
    "~/.bashrc",
    "~/.zshrc",
    "~/.profile",
    "~/.bash_profile",
    "~/.zprofile",
    "/etc", "/etc/**",
)

DEFAULT_PROTECTED_READS: tuple[str, ...] = (
    "**/.ssh", "**/.ssh/**",
    "~/.bashrc",
    "~/.zshrc",
    "~/.profile",
    "~/.bash_profile",
    "~/.zprofile",
    "/etc", "/etc/**",
)


@dataclass(frozen=True)
class SafetyPolicy:
    protected_writes: tuple[str, ...]
    protected_reads: tuple[str, ...]
    exempt: tuple[str, ...]


DEFAULT_SAFETY: SafetyPolicy = SafetyPolicy(
    protected_writes=DEFAULT_PROTECTED_WRITES,
    protected_reads=DEFAULT_PROTECTED_READS,
    exempt=(),
)
