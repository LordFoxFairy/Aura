"""Safety-protected path lists; matching lives in application.permission.safety.

Two invariants:
  - WRITES fires on is_destructive=True tools (.git/, .aura/, rc files, /etc).
  - READS is narrower (secret-content paths); .git/ and .aura/ are absent on
    purpose so legitimate reads are not blocked.

Both bare-dir (**/.git) and (**/.git/**) globs are required because pathspec
matches files INSIDE X/, not X itself.
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
