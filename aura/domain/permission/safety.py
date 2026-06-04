"""Safety-protected path lists; matching lives in application.permission.safety."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PROTECTED_WRITES: tuple[str, ...] = (
    "**/.git",
    "**/.git/**",
    "**/.aura",
    "**/.aura/**",
    "**/.ssh",
    "**/.ssh/**",
    "~/.bashrc",
    "~/.zshrc",
    "~/.profile",
    "~/.bash_profile",
    "~/.zprofile",
    "/etc",
    "/etc/**",
)

DEFAULT_PROTECTED_READS: tuple[str, ...] = (
    "**/.ssh",
    "**/.ssh/**",
    "~/.bashrc",
    "~/.zshrc",
    "~/.profile",
    "~/.bash_profile",
    "~/.zprofile",
    "/etc",
    "/etc/**",
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
