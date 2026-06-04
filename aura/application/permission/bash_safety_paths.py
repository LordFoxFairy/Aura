"""System-path classification for bash-safety rules."""

from __future__ import annotations

import os

from aura.application.permission.bash_safety_lex import _expand_braces

# System-owned prefixes — writing under any is a Tier A hard floor even with user grant.
_SYSTEM_PATH_PREFIXES: tuple[str, ...] = (
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/boot",
    "/sys",
    "/proc",
    "/dev",
    "/root",
    "/lib",
    "/lib64",
    "/System",
    "/Library",
)

# /dev/... targets that are legitimate output discards / tty handles.
_SAFE_DEV_TARGETS: frozenset[str] = frozenset(
    {
        "/dev/null",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/tty",
        "/dev/fd",
    }
)


def _is_system_path(path: str) -> bool:
    """True if any tilde/env-var/brace expansion of ``path`` lands under a system prefix."""
    if not path:
        return False
    cleaned = path.strip("\"'")
    cleaned = os.path.expandvars(os.path.expanduser(cleaned))
    candidates = _expand_braces(cleaned)
    return any(_check_single_path(c) for c in candidates)


def _check_single_path(cleaned: str) -> bool:
    if cleaned in _SAFE_DEV_TARGETS:
        return False
    if cleaned.startswith("/dev/fd/"):
        return False
    for prefix in _SYSTEM_PATH_PREFIXES:
        if cleaned == prefix or cleaned.startswith(prefix + "/"):
            return True
    return False
