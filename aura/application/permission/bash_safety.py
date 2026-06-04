"""Bash-command safety — Tier A hard floors no mode or rule can override."""

from __future__ import annotations

from aura.application.permission.bash_safety_lex import _expand_braces
from aura.application.permission.bash_safety_rules import (
    ZSH_DANGEROUS_COMMANDS,
    _check_cd_git_compound,
    _check_command_substitution,
    _check_cr_outside_quotes,
    _check_destructive_removal,
    _check_exec_destructive,
    _check_malformed_with_separator,
    _check_obfuscated_execution,
    _check_pipe_to_shell,
    _check_redirect_to_system_path,
    _check_root_chown,
    _check_sed_inplace_system_path,
    _check_world_writable_chmod,
    _check_zsh_dangerous,
)
from aura.application.permission.bash_safety_types import BashSafetyViolation

__all__ = [
    "ZSH_DANGEROUS_COMMANDS",
    "BashSafetyViolation",
    "_expand_braces",
    "check_bash_safety",
]

_CHECKS = (
    _check_cr_outside_quotes,
    _check_command_substitution,
    _check_obfuscated_execution,
    _check_pipe_to_shell,
    _check_zsh_dangerous,
    _check_exec_destructive,
    _check_destructive_removal,
    _check_world_writable_chmod,
    _check_root_chown,
    _check_sed_inplace_system_path,
    _check_redirect_to_system_path,
    _check_malformed_with_separator,
    _check_cd_git_compound,
)


def check_bash_safety(command: str) -> BashSafetyViolation | None:
    """Return the first Tier A violation in ``_CHECKS`` order, or None."""
    if not command:
        return None

    try:
        for check in _CHECKS:
            violation = check(command)
            if violation is not None:
                return violation
        return None
    except Exception:  # noqa: BLE001 — a parser bug must not block every bash call
        return None
