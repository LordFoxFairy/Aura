"""Bash-command safety — Tier A hard floors no mode or rule can override.

Pure policy (no I/O, no journaling). Failure-mode inversion: a bug in our
own parser returns ``None`` rather than raising — a typo here would
otherwise block every bash call. Rule-level signals (``shlex.split``
raising as rule 12's trigger) still fail closed.
"""

from __future__ import annotations

from aura.application.permission.bash_safety_lex import _expand_braces
from aura.application.permission.bash_safety_rules import (
    ZSH_DANGEROUS_COMMANDS,
    BashSafetyViolation,
    Reason,
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

__all__ = [
    "ZSH_DANGEROUS_COMMANDS",
    "BashSafetyViolation",
    "Reason",
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
    """Return the first Tier A violation, or None.

    Order (tested contract):
      1. cr_outside_double_quote
      2. command_substitution
      3. obfuscated_execution
      4. pipe_to_shell
      5. zsh_dangerous_command
      6. exec_destructive
      7. destructive_removal
      8. world_writable_chmod
      9. root_chown
     10. sed_inplace_system_path
     11. redirect_to_system_path
     12. malformed_with_separator
     13. cd_git_compound

    Rule 2 precedes rule 5 because ``$(zmodload x)`` tokenizes such that the
    zsh rule never sees ``zmodload`` as a first token, yet bash expands the
    substitution at runtime. Rule 3 precedes rule 4 because base64-into-shell
    is a strict subset of pipe-to-shell but its dedicated error message
    guides the model better.
    """
    if not command:
        return None

    try:
        for check in _CHECKS:
            violation = check(command)
            if violation is not None:
                return violation
        return None
    except Exception:  # noqa: BLE001 — failure-mode inversion; see module docstring
        return None
