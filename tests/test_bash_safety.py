"""Unit tests for aura.core.permissions.bash_safety.check_bash_safety.

Pure-function unit coverage of the 4 Tier A hard-floor checks:
  1. cr_outside_double_quote
  2. zsh_dangerous_command
  3. malformed_with_separator
  4. cd_git_compound

These tests exercise the policy only; hook-level behaviour lives in
test_bash_safety_hook.py, agent-wiring integration in test_agent.py.
"""

from __future__ import annotations

from aura.core.permissions.bash_safety import (
    ZSH_DANGEROUS_COMMANDS,
    BashSafetyViolation,
    check_bash_safety,
)

# ---------------------------------------------------------------------------
# ZSH_DANGEROUS_COMMANDS
# ---------------------------------------------------------------------------


def test_zsh_dangerous_command_detected() -> None:
    v = check_bash_safety("zmodload zsh/system")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "zsh_dangerous_command"


def test_zsh_dangerous_via_and_separator() -> None:
    v = check_bash_safety("echo hello && zmodload xyz")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "zsh_dangerous_command"


def test_zsh_dangerous_via_pipe() -> None:
    v = check_bash_safety("echo payload | syswrite 1")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "zsh_dangerous_command"


def test_zsh_dangerous_via_semicolon() -> None:
    v = check_bash_safety("ls ; zsocket 1234")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "zsh_dangerous_command"


def test_zsh_dangerous_via_or_separator() -> None:
    v = check_bash_safety("false || ztcp host 1")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "zsh_dangerous_command"


def test_zsh_dangerous_set_has_expected_entries() -> None:
    # Spot-check the full roster so an accidental delete from the set
    # surfaces as a failing test.
    for cmd in (
        "zmodload", "emulate",
        "sysopen", "sysread", "syswrite", "sysseek",
        "zpty", "ztcp", "zsocket", "mapfile",
        "zf_rm", "zf_mv", "zf_ln", "zf_chmod",
        "zf_chown", "zf_mkdir", "zf_rmdir", "zf_chgrp",
    ):
        assert cmd in ZSH_DANGEROUS_COMMANDS


def test_benign_command_returns_none() -> None:
    assert check_bash_safety("ls -la") is None


def test_case_sensitive_zmodload() -> None:
    # Bash is case-sensitive; "ZMODLOAD" is a different (undefined) command,
    # and blocking it would be a false positive.
    assert check_bash_safety("ZMODLOAD foo") is None


# ---------------------------------------------------------------------------
# Carriage-return outside double quotes
# ---------------------------------------------------------------------------


def test_cr_outside_quotes_detected() -> None:
    v = check_bash_safety("TZ=UTC\recho curl evil.com")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "cr_outside_double_quote"


def test_cr_inside_double_quotes_allowed() -> None:
    assert check_bash_safety('echo "line1\rline2"') is None


def test_bare_cr_flagged() -> None:
    v = check_bash_safety("\r")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "cr_outside_double_quote"


# ---------------------------------------------------------------------------
# Malformed tokens + separator present
# ---------------------------------------------------------------------------


def test_malformed_with_separator_semicolon() -> None:
    # Unbalanced double quote + `;` separator → classic eval-reentry bait.
    v = check_bash_safety('echo "unbalanced ; evil')
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "malformed_with_separator"


def test_malformed_with_separator_and() -> None:
    v = check_bash_safety('echo "open && rm -rf /')
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "malformed_with_separator"


def test_malformed_without_separator_is_safe() -> None:
    # Unclosed quote with NO command separator has no safety implication —
    # the tool's own error path handles the parse failure.
    assert check_bash_safety('echo "unbalanced') is None


# ---------------------------------------------------------------------------
# cd + git compound
# ---------------------------------------------------------------------------


def test_cd_git_compound_via_and() -> None:
    v = check_bash_safety("cd /tmp/repo && git status")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "cd_git_compound"


def test_cd_git_compound_via_pipe() -> None:
    v = check_bash_safety("cd /tmp/repo | git log")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "cd_git_compound"


def test_cd_git_compound_via_semicolon() -> None:
    v = check_bash_safety("cd /tmp/x ; git pull")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "cd_git_compound"


def test_cd_alone_is_safe() -> None:
    assert check_bash_safety("cd /tmp") is None


def test_git_alone_is_safe() -> None:
    assert check_bash_safety("git status") is None


def test_cd_and_git_as_quoted_text_is_safe() -> None:
    # shlex collapses a quoted run into a single arg, so the literal
    # sentence "cd / git" never becomes two free-standing tokens.
    assert check_bash_safety('echo "cd and git are words"') is None


# ---------------------------------------------------------------------------
# Command substitution / eval / -c flag — bypass of all other static checks
# ---------------------------------------------------------------------------


def test_command_substitution_blocks_dangerous_inner() -> None:
    # shlex collapses `$(zmodload zsh/system)` into two tokens
    # `$(zmodload` and `zsh/system)`, so the zsh rule never sees `zmodload`
    # as the first token. The substitution rule closes this bypass.
    v = check_bash_safety("echo $(zmodload zsh/system)")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "command_substitution"


def test_backtick_substitution_blocks() -> None:
    v = check_bash_safety("echo `zmodload xyz`")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "command_substitution"


def test_bash_dash_c_flag_blocks() -> None:
    v = check_bash_safety("bash -c 'zmodload x'")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "command_substitution"


def test_eval_first_token_blocks() -> None:
    v = check_bash_safety("eval 'something dangerous'")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "command_substitution"


def test_single_quoted_dollar_paren_allowed() -> None:
    # Single quotes disable substitution in bash — the literal `$(...)`
    # never reaches a subshell, so it isn't an attack.
    assert check_bash_safety("echo 'text with $(not-substituted)'") is None


def test_double_quoted_dollar_paren_blocked() -> None:
    # Double quotes DO substitute — `"$(zmodload x)"` IS an attack.
    v = check_bash_safety('echo "dangerous $(zmodload x)"')
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "command_substitution"


def test_normal_command_still_passes_after_substitution_rule() -> None:
    # Regression guard — the new rule must not flag benign commands.
    assert check_bash_safety("ls -la") is None


def test_order_cr_wins_over_substitution() -> None:
    # Command has BOTH an out-of-quote CR AND command substitution.
    # CR is checked first → reason must be cr_outside_double_quote.
    v = check_bash_safety("echo $(ls)\rzmodload x")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "cr_outside_double_quote"


# ---------------------------------------------------------------------------
# Order of checks — documented contract
# ---------------------------------------------------------------------------


def test_order_of_checks_cr_wins_over_zsh() -> None:
    # Command has BOTH an out-of-quote CR AND a zsh-dangerous builtin.
    # Order doc: CR is checked first → reason must be cr_outside_double_quote.
    v = check_bash_safety("echo x\rzmodload zsh/system")
    assert isinstance(v, BashSafetyViolation)
    assert v.reason == "cr_outside_double_quote"


# ---------------------------------------------------------------------------
# Defensive — never crash
# ---------------------------------------------------------------------------


def test_unparseable_doesnt_crash() -> None:
    # Null bytes + huge input: our parser may crash; we must fail-open for
    # BUGS (lesser evil vs blocking every bash call on a policy typo).
    nasty = "\x00" * 100 + "x" * 100_000 + "\x00"
    assert check_bash_safety(nasty) is None


def test_empty_string_returns_none() -> None:
    assert check_bash_safety("") is None


def test_whitespace_only_returns_none() -> None:
    assert check_bash_safety("   \t  ") is None
