"""Bash-command safety — Tier A hard floors.

Four classes of shell commands that the agent MUST refuse regardless of
permission mode, user rule grants, or ``--bypass-permissions``. Mirrors
the Tier A set from claude-code's ``bashSecurity.ts`` — each entry below
is a known shell parser-differential / privilege-escalation vector that
CANNOT be made safe by "the user said yes".

Layer boundary: this module is a PURE policy — no I/O, no journaling.
The hook layer (``aura.core.hooks.bash_safety``) is responsible for
journaling and returning a ``ToolResult``. Keeping the policy pure is
what lets it share the architectural shape of
``aura.core.permissions.safety`` (paths) and stay test-friendly.

The four checks (evaluated in this order — cheapest / earliest-blocking
first):

1. **cr_outside_double_quote** — a raw ``\\r`` anywhere outside a
   ``"..."`` region. The ``shell-quote`` JS tokenizer (used by claude-
   code's allowlist matcher) treats ``\\r`` as a token boundary, while
   bash treats it as command text — the differential is command
   injection past an allowlist. Python's shlex has a closely related
   differential; we reject the input outright rather than reason about
   which tokenizer agrees.

2. **zsh_dangerous_command** — any segment (split on ``;`` / ``&&`` /
   ``||`` / ``|`` / newline) whose first token is one of a small set of
   zsh builtins that bypass the usual file-access checks. Attack:
   ``zmodload zsh/system && syswrite ...`` uses ``syswrite`` to scribble
   on an fd bash would never let a normal ``echo >`` reach.

3. **malformed_with_separator** — ``shlex.split`` raises (unbalanced
   quote, invalid escape) AND the raw command contains a command
   separator outside quoted regions. Attack vector:
   ``echo {"hi":"hi;evil"}`` — the unbalanced-looking quote plus the
   ``;`` lets an eval-like mechanism re-enter with ``evil`` as a fresh
   command. A malformed command WITHOUT a separator is just a bad
   command; the tool's own error path handles it.

4. **cd_git_compound** — any command that parses to a token list
   containing BOTH ``cd`` AND ``git`` as free-standing tokens. Attack
   vector: cd into a malicious repo whose ``.git/config`` has
   ``core.fsmonitor = <rce payload>``; the next ``git status`` fires
   it. Over-broad in principle but extremely low false-positive in
   practice (benign prose with ``cd`` and ``git`` lives in quoted
   strings, which shlex collapses into a single arg).

Failure mode inversion
----------------------

Normally "fail closed" is the safe default for a safety check. This
module inverts that for ONE specific case: if our OWN parsing code
crashes on a weird input (huge buffer, embedded NULs, unexpected
encoding), we return ``None`` rather than raising. Rationale: a typo
in THIS file would otherwise block every bash call the agent ever
makes. The legitimate attack surface (the four rules above) uses
parseable inputs; a parser crash means our code is buggy, not that
the input is an attack. The lesser evil is fail-open for bugs.

``shlex`` / raw-string anomalies inside the rules themselves still
fail closed — e.g. ``shlex.split`` raising IS a rule-3 signal, not a
bug-fail-open.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Literal

# 18 entries — the exhaustive zsh-builtin list that bypasses bash's
# usual file-access / fd-owner checks. Not a superset of "dangerous zsh
# builtins" in general; only the subset that attackers have actually
# leveraged for sandbox escape in shell-allowlist tools.
ZSH_DANGEROUS_COMMANDS: frozenset[str] = frozenset({
    "zmodload", "emulate",
    "sysopen", "sysread", "syswrite", "sysseek",
    "zpty", "ztcp", "zsocket",
    "mapfile",
    "zf_rm", "zf_mv", "zf_ln", "zf_chmod",
    "zf_chown", "zf_mkdir", "zf_rmdir", "zf_chgrp",
})

Reason = Literal[
    "zsh_dangerous_command",
    "cr_outside_double_quote",
    "malformed_with_separator",
    "cd_git_compound",
]


@dataclass(frozen=True)
class BashSafetyViolation:
    """A single-reason reject. ``detail`` is cited verbatim in the
    ToolResult.error surfaced to the model, so keep it one line and
    actionable."""

    reason: Reason
    detail: str


# Segment-split regex — same separators shlex / bash treat as command
# boundaries. Compiled once; used for the per-segment zsh-builtin scan.
_SEGMENT_SPLIT = re.compile(r"(?:\|\||&&|[;|\n])")


def check_bash_safety(command: str) -> BashSafetyViolation | None:
    """Return the first Tier A violation, or None if the command is safe.

    Order (documented contract — tests depend on it):
      1. cr_outside_double_quote  — cheap, no tokenization needed
      2. zsh_dangerous_command    — per-segment first-token scan
      3. malformed_with_separator — shlex raises + separator outside quotes
      4. cd_git_compound          — both tokens present in the flat tokenization
    """
    if not isinstance(command, str) or not command:
        return None

    try:
        # 1. CR outside double quotes.
        violation = _check_cr_outside_quotes(command)
        if violation is not None:
            return violation

        # 2. ZSH dangerous command (per segment).
        violation = _check_zsh_dangerous(command)
        if violation is not None:
            return violation

        # 3. Malformed + separator.
        violation = _check_malformed_with_separator(command)
        if violation is not None:
            return violation

        # 4. cd + git compound.
        violation = _check_cd_git_compound(command)
        if violation is not None:
            return violation

        return None
    except Exception:  # noqa: BLE001 — see module docstring "failure mode inversion"
        # A bug in our own parser must not block every bash call the
        # agent ever makes. The four real attack classes all use
        # parseable inputs; a parser crash means we have a bug.
        return None


# ---------------------------------------------------------------------------
# Rule 1 — carriage return outside double quotes
# ---------------------------------------------------------------------------


def _check_cr_outside_quotes(command: str) -> BashSafetyViolation | None:
    """Naive toggle: ``"`` flips double-quote state; raw ``\\r`` outside
    is a reject. Doesn't model backslash-escapes or single quotes —
    bash's rules are complex, but the CR-specific differential (TZ=UTC
    \\r echo evil) has no quoting in the attack, so a naive state
    machine is sufficient."""
    in_double_quote = False
    for ch in command:
        if ch == '"':
            in_double_quote = not in_double_quote
        elif ch == "\r" and not in_double_quote:
            return BashSafetyViolation(
                reason="cr_outside_double_quote",
                detail="carriage return outside double-quoted region",
            )
    return None


# ---------------------------------------------------------------------------
# Rule 2 — zsh dangerous builtin as a segment's first token
# ---------------------------------------------------------------------------


def _check_zsh_dangerous(command: str) -> BashSafetyViolation | None:
    """Split on command separators; for each segment, try to tokenize;
    if the first token is in the dangerous set, reject. Silent-skip
    segments whose tokenization fails — rule 3 covers that case."""
    for segment in _SEGMENT_SPLIT.split(command):
        stripped = segment.strip()
        if not stripped:
            continue
        try:
            tokens = shlex.split(stripped, posix=True)
        except ValueError:
            continue
        if tokens and tokens[0] in ZSH_DANGEROUS_COMMANDS:
            return BashSafetyViolation(
                reason="zsh_dangerous_command",
                detail=f"zsh builtin '{tokens[0]}' bypasses file-access checks",
            )
    return None


# ---------------------------------------------------------------------------
# Rule 3 — malformed tokenization AND separator present outside quotes
# ---------------------------------------------------------------------------


_QUOTED_REGION = re.compile(r'"[^"]*"|\'[^\']*\'')
_SEPARATOR_OUTSIDE_QUOTES = re.compile(r"(?:\|\||&&|;)")


def _check_malformed_with_separator(command: str) -> BashSafetyViolation | None:
    try:
        shlex.split(command, posix=True)
    except ValueError:
        # Strip matched quoted regions, THEN look for separators in what
        # remains. If a separator survives, a malformed tokenization +
        # re-entry vector is present.
        stripped = _QUOTED_REGION.sub("", command)
        if _SEPARATOR_OUTSIDE_QUOTES.search(stripped):
            return BashSafetyViolation(
                reason="malformed_with_separator",
                detail="unparseable shell tokens adjacent to a command separator",
            )
    return None


# ---------------------------------------------------------------------------
# Rule 4 — cd + git compound
# ---------------------------------------------------------------------------


def _check_cd_git_compound(command: str) -> BashSafetyViolation | None:
    """Flat tokenization: if BOTH ``cd`` and ``git`` appear as
    free-standing tokens anywhere in the command, reject. Order and
    adjacency don't matter — a malicious ``.git/config`` fires on any
    subsequent git invocation within the repo."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        # Rule 3 already handled; here the compound check simply
        # can't run, so no rule-4 signal.
        return None
    if "cd" in tokens and "git" in tokens:
        return BashSafetyViolation(
            reason="cd_git_compound",
            detail="cd + git in the same command enables .git/config RCE",
        )
    return None
