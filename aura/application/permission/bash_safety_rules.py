"""Tier A bash-safety rule predicates."""

from __future__ import annotations

import re
import shlex

from aura.application.permission.bash_safety_lex import (
    _first_token,
    _pipe_segments_quote_aware,
    _split_segments_quote_aware,
)
from aura.application.permission.bash_safety_paths import _is_system_path
from aura.application.permission.bash_safety_types import BashSafetyViolation

# Exhaustive zsh-builtin set that bypasses bash's file-access / fd-owner checks.
# Only the subset attackers have leveraged for sandbox escape — not "all dangerous zsh builtins".
ZSH_DANGEROUS_COMMANDS: frozenset[str] = frozenset({
    "zmodload", "emulate",
    "sysopen", "sysread", "syswrite", "sysseek",
    "zpty", "ztcp", "zsocket",
    "mapfile",
    "zf_rm", "zf_mv", "zf_ln", "zf_chmod",
    "zf_chown", "zf_mkdir", "zf_rmdir", "zf_chgrp",
})

# A shell name as the first token after ``|`` means the preceding segment's stdout
# is executed as script — canonical ``curl X | bash`` remote-exec pattern.
_SHELL_NAMES: frozenset[str] = frozenset({
    "sh", "bash", "zsh", "ksh", "dash", "csh", "tcsh", "fish",
})

# ``exec CMD`` replaces the shell with CMD; if CMD is destructive the shell cannot recover.
_DESTRUCTIVE_COMMANDS: frozenset[str] = frozenset({
    "rm", "chmod", "chown", "dd", "mkfs", "shred", "wipe",
})


_SEGMENT_SPLIT = re.compile(r"(?:\|\||&&|[;|\n])")


def _check_cr_outside_quotes(command: str) -> BashSafetyViolation | None:
    """Naive ``"`` toggle; raw ``\\r`` outside is reject. The CR attack has
    no quoting around its payload, so a one-state machine suffices."""
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


# Single-quote regions are stripped before substitution scan — bash disables expansion in '...'.
_SINGLE_QUOTED = re.compile(r"'[^']*'")
_DYNAMIC_EXEC = re.compile(
    r"""
    \$\(           |
    `              |
    (?<![\w-])(?:bash|sh|zsh)\s+-c(?:\s|$) |
    (?<![\w-])eval(?:\s|$)
    """,
    re.VERBOSE,
)


def _check_command_substitution(command: str) -> BashSafetyViolation | None:
    """Reject ``$(...)`` / backticks / ``(ba|z|)sh -c`` / ``eval`` — each is a
    runtime-expansion vector where the outer command is innocent and the inner
    payload only materializes when bash expands it."""
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


def _check_zsh_dangerous(command: str) -> BashSafetyViolation | None:
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


_QUOTED_REGION = re.compile(r'"[^"]*"|\'[^\']*\'')
_SEPARATOR_OUTSIDE_QUOTES = re.compile(r"(?:\|\||&&|;)")


def _check_malformed_with_separator(command: str) -> BashSafetyViolation | None:
    """Unparseable tokens adjacent to a separator outside quotes — a
    re-entry vector after a failed tokenization."""
    try:
        shlex.split(command, posix=True)
    except ValueError:
        stripped = _QUOTED_REGION.sub("", command)
        if _SEPARATOR_OUTSIDE_QUOTES.search(stripped):
            return BashSafetyViolation(
                reason="malformed_with_separator",
                detail="unparseable shell tokens adjacent to a command separator",
            )
    return None


def _check_cd_git_compound(command: str) -> BashSafetyViolation | None:
    """Both ``cd`` and ``git`` as free tokens — a malicious .git/config fires
    on any subsequent git invocation in the new dir, so order doesn't matter."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    if "cd" in tokens and "git" in tokens:
        return BashSafetyViolation(
            reason="cd_git_compound",
            detail="cd + git in the same command enables .git/config RCE",
        )
    return None


def _check_pipe_to_shell(command: str) -> BashSafetyViolation | None:
    """Shell interpreter as first token of any segment after ``|``.
    Misses aliased shells (beyond static reach without running bash)."""
    segments = _pipe_segments_quote_aware(command)
    if len(segments) < 2:
        return None
    for seg in segments[1:]:
        tok = _first_token(seg)
        if tok is None:
            continue
        basename = tok.rsplit("/", 1)[-1]
        if basename in _SHELL_NAMES:
            return BashSafetyViolation(
                reason="pipe_to_shell",
                detail=f"piping into shell interpreter '{basename}' executes arbitrary input",
            )
    return None


def _check_exec_destructive(command: str) -> BashSafetyViolation | None:
    """``exec CMD`` replaces the shell with CMD; destructive CMDs cannot be recovered from."""
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


def _check_destructive_removal(command: str) -> BashSafetyViolation | None:
    """``rm`` + any of ``-r``/``-R``/``-f`` (incl. combined ``-rf``) +
    positional under a system prefix (or ``/``). ``rm -rf /tmp/foo`` is
    allowed — the system-path gate is what keeps this Tier A rather than
    an overbroad kill-switch."""
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


# Boundary-anchored ``chmod [-R] 777`` / ``0777`` / ``a+rwx`` so ``chmod7777`` doesn't match.
_CHMOD_WORLD_WRITABLE = re.compile(
    r"""
    \bchmod\b
    (?:\s+-[A-Za-z]+)?
    \s+
    (?:
        0?777
        |
        [ugoa]*[+=][rwx]*w[rwx]*
    )
    \b
    """,
    re.VERBOSE,
)


def _check_world_writable_chmod(command: str) -> BashSafetyViolation | None:
    """Reject ``chmod 777`` / ``a+w`` / ``o+w`` on ANY path — legit use
    cases are vanishingly rare vs the attack surface."""
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
        reconstructed = " ".join(tokens)
        if _CHMOD_WORLD_WRITABLE.search(reconstructed):
            return BashSafetyViolation(
                reason="world_writable_chmod",
                detail="chmod to world-writable permissions (777 / a+w / o+w)",
            )
    return None


def _check_root_chown(command: str) -> BashSafetyViolation | None:
    """``chown root`` / ``chown 0:0`` — no legit agent workflow reparents to root."""
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
        for tok in tokens[1:]:
            if tok.startswith("-"):
                continue
            spec = tok.split(":")[0]
            if spec in ("root", "0"):
                return BashSafetyViolation(
                    reason="root_chown",
                    detail=f"chown to root ('{tok}') escalates ownership",
                )
            break
    return None


def _check_sed_inplace_system_path(command: str) -> BashSafetyViolation | None:
    """``sed -i`` (or combined ``-iE``/``-Ei``, or ``--in-place``) targeting a
    system-path positional. BSD-sed's ``-i ''`` backup-suffix arg is not
    modelled; a system-path positional remains system-path wherever it lands."""
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


# Leading ``\d`` lookbehind excludes ``2>&1`` (fd-dup, not file redirect).
_REDIRECT_OUTSIDE_QUOTES = re.compile(
    r"""
    (?<!\d)
    (?:&>>|&>|>>|>)
    \s*
    (
        (?:'[^']*')
        |
        (?:"[^"]*")
        |
        (?:[^\s;&|<>()]+)
    )
    """,
    re.VERBOSE,
)


def _check_redirect_to_system_path(command: str) -> BashSafetyViolation | None:
    """Redirect targets outside single-quoted regions. Double-quoted targets
    are honored — bash still expands them. ``>&filename`` (deprecated) not
    modelled."""
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


# Decoder-into-shell pipelines — stdlib can't distinguish ``base64 -d`` from
# ``base64 --help`` without running it; false-positive rate accepted because
# base64-into-shell has no legitimate agent use case.
_OBFUSCATED_DECODERS = (
    r"base64\s+(?:-d|-D|--decode)",
    r"xxd\s+-r",
    r"openssl\s+(?:base64|enc)\s+-d",
)
_OBFUSCATED_EXEC_RE = re.compile(
    r"(?:" + "|".join(_OBFUSCATED_DECODERS) + r")"
    r"[^|]*\|\s*(?:/[A-Za-z0-9_/.-]+/)?"
    r"(?:" + "|".join(re.escape(s) for s in _SHELL_NAMES) + r")\b",
)


def _check_obfuscated_execution(command: str) -> BashSafetyViolation | None:
    """Decoder-pipe-shell as a dedicated reason (not just pipe_to_shell)
    so the model's error message guides toward a transparent alternative."""
    residual = _SINGLE_QUOTED.sub("", command)
    if _OBFUSCATED_EXEC_RE.search(residual):
        return BashSafetyViolation(
            reason="obfuscated_execution",
            detail="decoding encoded input into a shell (base64 -d | sh, etc.)",
        )
    return None
