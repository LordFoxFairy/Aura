"""Bash-specialized permission widget.

Matches claude-code's ``BashPermissionRequest`` — when the pending tool
is ``bash`` or ``bash_background`` we render a dedicated widget that:

1. Highlights the command with a bash-aware syntax lexer (rendered via
   ``rich`` to ANSI, then handed to prompt_toolkit as an ANSI string).
2. Detects a small closed set of obviously-dangerous patterns (``rm
   -rf``, ``sudo``, ``curl | sh``, ``chmod 777``, redirects to system
   paths, raw block-device writes, destructive-background-operator
   combos) and — when present — renders a prominent ``⚠ DANGEROUS``
   banner in red ABOVE the command block.
3. Falls back to the shared 4-option pt.Application driver from
   :mod:`aura.cli.permission_generic` for the accept / always / no
   flow + Tab-to-amend + Esc-cancel + Ctrl+E explanation.

Detection is purely informational — it does NOT auto-deny; that's the
safety hook's job. The banner exists so the operator's eye lands on
the risky part instantly instead of scanning a 3-line shell pipeline.

Spec alignment: claude-code's destructiveCommandWarning.ts is richer
(20+ git / db / k8s patterns). We ship the Tier-A patterns the task
brief demands; additions land as they're needed without touching the
widget's structure.
"""

from __future__ import annotations

import io
import re
from typing import Any, Literal

from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from rich.console import Console
from rich.syntax import Syntax

from aura.cli.permission_generic import (
    _build_explanation,
    _run_widget,
    _tool_title,
    _tool_verb,
)

# Dangerous-pattern table. (pattern, human-readable warning). Order is
# insignificant — the widget only needs ANY match to flip the banner.
# Kept as a module-level constant so tests (and the renderer) can
# re-use the same regexes instead of duplicating them.
#
# Patterns are case-sensitive where that matters (``sudo`` is a Unix
# binary name; uppercase SUDO is not a real command). ``chmod 777``
# uses ``\b`` anchors so a literal ``chmod7777`` argument doesn't
# false-match.
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # rm -rf / rm -r / rm -f (and combined short flags like -rfv)
    (re.compile(r"(?:^|[;&|\s])rm\s+(?:-[a-zA-Z]*[rRf][a-zA-Z]*)"),
     "rm -r / -f / -rf: recursive / force removal"),
    # sudo — any elevation
    (re.compile(r"(?:^|[;&|\s])sudo\b"),
     "sudo: privilege escalation"),
    # curl | sh  /  wget | sh  (pipe-to-shell install). Accept any
    # variant of the shell interpreter on the right of the pipe.
    (re.compile(r"\b(?:curl|wget)\b[^|]*\|\s*(?:sh|bash|zsh|ksh|sudo)\b"),
     "pipe-to-shell: remote script executed blind"),
    # chmod 777 / chmod -R 777 / chmod 0777
    (re.compile(r"\bchmod\s+(?:-R\s+)?0?777\b"),
     "chmod 777: world-writable permissions"),
    # Redirects to system paths: > /etc/..., >> /usr/..., > /bin/...
    (re.compile(r">{1,2}\s*/(?:etc|usr|bin|sbin|boot|sys|proc)(?:/|\b)"),
     "writes to system path"),
    # Writes to raw block devices: > /dev/sda, > /dev/nvme0n1, dd of=/dev/sd*
    (re.compile(r"(?:>{1,2}\s*|\bof=)\s*/dev/(?:sd[a-z]|nvme|hd[a-z]|vd[a-z])"),
     "writes to raw block device"),
    # Destructive background combos: rm ... & / rm ... ; ... & (rm followed
    # by a background operator). Anchors on rm so a bare ``&`` doesn't fire.
    (re.compile(r"\brm\b[^;&\n]*&(?!&)"),
     "rm combined with background operator"),
]


def detect_dangerous(command: str) -> list[str]:
    """Return the list of dangerous-pattern warnings that match ``command``.

    Empty list means "no known dangerous pattern" (safe enough to show
    without a banner). The returned strings are human-readable — the
    caller joins them into the banner body.

    Kept module-level (not tucked inside the widget) so tests can pin
    detection behavior without driving the full pt.Application.
    """
    hits: list[str] = []
    for pattern, warning in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            hits.append(warning)
    return hits


def _render_command_ansi(command: str) -> str:
    """Render ``command`` through rich's bash syntax lexer to an ANSI
    string.

    We render to a detached ``rich.Console`` (``record=True`` +
    ``export_text`` with ``styles=True``) so the output carries ANSI
    escape codes that prompt_toolkit's ``ANSI`` wrapper can interpret.
    This keeps rich as the single syntax-highlighting engine (no
    pygments-via-pt dependency) while still letting pt lay out the
    fragment.

    Long commands (>400 chars) are truncated — the full command still
    reaches the tool downstream, this is purely visual for the dialog
    height. A literal ``…`` is appended when truncation fired.
    """
    display = command if len(command) <= 400 else command[:399] + "…"
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        color_system="truecolor",
        width=100,
        record=False,
    )
    syntax = Syntax(display, "bash", theme="ansi_dark", background_color="default")
    console.print(syntax, end="")
    return buf.getvalue()


def _highlight_dangerous_tokens(command: str) -> list[tuple[str, str]]:
    """Return pt fragments for the command line with dangerous tokens
    coloured ansired.

    Used when the syntax-highlight path isn't available (fallback) or
    as an ADDITIONAL layer beneath the syntax highlight — callers chose
    which. We render a simple split-by-regex pass: every substring
    matched by any ``_DANGEROUS_PATTERNS`` regex becomes
    ``ansired bold``; everything else stays unstyled.

    Token overlap is handled by picking the leftmost next match on each
    iteration — linear-time, no regex alternation compilation surprise.
    """
    frags: list[tuple[str, str]] = []
    pos = 0
    while pos < len(command):
        # Find the earliest match starting at or after ``pos``.
        best_start = len(command)
        best_end = len(command)
        for pattern, _warning in _DANGEROUS_PATTERNS:
            m = pattern.search(command, pos)
            if m is not None and m.start() < best_start:
                best_start = m.start()
                best_end = m.end()
        if best_start >= len(command):
            # No more matches; emit the tail as plain.
            frags.append(("", command[pos:]))
            break
        if best_start > pos:
            frags.append(("", command[pos:best_start]))
        frags.append(("ansired bold", command[best_start:best_end]))
        pos = best_end
    return frags


async def run_bash_permission(
    *,
    tool: Any,  # BaseTool, but we avoid the import to keep this module light
    command: str,
    args_preview: str,  # noqa: ARG001 — kept for signature symmetry with _generic
    args: dict[str, Any],
    tag: Literal["destructive", "read-only", "safe"],
    option_two_label: str,
    default_choice: int,
    timeout: float | None = None,
) -> tuple[int | None, str]:
    """Render the bash-specialized permission widget.

    Returns ``(choice, feedback)`` — same shape as
    :func:`aura.cli.permission_generic.run_generic_permission` so the
    dispatch router is trivial.
    """
    title = _tool_title(tool)
    verb = _tool_verb(tool)

    warnings = detect_dangerous(command)

    header: list[tuple[str, str]] = [
        ("bold", f"  {title}\n"),
        ("", "\n"),
    ]

    if warnings:
        # Prominent red banner ABOVE the command. Lead with a glyph +
        # all-caps marker so the operator's eye lands here first.
        header.append(("ansired bold", "  ⚠ DANGEROUS\n"))
        for w in warnings:
            header.append(("ansired", f"    • {w}\n"))
        header.append(("", "\n"))

    # Command preview — syntax-highlighted via rich, with a secondary
    # per-token red overlay for the dangerous substrings. The overlay
    # renders first (plain fragments) so tests can assert on token
    # styling; the syntax-coloured ANSI block follows so the user sees
    # the prettier view.
    #
    # We render the token-highlight path as the primary view (one line,
    # no newlines in the middle — keeps the dialog compact). The ANSI
    # syntax block from rich is only used when the command has no
    # dangerous tokens to highlight (so colours don't fight each
    # other).
    if warnings:
        # Token-highlight path: custom fragments so dangerous regions
        # are ansired bold. Indent matches the rest of the widget.
        header.append(("", "    "))
        header.extend(_highlight_dangerous_tokens(command))
        header.append(("", "\n"))
    else:
        ansi = _render_command_ansi(command).rstrip("\n")
        # Indent each line of the ANSI block so it lines up with the
        # rest of the widget. ``to_formatted_text(ANSI(...))`` converts
        # the ANSI-escape string into pt's fragment shape so it sits
        # alongside our other tuples in the same fragment stream.
        for line in ansi.splitlines() or [""]:
            indented = "    " + line + "\n"
            for frag in to_formatted_text(ANSI(indented)):
                header.append((frag[0], frag[1]))

    if verb:
        header.append(("class:dim", f"  {verb}\n"))
    header.append(("", "\n"))

    explanation_frags = _build_explanation(tool, args, tag)

    return await _run_widget(
        header_frags=header,
        option_two_label=option_two_label,
        default_choice=default_choice,
        explanation_frags=explanation_frags,
        timeout=timeout,
    )
