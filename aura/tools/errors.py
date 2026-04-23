"""Shared tool-error hint table.

Lives here (rather than in ``aura.cli.render``) because BOTH the renderer
(visual panel for the user) AND the loop's ToolMessage builder (text the
MODEL sees) need to consult it. Keeping the table in ``cli`` would mean
the loop depends on ``aura.cli`` — a layering inversion.

The table is substring-keyed: first hit wins, so more-specific phrases
go BEFORE shorter prefixes. Matching is case-insensitive on the lowercased
error text.
"""

from __future__ import annotations

# Order matters: first substring hit wins. Put the more-specific messages
# BEFORE shorter ones that would otherwise prefix-match. Mirrors the
# behaviour contract documented on the old ``render._hint_for_error``.
_ERROR_HINTS: list[tuple[str, str]] = [
    ("ripgrep",
     "install ripgrep — brew install ripgrep  (or the platform equivalent)"),
    ("old_str not found",
     "the string may have changed on disk; re-read the file to see "
     "current content before the next edit_file"),
    ("not found",
     "check the path; the file may have been moved or deleted"),
    ("has not been read yet",
     "call read_file({path}) first — edit_file / write_file require a "
     "prior read to catch staleness"),
    ("file has changed since last read",
     "re-read the file; its content drifted between read and edit"),
    ("file was only partially read",
     "read_file the target again with offset=0 limit=None before edit"),
    ("unknown task_id",
     "only IDs returned by task_create are valid; /tasks lists them"),
    ("cannot create parent dir",
     "check write permissions on the parent directory"),
    ("bash safety blocked",
     "the command pattern is a hard floor; split into safer steps or "
     "use non-matching syntax"),
    ("plan mode",
     "plan mode is active — switch to a different permission mode to "
     "actually execute tools"),
    ("timeout",
     "raise the timeout kwarg for long-running work, or break the "
     "command into chunks"),
]


def hint_for_error(tool_name: str, error: str) -> str | None:
    """Return an actionable hint for ``error``, or ``None``.

    First substring-match wins. ``tool_name`` is accepted for future
    per-tool routing; currently unused. ``None`` means no hint applies —
    callers should omit the hint line entirely rather than pad with
    generic filler.
    """
    _ = tool_name
    haystack = error.lower()
    for needle, hint in _ERROR_HINTS:
        if needle.lower() in haystack:
            return hint
    return None
