"""File-write / file-edit specialized permission widget.

Matches claude-code's ``FileWritePermissionRequest`` /
``FileEditPermissionRequest`` — when the pending tool is ``write_file``
or ``edit_file`` we render a dedicated widget that:

1. Prints the target path (dim) so the operator sees exactly which
   file is being touched.
2. For ``edit_file`` — builds a compact unified diff from ``old_str``
   → ``new_str`` via ``difflib.unified_diff`` and colors ``-`` lines
   ansired / ``+`` lines ansigreen so the change is readable at a
   glance.
3. For ``write_file`` — shows the total byte count of the new content
   plus the first 5 lines as a preview (dim-prefixed with ``│``).
4. Falls back to the shared 4-option pt.Application driver from
   :mod:`aura.cli.permission_generic` for the accept / always / no
   flow + Tab-to-amend + Esc-cancel + Ctrl+E explanation.

Previews are visual-only — the tool still receives the full args
downstream (we never truncate the content actually passed to the
tool).
"""

from __future__ import annotations

import difflib
from typing import Any, Literal

from langchain_core.tools import BaseTool

from aura.cli.permission_generic import (
    _build_explanation,
    _run_widget,
    _tool_title,
    _tool_verb,
)

# Cap the diff body so a 10k-line refactor doesn't explode the widget.
# Truncation is visual-only. When truncation fires we append a dim
# "… N more lines" footer so the operator knows the view is partial.
_DIFF_MAX_LINES = 40

# Write-preview cap: how many lines of the new content we show before
# saying "… rest elided". Matches claude-code's shortPreview behaviour.
_WRITE_PREVIEW_LINES = 5


def build_diff_fragments(
    old_str: str, new_str: str, path: str,
) -> list[tuple[str, str]]:
    """Render a compact unified diff of ``old_str`` → ``new_str`` as
    pt fragments.

    ``-`` lines become ansired, ``+`` lines ansigreen, ``@@`` hunk
    headers ansicyan, file headers dim. Empty diffs (identical strings)
    return a single dim "(no change)" line instead of an empty block
    so the UI doesn't look broken.
    """
    old_lines = old_str.splitlines(keepends=False)
    new_lines = new_str.splitlines(keepends=False)
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
            n=2,
        )
    )
    if not diff:
        return [("class:dim", "    (no change)\n")]

    frags: list[tuple[str, str]] = []
    for shown, line in enumerate(diff):
        if shown >= _DIFF_MAX_LINES:
            remaining = len(diff) - shown
            frags.append(("class:dim", f"    … {remaining} more diff lines\n"))
            break
        if line.startswith(("---", "+++")):
            frags.append(("class:dim", f"    {line}\n"))
        elif line.startswith("@@"):
            frags.append(("ansicyan", f"    {line}\n"))
        elif line.startswith("-"):
            frags.append(("ansired", f"    {line}\n"))
        elif line.startswith("+"):
            frags.append(("ansigreen", f"    {line}\n"))
        else:
            frags.append(("", f"    {line}\n"))
    return frags


def build_write_preview(content: str) -> list[tuple[str, str]]:
    """Render a ``write_file`` preview: byte-count summary + first N
    lines of content.

    The summary line is dim; preview lines are prefixed with ``│`` in
    dim so they read as a block quote of the new file body.
    """
    encoded = content.encode("utf-8", errors="replace")
    byte_count = len(encoded)
    lines = content.splitlines()

    frags: list[tuple[str, str]] = [
        ("class:dim", f"    {byte_count} bytes, {len(lines)} lines\n"),
    ]
    preview_lines = lines[:_WRITE_PREVIEW_LINES]
    for line in preview_lines:
        # Keep each preview line bounded so a pathological single-line
        # giant-string file doesn't wrap 50 times.
        display = line if len(line) <= 120 else line[:119] + "…"
        frags.append(("class:dim", f"    │ {display}\n"))
    remaining = len(lines) - len(preview_lines)
    if remaining > 0:
        frags.append(("class:dim", f"    │ … {remaining} more lines\n"))
    return frags


async def run_write_permission(
    *,
    tool: BaseTool,
    args: dict[str, Any],
    tag: Literal["destructive", "read-only", "safe"],
    option_two_label: str,
    default_choice: int,
    timeout: float | None = None,
) -> tuple[int | None, str]:
    """Render the write/edit-specialized permission widget.

    Dispatches on ``tool.name`` to pick the diff-preview path
    (``edit_file``) or the size-preview path (``write_file``). Any
    other tool here is a caller bug — the dispatcher in
    :mod:`aura.cli.permission` should have routed it to the generic
    widget.

    Returns ``(choice, feedback)`` — same shape as the other widgets.
    """
    title = _tool_title(tool)
    verb = _tool_verb(tool)
    path = str(args.get("path", "") or "")

    header: list[tuple[str, str]] = [
        ("bold", f"  {title}\n"),
        ("", "\n"),
    ]
    if path:
        header.append(("class:dim", f"    {path}\n"))
        header.append(("", "\n"))

    if tool.name == "edit_file":
        old_str = str(args.get("old_str", "") or "")
        new_str = str(args.get("new_str", "") or "")
        header.extend(build_diff_fragments(old_str, new_str, path or "file"))
        header.append(("", "\n"))
    elif tool.name == "write_file":
        content = str(args.get("content", "") or "")
        header.extend(build_write_preview(content))
        header.append(("", "\n"))
    # Any other tool name here is unexpected (the dispatcher should
    # have filtered it), but we degrade gracefully by just showing the
    # path + verb instead of crashing.

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
