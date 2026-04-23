"""Generic permission widget — the default pt.Application dialog.

Extracted verbatim from ``aura.cli.permission`` during the v0.7.6
per-tool-specialization refactor. This is the fallback widget used for
any tool that doesn't have a specialized permission UI (anything except
``bash`` / ``bash_background`` / ``write_file`` / ``edit_file``).

Layout (mirrors claude-code's PermissionDialog — screenshot 2026-04-23,
see module docstring of the old ``permission.py``):

    ──────────────────────────  (horizontal rule, separates from chat)

    <Tool-Title> command

      <preview>
    <verb>

    Do you want to proceed?
    ❯ 1. Yes
      2. Yes, and don't ask again for: ...
      3. No

    Esc to cancel · Enter to confirm · 1/2/3 to jump · Tab to amend

One job: present the choice, capture the answer. The widget does NOT
decide (that's the hook), does NOT persist (that's the store), and does
NOT emit domain events (caller handles journal).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from langchain_core.tools import BaseTool
from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from aura.cli._coordination import pause_spinner_if_active, prompt_mutex

# Visual cap on the args preview so a huge ``bash`` command (or similar)
# doesn't overflow the terminal. Visual only — the tool still receives
# the full args.
_PREVIEW_MAX_CHARS = 200

# Width of the horizontal separator rendered above the widget. Fixed
# at 78 — wide enough to read as a divider on 80-col terminals, narrow
# enough not to wrap on modern defaults. Drawn with U+2500 so it
# renders as an unambiguous rule in any monospace font.
_SEPARATOR = "─" * 78

# Per-tool short description shown under the command preview (mirrors
# claude-code's "Run shell command" / "Write to file" line — a
# plain-English verb so the user knows what the tool does without
# knowing its registry name). Unknown tools fall back to the tool's
# own ``description`` field, truncated to one line.
_TOOL_VERB: dict[str, str] = {
    "bash": "Run shell command",
    "bash_background": "Run shell command in background",
    "read_file": "Read file contents",
    "write_file": "Write to file",
    "edit_file": "Edit file in place",
    "glob": "Find files by pattern",
    "grep": "Search file contents",
    "web_fetch": "Fetch web page",
    "web_search": "Search the web",
    "todo_write": "Update the todo list",
    "task_create": "Spawn a background subagent",
    "task_output": "Read subagent output",
    "ask_user_question": "Ask the user a question",
}


def _tool_title(tool: BaseTool) -> str:
    """Human-readable section heading for the widget.

    ``bash`` → "Bash command", ``read_file`` → "Read file". Title case
    on the first segment; rest left alone. Matches claude-code's
    convention of a short proper-noun-ish title.
    """
    name = tool.name.replace("_", " ")
    if " " in name:
        head, *rest = name.split(" ")
        return " ".join([head.capitalize(), *rest])
    # Single-word tool (e.g. "bash", "grep"). Append " command" so it
    # reads as a sentence fragment, not a bare identifier.
    return f"{name.capitalize()} command"


def _tool_verb(tool: BaseTool) -> str:
    """Short description line under the preview.

    Preferred source order:
      1. ``_TOOL_VERB`` explicit map (curated phrasing)
      2. First line of ``tool.description`` truncated to 80 chars
      3. Empty string (subtext piece elided)
    """
    if tool.name in _TOOL_VERB:
        return _TOOL_VERB[tool.name]
    desc = (tool.description or "").strip().splitlines()
    if desc:
        first = desc[0].strip()
        if len(first) > 80:
            first = first[:79] + "…"
        return first
    return ""


# Risk one-liners shown in the Ctrl+E explanation block. Tag → human
# phrasing. Kept here (not in _TOOL_VERB or the tag resolver) so the
# wording stays close to the rendering site.
_RISK_LINES: dict[str, str] = {
    "destructive": "⚠ This tool can modify or delete data.",
    "read-only": "● Read-only — no side effects.",
    "safe": "● Low risk — creates new data without overwrite.",
}


def _build_explanation(
    tool: BaseTool,
    args: dict[str, Any],
    tag: Literal["destructive", "read-only", "safe"],
) -> list[tuple[str, str]]:
    """Render the Ctrl+E explanation block for this tool invocation.

    Static, local-only — no LLM, no agent callback. Pulls from:

      * ``tool.description`` (first paragraph, up to 4 lines) for *what*
        this tool does in general,
      * ``args`` dict for *how* it's being called this time (2-column
        ``key: value`` block, values truncated to 80 chars so a giant
        bash command doesn't balloon the widget),
      * ``tag`` for the risk one-liner,
      * ``_TOOL_VERB`` for the "what happens if you approve" line.

    Framed with ``┌ Explanation`` / ``└`` so it reads as a collapsible
    panel. All dim except the header to visually demote it below the
    question + options.
    """
    frags: list[tuple[str, str]] = [
        ("class:dim bold", "  ┌ Explanation\n"),
    ]

    # "What this tool does" — first paragraph of description, up to 4
    # lines. Paragraph-break on the first blank line so long docstrings
    # don't bleed their full API reference into the widget.
    desc = (tool.description or "").strip()
    para_lines: list[str] = []
    for line in desc.splitlines():
        if not line.strip():
            break
        para_lines.append(line.strip())
        if len(para_lines) >= 4:
            break
    if para_lines:
        frags.append(("class:dim bold", "  │ What this tool does:\n"))
        for line in para_lines:
            frags.append(("class:dim", f"  │     {line}\n"))

    # "Arguments" — 2-column key: value. Values truncated to 80 chars
    # (visual-only; the tool still receives the full args downstream).
    frags.append(("class:dim bold", "  │ Arguments:\n"))
    if args:
        for key, value in args.items():
            rendered = repr(value) if not isinstance(value, str) else value
            if len(rendered) > 80:
                rendered = rendered[:79] + "…"
            frags.append(("class:dim", f"  │     {key}: {rendered}\n"))
    else:
        frags.append(("class:dim", "  │     (none)\n"))

    # "Risk" — static one-liner per tag.
    frags.append(("class:dim bold", "  │ Risk:\n"))
    frags.append(("class:dim", f"  │     {_RISK_LINES[tag]}\n"))

    # "What happens if you approve" — per-tool verb or generic fallback.
    frags.append(("class:dim bold", "  │ What happens if you approve:\n"))
    verb = _TOOL_VERB.get(tool.name)
    if verb:
        happens = f"{verb} with the arguments above."
    else:
        happens = "The tool will be invoked with the arguments above."
    frags.append(("class:dim", f"  │     {happens}\n"))

    frags.append(("class:dim bold", "  └\n"))
    return frags


async def _run_widget(
    *,
    header_frags: list[tuple[str, str]],
    option_two_label: str,
    default_choice: int,
    explanation_frags: list[tuple[str, str]],
    timeout: float | None = None,
) -> tuple[int | None, str]:
    """Shared pt.Application driver used by all three specialized widgets.

    ``header_frags`` — the widget-specific preamble (title + preview +
    verb + any per-widget extras like bash's DANGEROUS banner or
    write's diff). Rendered above the question + options.

    ``option_two_label`` — already-composed "Yes, always" label. The
    caller owns rule derivation (stays in the dispatcher / caller).

    ``default_choice`` — which of 1/2/3 the cursor starts on.

    ``explanation_frags`` — pre-built Ctrl+E panel. Lazy-rendered only
    when the user hits Ctrl+E.

    Returns ``(choice, feedback)`` — ``choice`` is 1/2/3 on commit or
    ``None`` on cancel (Esc / Ctrl+C); ``feedback`` is the Tab-to-amend
    note or ``""``. Feedback is captured on ACCEPT (choice=1), ALWAYS
    (choice=2), AND DENY (choice=3) — matches claude-code's
    acceptFeedback + rejectFeedback symmetry.
    """
    # Pause any active thinking spinner BEFORE we install pt's render
    # loop — otherwise rich.Live and pt fight for the same region.
    await pause_spinner_if_active()

    options = (
        (1, "Yes"),
        (2, option_two_label),
        (3, "No"),
    )
    cursor: list[int] = [default_choice - 1]
    committed: list[int | None] = [None]
    # Tab-to-amend state. Single-slot lists so nested closures can
    # mutate without nonlocal ceremony.
    awaiting_feedback: list[bool] = [False]
    feedback_buf: list[str] = [""]
    feedback_returned: list[str] = [""]
    # Ctrl+E explanation-panel toggle. Starts collapsed; Ctrl+E flips.
    explain_visible: list[bool] = [False]

    def _widget_fragments() -> FormattedText:
        frags: list[tuple[str, str]] = [
            ("class:dim", _SEPARATOR + "\n"),
            ("", "\n"),
        ]
        frags.extend(header_frags)
        if explain_visible[0]:
            frags.extend(explanation_frags)
            frags.append(("", "\n"))
        frags.append(("bold", "  Do you want to proceed?\n"))
        for i, (n, label) in enumerate(options):
            is_sel = i == cursor[0]
            if is_sel:
                frags.append(("ansicyan bold", f"  ❯ {n}. {label}\n"))
            else:
                frags.append(("class:dim", f"    {n}. {label}\n"))
        frags.append(("", "\n"))
        if awaiting_feedback[0]:
            frags.append(("ansicyan", f"  › {feedback_buf[0]}▌\n"))
            frags.append((
                "class:dim",
                "  Add feedback (Enter to submit, Esc to cancel)",
            ))
        else:
            explain_hint = (
                "ctrl+e to hide" if explain_visible[0] else "ctrl+e to explain"
            )
            frags.append((
                "class:dim",
                "  Esc to cancel · Enter to confirm · 1/2/3 to jump · "
                f"Tab to amend · {explain_hint}",
            ))
        return FormattedText(frags)

    kb = KeyBindings()

    def _in_option_mode() -> bool:
        return not awaiting_feedback[0]

    option_mode = Condition(_in_option_mode)

    @kb.add("up", filter=option_mode)
    def _(event: Any) -> None:
        cursor[0] = (cursor[0] - 1) % len(options)
        event.app.invalidate()

    @kb.add("down", filter=option_mode)
    def _(event: Any) -> None:
        cursor[0] = (cursor[0] + 1) % len(options)
        event.app.invalidate()

    @kb.add("enter", filter=option_mode)
    @kb.add("c-m", filter=option_mode)
    @kb.add("c-j", filter=option_mode)
    def _(event: Any) -> None:
        committed[0] = options[cursor[0]][0]
        event.app.exit()

    @kb.add("c-c", filter=option_mode)
    @kb.add("escape", filter=option_mode)
    def _(event: Any) -> None:
        committed[0] = None
        event.app.exit()

    @kb.add("tab", filter=option_mode)
    def _(event: Any) -> None:
        awaiting_feedback[0] = True
        event.app.invalidate()

    @kb.add("c-e", filter=option_mode)
    def _(event: Any) -> None:
        explain_visible[0] = not explain_visible[0]
        event.app.invalidate()

    # Number shortcuts — 1/2/3 jump the cursor AND commit in one keystroke.
    for i, (n, _label) in enumerate(options):
        @kb.add(str(n), filter=option_mode)
        def _(event: Any, idx: int = i) -> None:
            cursor[0] = idx
            committed[0] = options[idx][0]
            event.app.exit()

    feedback_mode = Condition(lambda: awaiting_feedback[0])

    @kb.add("enter", filter=feedback_mode)
    @kb.add("c-m", filter=feedback_mode)
    @kb.add("c-j", filter=feedback_mode)
    def _(event: Any) -> None:
        committed[0] = options[cursor[0]][0]
        feedback_returned[0] = feedback_buf[0]
        event.app.exit()

    @kb.add("escape", filter=feedback_mode)
    def _(event: Any) -> None:
        awaiting_feedback[0] = False
        feedback_buf[0] = ""
        event.app.invalidate()

    @kb.add("c-c", filter=feedback_mode)
    def _(event: Any) -> None:
        committed[0] = None
        feedback_returned[0] = ""
        event.app.exit()

    @kb.add("backspace", filter=feedback_mode)
    def _(event: Any) -> None:
        if feedback_buf[0]:
            feedback_buf[0] = feedback_buf[0][:-1]
            event.app.invalidate()

    @kb.add("<any>", filter=feedback_mode)
    def _(event: Any) -> None:
        data = event.data
        if data and data.isprintable():
            feedback_buf[0] += data
            event.app.invalidate()

    layout = Layout(Window(FormattedTextControl(_widget_fragments)))

    app: Application[Any] = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=False,
        mouse_support=False,
        erase_when_done=True,
    )

    async with prompt_mutex():
        if timeout is not None:
            try:
                await asyncio.wait_for(app.run_async(), timeout=timeout)
            except TimeoutError:
                if app.is_running:
                    app.exit()
                raise
        else:
            await app.run_async()
    return committed[0], feedback_returned[0]


async def run_generic_permission(
    *,
    tool: BaseTool,
    preview: str,
    tag: Literal["destructive", "read-only", "safe"],
    option_two_label: str,
    default_choice: int,
    args: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> tuple[int | None, str]:
    """Generic permission widget — used for any tool without a specialized UI.

    ``tag`` is accepted (for the Ctrl+E risk line) but deliberately NOT
    rendered in the widget header — the tool name + command preview
    are enough for the user to recognize risk.
    """
    title = _tool_title(tool)
    verb = _tool_verb(tool)

    header: list[tuple[str, str]] = [
        ("bold", f"  {title}\n"),
        ("", "\n"),
    ]
    if preview:
        header.append(("", f"    {preview}\n"))
    if verb:
        header.append(("class:dim", f"  {verb}\n"))
    header.append(("", "\n"))

    explanation_frags = _build_explanation(tool, args or {}, tag)

    return await _run_widget(
        header_frags=header,
        option_two_label=option_two_label,
        default_choice=default_choice,
        explanation_frags=explanation_frags,
        timeout=timeout,
    )
