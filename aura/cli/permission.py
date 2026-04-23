"""CLI permission asker — inline interactive list widget.

Matches claude-code's permission UX: no bordered dialog, no popup, no
"type a number and hit Enter". The user sees a short inline block with
three options, moves the cursor with ↑/↓ (or types 1/2/3 as a
shortcut), hits Enter to confirm, Ctrl+C or Esc to cancel.

Built on ``prompt_toolkit.Application`` so arrow keys are a first-class
interaction (the earlier typed-number flow captured ↑/↓ as raw escape
codes, which read as garbage). Rendered inline — no ``full_screen``,
no ``Dialog`` frame — so the block lives in the scrollback the same
way the model's output does.

Spec alignment: ``docs/specs/2026-04-19-aura-permission.md`` §8.1–§8.5.
The spec described a radiolist dialog for §8.1; this module deliberately
diverges (documented here so a future reader doesn't "fix" the
divergence by re-introducing the dialog).

One job: present the choice, capture the answer. The asker does NOT
decide (that's the hook), does NOT persist (that's the store), and does
NOT emit domain events beyond the two I/O-boundary journal lines
(``permission_asked`` / ``permission_answered``).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from langchain_core.tools import BaseTool
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from rich.console import Console

from aura.cli._coordination import pause_spinner_if_active, prompt_mutex
from aura.core.hooks.permission import AskerResponse, PermissionAsker
from aura.core.permissions.rule import Rule
from aura.core.permissions.rule_hint import derive_rule_hint
from aura.core.persistence import journal

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


def _tag(tool: BaseTool) -> Literal["destructive", "read-only", "safe"]:
    """Classification tag. Kept for journal/telemetry; NOT rendered in
    the widget header anymore (claude-code's design — see dogfood
    screenshot 2026-04-23 — uses a clean title instead of a risk glyph
    because the command preview itself makes the risk obvious)."""
    metadata = tool.metadata or {}
    if metadata.get("is_destructive"):
        return "destructive"
    if metadata.get("is_read_only"):
        return "read-only"
    return "safe"


def _preview(tool: BaseTool, args: dict[str, Any]) -> str:
    """One-line preview of this call's args; falls back to the tool name.

    Capped at ``_PREVIEW_MAX_CHARS`` — visual only; the tool still
    receives the full args. Also strips a leading ``"command: "``
    prefix when present: that prefix was redundant under the
    "Bash command" header.
    """
    preview_fn = (tool.metadata or {}).get("args_preview")
    if callable(preview_fn):
        try:
            out = preview_fn(args)
        except Exception:  # noqa: BLE001 — preview must never break the prompt
            return tool.name
        if isinstance(out, str) and out:
            # The bash / shell tools prefix with "command: " which was
            # useful when the widget was a generic dialog. In the new
            # design the header already names the tool, so strip it.
            if out.startswith("command: "):
                out = out[len("command: "):]
            if len(out) > _PREVIEW_MAX_CHARS:
                return out[: _PREVIEW_MAX_CHARS - 1] + "…"
            return out
    return tool.name


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


def _compose_option_two(
    tool: BaseTool, args: dict[str, Any],
) -> tuple[str, Rule, Literal["project", "session"]]:
    """Return ``(label, rule, scope)`` for the "yes, always" option.

    - Matcher present → project scope, precise pattern
    - No matcher → session scope, tool-wide fallback

    Wording mirrors claude-code's "Yes, and don't ask again for: X".
    """
    derived = derive_rule_hint(tool, args)
    if derived is not None:
        return (
            f"Yes, and don't ask again for `{derived.to_string()}` in this project",
            derived,
            "project",
        )
    return (
        f"Yes, and don't ask again for `{tool.name}` this session",
        Rule(tool=tool.name, content=None),
        "session",
    )


async def _pick_choice_interactive(
    *,
    tool: BaseTool,
    preview: str,
    tag: Literal["destructive", "read-only", "safe"],
    option_two_label: str,
    default_choice: int,
    args: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> tuple[int | None, str]:
    """Run an inline ``prompt_toolkit.Application`` that lets the user
    arrow-key through three options and commits on Enter.

    Layout (mirrors claude-code's PermissionDialog — screenshot
    2026-04-23, see module docstring):

        ──────────────────────────  (horizontal rule, separates from chat)

        <Tool-Title> command

          <preview>
        <verb>

        Do you want to proceed?
        ❯ 1. Yes
          2. Yes, and don't ask again for: ...
          3. No

        Esc to cancel · Enter to confirm · 1/2/3 to jump · Tab to amend

    Returns ``(choice, feedback)`` — ``choice`` is 1/2/3 on commit or
    ``None`` on cancel (Esc / Ctrl+C); ``feedback`` is the free-text
    note the user typed via the Tab-to-amend path, or ``""`` when
    absent (the common case).

    Tab-to-amend (matches claude-code's ``useShellPermissionFeedback``):
    user presses Tab while the cursor is on any option → a text-input
    buffer appears below the options with a dim ``Add feedback (Enter
    to submit, Esc to cancel)`` hint; printable keys append, Backspace
    pops, Enter commits the focused option with the feedback, Esc
    aborts the feedback entry (returns to option mode, no commit).

    Not ``full_screen`` — pt renders the block at the bottom of the
    scrollback and ``erase_when_done=True`` clears it on exit. An
    audit line is printed afterward (see ``_render_decision_audit_line``)
    so the transcript keeps a record even though the widget itself
    vanishes.

    ``tag`` is accepted (and passed through from the asker) so tests
    and telemetry keep the same contract, but is deliberately NOT
    rendered in the widget — the tool name + command preview are
    enough for the user to recognize risk.
    """
    # Pause any active thinking spinner BEFORE we install pt's render
    # loop — otherwise rich.Live and pt fight for the same region.
    await pause_spinner_if_active()

    title = _tool_title(tool)
    verb = _tool_verb(tool)
    options = (
        (1, "Yes"),
        (2, option_two_label),
        (3, "No"),
    )
    cursor: list[int] = [default_choice - 1]
    committed: list[int | None] = [None]
    # Tab-to-amend state. Single-slot lists so nested closures can
    # mutate without nonlocal ceremony (same pattern as ``cursor`` /
    # ``committed`` above).
    awaiting_feedback: list[bool] = [False]
    feedback_buf: list[str] = [""]
    feedback_returned: list[str] = [""]
    # Ctrl+E explanation-panel toggle. Starts collapsed; Ctrl+E flips.
    # Kept out of feedback mode so a Ctrl+E mid-note doesn't steal the
    # keystroke from the buffer.
    explain_visible: list[bool] = [False]
    explanation_frags = _build_explanation(tool, args or {}, tag)

    def _widget_fragments() -> FormattedText:
        # Single FormattedText for the whole widget — keeps the block
        # tight (no stray inter-Window blank rows) and renders as one
        # visual unit. The layout deliberately uses just dim + cyan
        # (for the selected option) — no red/yellow/green alarm colors,
        # matching claude-code's calm minimal palette.
        frags: list[tuple[str, str]] = [
            # Horizontal rule. Visually cuts the prompt block off from
            # the preceding conversation so the user's eye lands on
            # the question, not on buried scrollback.
            ("class:dim", _SEPARATOR + "\n"),
            ("", "\n"),
            ("bold", f"  {title}\n"),
            ("", "\n"),
        ]
        if preview:
            frags.append(("", f"    {preview}\n"))
        if verb:
            frags.append(("class:dim", f"  {verb}\n"))
        frags.append(("", "\n"))
        if explain_visible[0]:
            # Insert the pre-built explanation block between the verb
            # and the question. Pre-built (not lazily rebuilt on every
            # render) because tool/args don't change during the widget's
            # lifetime — so we avoid re-walking description lines on
            # every keystroke.
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
            # Feedback-entry mode: show the buffer on its own line with
            # a caret cursor, plus the help text beneath. The hint line
            # replaces the usual footer — elide "Tab to amend" since
            # the input line itself is the cue. Ctrl+E is intentionally
            # NOT advertised here: it's inert in feedback mode.
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

    # --- Option-mode bindings (cursor navigation + number shortcuts +
    # Tab to enter feedback mode). Guarded with a filter so they go
    # inert while the feedback buffer is active — otherwise typing "1"
    # as part of a note would commit the prompt.
    from prompt_toolkit.filters import Condition

    option_mode = Condition(_in_option_mode)

    @kb.add("up", filter=option_mode)
    def _(event: Any) -> None:
        cursor[0] = (cursor[0] - 1) % len(options)
        event.app.invalidate()

    @kb.add("down", filter=option_mode)
    def _(event: Any) -> None:
        cursor[0] = (cursor[0] + 1) % len(options)
        event.app.invalidate()

    # Enter / Return. Real terminals send ``\r`` (CR), which pt maps to
    # ``c-m``; some callers send ``\n`` (LF) → ``c-j``. Bind both so
    # neither a real tty nor a piped stdin makes Enter silently no-op.
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

    # Ctrl+E — toggle the static explanation panel. Option-mode-only
    # (feedback mode's ``<any>`` binding would otherwise eat it as the
    # literal ``\x05`` byte, which is non-printable and gets filtered
    # anyway — explicit option_mode filter is defence-in-depth so the
    # keystroke is unambiguously ignored during feedback entry).
    @kb.add("c-e", filter=option_mode)
    def _(event: Any) -> None:
        explain_visible[0] = not explain_visible[0]
        event.app.invalidate()

    # Number shortcuts — 1/2/3 jump the cursor AND commit in one keystroke
    # (matches claude-code's number-to-commit convention).
    for i, (n, _label) in enumerate(options):
        @kb.add(str(n), filter=option_mode)
        def _(event: Any, idx: int = i) -> None:
            cursor[0] = idx
            committed[0] = options[idx][0]
            event.app.exit()

    # --- Feedback-mode bindings. Enter commits with feedback, Esc aborts
    # feedback entry (returns to option mode without committing),
    # Backspace pops, any other printable key appends.
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
        # Esc cancels feedback entry but keeps the dialog open so the
        # user can still pick an option. Clear the buffer so the next
        # Tab starts fresh.
        awaiting_feedback[0] = False
        feedback_buf[0] = ""
        event.app.invalidate()

    @kb.add("c-c", filter=feedback_mode)
    def _(event: Any) -> None:
        # Ctrl+C is still a hard cancel — tear down the whole prompt.
        committed[0] = None
        feedback_returned[0] = ""
        event.app.exit()

    @kb.add("backspace", filter=feedback_mode)
    def _(event: Any) -> None:
        if feedback_buf[0]:
            feedback_buf[0] = feedback_buf[0][:-1]
            event.app.invalidate()

    # Any printable character appends to the feedback buffer. Using
    # ``<any>`` covers single-char key presses that aren't otherwise
    # bound; the guard filters out non-printable control characters.
    @kb.add("<any>", filter=feedback_mode)
    def _(event: Any) -> None:
        data = event.data
        if data and data.isprintable():
            feedback_buf[0] += data
            event.app.invalidate()

    # Single-Window layout: one FormattedTextControl whose fragments
    # include embedded ``\n`` characters. pt wraps it to the correct
    # height automatically — no manual ``height=`` tuning that produces
    # padding lines.
    layout = Layout(Window(FormattedTextControl(_widget_fragments)))

    app: Application[Any] = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=False,
        mouse_support=False,
        erase_when_done=True,  # clear the widget area on exit; we'll
                               # re-print the chosen option explicitly so
                               # the scrollback keeps an audit trail.
    )
    # Key bindings attach to the Application, which dispatches them
    # regardless of focused element (and we have no focusable element
    # on purpose — the controls are read-only FormattedTextControls).

    # Serialize against any other in-flight interactive prompt
    # (permission from a subagent, ask_user_question, future confirm
    # dialogs). pt's Application requires exclusive terminal ownership;
    # running two concurrently garbles both. The mutex is asyncio
    # FIFO-fair so a queue of prompts resolves in arrival order — one
    # widget at a time, just like claude-code's promptQueue popping
    # queue[0] at a time.
    #
    # ``timeout`` wraps the pt Application with ``asyncio.wait_for``: if
    # the user doesn't answer within N seconds the coroutine raises
    # ``asyncio.TimeoutError``, which the caller resolves to a denial
    # (fail-safe — unattended prompts must NOT hang the turn forever).
    # ``None`` preserves legacy "wait forever" behavior. The pt Application
    # is explicitly cancelled on timeout so its render loop tears down
    # cleanly before we return.
    async with prompt_mutex():
        if timeout is not None:
            try:
                await asyncio.wait_for(app.run_async(), timeout=timeout)
            except TimeoutError:
                # pt may still own the terminal; exit the app so the
                # render loop unwinds and the cursor returns to a sane
                # position before the caller prints the deny line.
                if app.is_running:
                    app.exit()
                raise
        else:
            await app.run_async()
    return committed[0], feedback_returned[0]


def _render_decision_audit_line(
    console: Console,
    *,
    tool: BaseTool,
    tag: Literal["destructive", "read-only", "safe"],
    preview: str,
    choice: int | None,
    feedback: str = "",
) -> None:
    """Print a one-line audit trace of the decision the user just made.

    ``erase_when_done=True`` on the Application removes the widget from
    the scrollback, which is the right UX (no cluttered option lists
    piling up) but leaves no record of what happened. Log a dim line
    here so the transcript reads linearly.

    Format: ``● bash(pwd) — yes`` / ``⚠ bash(rm) — no`` / etc.
    Non-empty ``feedback`` (from the Tab-to-amend flow) appears as a
    dim trailing `` — "note"`` so the scrollback records what the
    user actually said.
    """
    color_map = {"destructive": "red", "read-only": "green", "safe": "yellow"}
    color = color_map[tag]
    marker = "⚠" if tag == "destructive" else "●"
    decision = {1: "yes", 2: "yes (always)", 3: "no", None: "cancelled"}[choice]
    suffix = f' — "{feedback}"' if feedback else ""
    console.print(
        f"[{color}]{marker}[/{color}] [bold]{tool.name}[/bold]"
        f"[dim]({preview}) — {decision}{suffix}[/dim]"
    )


def make_cli_asker(
    console: Console | None = None,
    *,
    timeout: float | None = None,
) -> PermissionAsker:
    """Return a ``PermissionAsker`` backed by an inline interactive widget.

    ``console`` — optional rich Console for tests (StringIO-backed for
    capture). Production path creates a fresh one.

    ``timeout`` — seconds to wait for the user to respond before
    treating the non-response as a denial (fail-safe for unattended /
    headless sessions). ``None`` preserves legacy "wait forever"
    behavior. The CLI threads ``PermissionsConfig.prompt_timeout_sec``
    (default 300s = 5 min) through to this kwarg.
    """
    _console = console or Console()

    async def _ask(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,  # noqa: ARG001 — part of Protocol; derivation is local
    ) -> AskerResponse:
        tag = _tag(tool)
        preview = _preview(tool, args)
        option_two_label, option_two_rule, option_two_scope = _compose_option_two(
            tool, args,
        )
        default_choice = 3 if tag == "destructive" else 1

        journal.write(
            "permission_asked",
            tool=tool.name,
            args_preview=preview,
            rule_hint=option_two_rule.to_string(),
        )

        try:
            choice, feedback = await _pick_choice_interactive(
                tool=tool,
                preview=preview,
                tag=tag,
                option_two_label=option_two_label,
                default_choice=default_choice,
                args=args,
                timeout=timeout,
            )
        except TimeoutError:
            # Fail-safe: unattended / stale sessions MUST NOT hang the
            # turn forever. Resolve a non-response to deny and annotate
            # the journal so the audit trail records *why* the tool was
            # blocked (matters for headless / CI runs where no human
            # sees the prompt).
            journal.write(
                "permission_prompt_timeout",
                tool=tool.name,
                timeout_sec=timeout,
            )
            journal.write(
                "permission_answered",
                tool=tool.name,
                choice="deny",
                reason="timeout",
            )
            return AskerResponse(choice="deny")
        except (KeyboardInterrupt, SystemExit):
            # Defensive — pt normally consumes these via the c-c /
            # escape bindings, but an outer Ctrl+C that propagates
            # past pt should still resolve to deny (not tear down the
            # turn).
            journal.write("permission_answered", tool=tool.name, choice="deny")
            return AskerResponse(choice="deny")
        except Exception as exc:  # noqa: BLE001 — no-TTY / pt failures
            journal.write(
                "permission_prompt_unavailable",
                tool=tool.name,
                detail=repr(exc),
            )
            return AskerResponse(choice="deny")

        _render_decision_audit_line(
            _console,
            tool=tool,
            tag=tag,
            preview=preview,
            choice=choice,
            feedback=feedback,
        )

        # Thread ``feedback`` into the journal event AND the returned
        # AskerResponse. Empty string is the common case (user didn't
        # press Tab); non-empty flows on to the hook, which embeds it
        # into the model-facing deny message and the decision journal.
        answered_extra: dict[str, Any] = {}
        if feedback:
            answered_extra["feedback"] = feedback
        if choice == 1:
            journal.write(
                "permission_answered",
                tool=tool.name,
                choice="accept",
                **answered_extra,
            )
            return AskerResponse(choice="accept", feedback=feedback)
        if choice == 2:
            journal.write(
                "permission_answered",
                tool=tool.name,
                choice="always",
                **answered_extra,
            )
            return AskerResponse(
                choice="always",
                scope=option_two_scope,
                rule=option_two_rule,
                feedback=feedback,
            )
        # choice == 3 (explicit No) OR None (Ctrl+C / Esc cancelled)
        journal.write(
            "permission_answered",
            tool=tool.name,
            choice="deny",
            **answered_extra,
        )
        return AskerResponse(choice="deny", feedback=feedback)

    return _ask


def print_bypass_banner(console: Console) -> None:
    """Print the bypass-mode startup warning (spec §8.5)."""
    console.print(
        "[bold red]⚠  PERMISSION CHECKS DISABLED — "
        "all tool calls will run without asking[/bold red]",
    )
