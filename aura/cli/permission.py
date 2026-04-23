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
    tag: Literal["destructive", "read-only", "safe"],  # noqa: ARG001
    option_two_label: str,
    default_choice: int,
) -> int | None:
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

        Esc to cancel · Enter to confirm · 1/2/3 to jump

    Returns 1/2/3 on commit, ``None`` on cancel (Esc / Ctrl+C).

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
        frags.append(("bold", "  Do you want to proceed?\n"))
        for i, (n, label) in enumerate(options):
            is_sel = i == cursor[0]
            if is_sel:
                frags.append(("ansicyan bold", f"  ❯ {n}. {label}\n"))
            else:
                frags.append(("class:dim", f"    {n}. {label}\n"))
        frags.append(("", "\n"))
        frags.append((
            "class:dim",
            "  Esc to cancel · Enter to confirm · 1/2/3 to jump",
        ))
        return FormattedText(frags)

    kb = KeyBindings()

    @kb.add("up")
    def _(event: Any) -> None:
        cursor[0] = (cursor[0] - 1) % len(options)
        event.app.invalidate()

    @kb.add("down")
    def _(event: Any) -> None:
        cursor[0] = (cursor[0] + 1) % len(options)
        event.app.invalidate()

    # Enter / Return. Real terminals send ``\r`` (CR), which pt maps to
    # ``c-m``; some callers send ``\n`` (LF) → ``c-j``. Bind both so
    # neither a real tty nor a piped stdin makes Enter silently no-op.
    @kb.add("enter")
    @kb.add("c-m")
    @kb.add("c-j")
    def _(event: Any) -> None:
        committed[0] = options[cursor[0]][0]
        event.app.exit()

    @kb.add("c-c")
    @kb.add("escape")
    def _(event: Any) -> None:
        committed[0] = None
        event.app.exit()

    # Number shortcuts — 1/2/3 jump the cursor AND commit in one keystroke
    # (matches claude-code's number-to-commit convention).
    for i, (n, _label) in enumerate(options):
        @kb.add(str(n))
        def _(event: Any, idx: int = i) -> None:
            cursor[0] = idx
            committed[0] = options[idx][0]
            event.app.exit()

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
    async with prompt_mutex():
        await app.run_async()
    return committed[0]


def _render_decision_audit_line(
    console: Console,
    *,
    tool: BaseTool,
    tag: Literal["destructive", "read-only", "safe"],
    preview: str,
    choice: int | None,
) -> None:
    """Print a one-line audit trace of the decision the user just made.

    ``erase_when_done=True`` on the Application removes the widget from
    the scrollback, which is the right UX (no cluttered option lists
    piling up) but leaves no record of what happened. Log a dim line
    here so the transcript reads linearly.

    Format: ``● bash(pwd) — yes`` / ``⚠ bash(rm) — no`` / etc.
    """
    color_map = {"destructive": "red", "read-only": "green", "safe": "yellow"}
    color = color_map[tag]
    marker = "⚠" if tag == "destructive" else "●"
    decision = {1: "yes", 2: "yes (always)", 3: "no", None: "cancelled"}[choice]
    console.print(
        f"[{color}]{marker}[/{color}] [bold]{tool.name}[/bold]"
        f"[dim]({preview}) — {decision}[/dim]"
    )


def make_cli_asker(console: Console | None = None) -> PermissionAsker:
    """Return a ``PermissionAsker`` backed by an inline interactive widget.

    ``console`` — optional rich Console for tests (StringIO-backed for
    capture). Production path creates a fresh one.
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
            choice = await _pick_choice_interactive(
                tool=tool,
                preview=preview,
                tag=tag,
                option_two_label=option_two_label,
                default_choice=default_choice,
            )
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
            _console, tool=tool, tag=tag, preview=preview, choice=choice,
        )

        if choice == 1:
            journal.write("permission_answered", tool=tool.name, choice="accept")
            return AskerResponse(choice="accept")
        if choice == 2:
            journal.write("permission_answered", tool=tool.name, choice="always")
            return AskerResponse(
                choice="always",
                scope=option_two_scope,
                rule=option_two_rule,
            )
        # choice == 3 (explicit No) OR None (Ctrl+C / Esc cancelled)
        journal.write("permission_answered", tool=tool.name, choice="deny")
        return AskerResponse(choice="deny")

    return _ask


def print_bypass_banner(console: Console) -> None:
    """Print the bypass-mode startup warning (spec §8.5)."""
    console.print(
        "[bold red]⚠  PERMISSION CHECKS DISABLED — "
        "all tool calls will run without asking[/bold red]",
    )
