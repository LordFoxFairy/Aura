"""CLI implementation of the ``ask_user_question`` tool's asker callable.

Mirrors ``aura/cli/permission.py``: an inline interactive widget rendered
via ``prompt_toolkit.Application`` — arrow-key selection for multi-choice
questions, an inline line editor for free-text. No bordered dialog.

Shares the ``prompt_mutex`` from ``aura.cli._coordination`` with the
permission asker so that concurrent requests (e.g. a subagent asking a
question while the parent is approving a tool) serialize FIFO, matching
claude-code's single-queue-single-render-at-a-time model.

Cancellation (Ctrl+C / Ctrl+D / Esc) returns an empty string so the LLM
always gets a well-typed ``{"answer": ""}`` result.
"""

from __future__ import annotations

from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from rich.console import Console

from aura.cli._coordination import pause_spinner_if_active, prompt_mutex
from aura.tools.ask_user import QuestionAsker

_SEPARATOR = "─" * 78


async def _pick_choice_interactive(
    question: str,
    options: list[str],
    default: str | None,
) -> str | None:
    """Interactive arrow-key picker for multi-option questions.

    Layout mirrors the permission widget:

        ──────────────────────────

          Question

          ❯ 1. <option>
            2. <option>
            ...

        Esc / Enter / 1-9 to jump

    Returns the chosen option string, or ``None`` on Esc / Ctrl+C.
    Serializes against the permission widget via ``prompt_mutex`` so
    two concurrent asks don't fight for the terminal.
    """
    await pause_spinner_if_active()

    default_idx = 0
    if default is not None:
        try:
            default_idx = options.index(default)
        except ValueError:
            default_idx = 0
    cursor: list[int] = [default_idx]
    committed: list[str | None] = [None]

    def _frags() -> FormattedText:
        frags: list[tuple[str, str]] = [
            ("class:dim", _SEPARATOR + "\n"),
            ("", "\n"),
            ("bold", f"  {question}\n"),
            ("", "\n"),
        ]
        for i, opt in enumerate(options):
            n = i + 1
            is_sel = i == cursor[0]
            if is_sel:
                frags.append(("ansicyan bold", f"  ❯ {n}. {opt}\n"))
            else:
                frags.append(("class:dim", f"    {n}. {opt}\n"))
        frags.append(("", "\n"))
        frags.append((
            "class:dim",
            f"  Esc to cancel · Enter to confirm · 1-{len(options)} to jump",
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

    @kb.add("enter")
    @kb.add("c-m")
    @kb.add("c-j")
    def _(event: Any) -> None:
        committed[0] = options[cursor[0]]
        event.app.exit()

    @kb.add("c-c")
    @kb.add("escape")
    def _(event: Any) -> None:
        committed[0] = None
        event.app.exit()

    # Number shortcuts 1..len(options). Only single-digit — a 10-option
    # prompt should use a scrolling list, not this widget.
    for i in range(len(options)):
        if i >= 9:
            break
        @kb.add(str(i + 1))
        def _(event: Any, idx: int = i) -> None:
            cursor[0] = idx
            committed[0] = options[idx]
            event.app.exit()

    layout = Layout(Window(FormattedTextControl(_frags)))
    app: Application[Any] = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=False,
        mouse_support=False,
        erase_when_done=True,
    )
    async with prompt_mutex():
        await app.run_async()
    return committed[0]


async def _read_free_text(question: str, default: str | None) -> str | None:
    """Prompt the user for free-text input inline.

    Uses a transient ``PromptSession`` — simpler than a full pt
    Application because free-text doesn't need cursor-navigation.
    Serialized against other prompts via ``prompt_mutex``.
    """
    await pause_spinner_if_active()

    # Print the question above the input line so the user sees what
    # they're answering. The prompt_async line editor then reads on
    # its own line, which is the claude-code pattern.
    # We use stdout directly (via rich Console) for the question so
    # the label is part of the scrollback, not a pt widget.
    console = Console()
    console.print()
    console.print(_SEPARATOR, style="dim")
    console.print()
    console.print(f"[bold]{question}[/bold]")
    default_hint = f" [dim](default: {default})[/dim]" if default else ""
    console.print(f"  [dim]Enter to confirm · Esc / Ctrl+C to cancel{default_hint}[/dim]")
    console.print()

    session: PromptSession[str] = PromptSession()
    try:
        async with prompt_mutex():
            raw = await session.prompt_async("  ❯ ")
    except (KeyboardInterrupt, SystemExit, EOFError):
        return None
    except Exception:  # noqa: BLE001 — no-TTY / prompt failures
        return None
    tok = raw.strip()
    if tok == "" and default:
        return default
    return tok


def make_cli_user_asker(console: Console | None = None) -> QuestionAsker:  # noqa: ARG001 — kept for test compat
    """Return an async callable the ``ask_user_question`` tool delegates to.

    ``console`` — accepted for backwards compatibility with tests and
    callers that used to inject a StringIO-backed Console; the widget
    now renders via pt directly, so the parameter is unused. A
    deprecation nudge would land too late — we already committed to a
    stable PermissionAsker signature — so the slot just stays.
    """

    async def _ask(
        question: str, options: list[str] | None, default: str | None,
    ) -> str:
        if options:
            try:
                chosen = await _pick_choice_interactive(question, options, default)
            except (KeyboardInterrupt, SystemExit):
                return ""
            except Exception:  # noqa: BLE001
                return ""
            return chosen if chosen is not None else ""

        # Free-text path
        raw = await _read_free_text(question, default)
        return raw if raw is not None else ""

    return _ask
