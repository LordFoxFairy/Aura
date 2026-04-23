"""CLI implementation of the ``ask_user_question`` tool's asker callable.

Mirrors ``aura/cli/permission.py``: an inline numbered / free-text
prompt rather than a bordered dialog. Keeps every user-facing input
surface consistent with the rest of the REPL — no popups, no boxes,
everything lives in the scrollback.

Cancellation (Ctrl+C / Ctrl+D / Esc) returns an empty string so the
LLM always gets a well-typed ``{"answer": ""}`` result rather than
raising. The LLM can then decide whether to retry, pivot, or give up.
"""

from __future__ import annotations

from prompt_toolkit import PromptSession
from rich.console import Console

from aura.tools.ask_user import QuestionAsker


async def _read_line(prompt: str) -> str:
    """Read one line via a transient prompt_toolkit session.

    Each call spins up its own session — cheap, avoids global state,
    matches the pattern in ``permission._read_choice``.
    """
    session: PromptSession[str] = PromptSession()
    return await session.prompt_async(prompt)


def _render_multi_choice(
    console: Console, question: str, options: list[str], default: str | None,
) -> int:
    """Print the question + numbered options; return the default index.

    The index (1-based) of ``default`` in ``options`` is returned so
    the caller knows what a bare Enter resolves to.
    """
    console.print()
    console.print(f"[bold]?[/bold] {question}")
    console.print()
    default_idx = 1
    for i, opt in enumerate(options, start=1):
        if default is not None and opt == default:
            default_idx = i
        cursor = "❯" if (default is not None and opt == default) else " "
        console.print(f"  {cursor} {i}. {opt}")
    console.print()
    console.print(
        f"  [dim]Enter = default ({default_idx}) · "
        "type the number · Ctrl+C to cancel[/dim]"
    )
    return default_idx


def _parse_multi_choice(raw: str, *, options: list[str], default_idx: int) -> str | None:
    """Return the chosen option string, or ``None`` to signal "reprompt".

    Accepts: empty (→ default), a 1-based index, or an exact option
    string match. Anything else → None.
    """
    tok = raw.strip()
    if tok == "":
        return options[default_idx - 1]
    if tok.isdigit():
        idx = int(tok)
        if 1 <= idx <= len(options):
            return options[idx - 1]
        return None
    # Allow exact option text too (makes e.g. "yes" answer question with
    # options ["yes","no"] without the user counting).
    for opt in options:
        if tok.lower() == opt.lower():
            return opt
    return None


def make_cli_user_asker(console: Console | None = None) -> QuestionAsker:
    """Return an async callable the ``ask_user_question`` tool delegates to.

    ``console`` — optional rich Console for tests; production creates
    a fresh one.
    """
    _console = console or Console()

    async def _ask(
        question: str, options: list[str] | None, default: str | None,
    ) -> str:
        if options:
            default_idx = _render_multi_choice(_console, question, options, default)
            try:
                while True:
                    raw = await _read_line("  ❯ ")
                    chosen = _parse_multi_choice(
                        raw, options=options, default_idx=default_idx,
                    )
                    if chosen is not None:
                        return chosen
                    _console.print(
                        f"  [yellow]Please enter 1–{len(options)}, "
                        "an option name, or Enter for default.[/yellow]"
                    )
            except (KeyboardInterrupt, SystemExit, EOFError):
                # Same contract as the old dialog: cancellation → empty
                # string so the LLM sees a well-typed result.
                _console.print()
                return ""
            except Exception:  # noqa: BLE001 — no-TTY / prompt failures
                return ""

        # Free-text path.
        _console.print()
        _console.print(f"[bold]?[/bold] {question}")
        default_hint = f" [dim](default: {default})[/dim]" if default else ""
        try:
            raw = await _read_line(f"  ❯{default_hint} ")
        except (KeyboardInterrupt, SystemExit, EOFError):
            _console.print()
            return ""
        except Exception:  # noqa: BLE001
            return ""
        tok = raw.strip()
        if tok == "" and default:
            return default
        return tok

    return _ask
