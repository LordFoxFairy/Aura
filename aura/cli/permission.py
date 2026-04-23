"""CLI permission asker — inline numbered prompt.

Matches claude-code's permission UX: a short block printed into the
scrollback, then a ``❯`` cursor for the user's numeric choice. No
bordered dialog, no popup — the permission decision lives in the
conversation timeline the same way the model's output does, so the
user's flow stays linear top-to-bottom.

Spec alignment: ``docs/specs/2026-04-19-aura-permission.md`` §8.1–§8.5.
(The spec described a radiolist dialog for §8.1; this module deliberately
diverges to the inline form because the boxed dialog fought the rest of
the CLI's rich+prompt_toolkit REPL layout.)

One job: present the choice, capture the answer. The asker does NOT
decide (that's the hook), does NOT persist (that's the store), and does
NOT emit domain events beyond the two I/O-boundary journal lines
(``permission_asked`` / ``permission_answered``).
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from prompt_toolkit import PromptSession
from rich.console import Console

from aura.core.hooks.permission import AskerResponse, PermissionAsker
from aura.core.permissions.rule import Rule
from aura.core.permissions.rule_hint import derive_rule_hint
from aura.core.persistence import journal

# Visual cap on the args preview so a huge ``bash`` command (or similar)
# doesn't overflow the terminal. Visual only — the tool still receives
# the full args.
_PREVIEW_MAX_CHARS = 200


_TAG_STYLE: dict[str, tuple[str, str]] = {
    # color → rich markup name; marker → leading glyph on the header line.
    "destructive": ("red", "⚠"),
    "read-only": ("green", "●"),
    "safe": ("yellow", "●"),
}


def _tag(tool: BaseTool) -> Literal["destructive", "read-only", "safe"]:
    """Classification tag for the header line.

    Precedence: destructive > read-only > safe. The explicit ordering
    guards against tools that (wrongly) set both flags.
    """
    metadata = tool.metadata or {}
    if metadata.get("is_destructive"):
        return "destructive"
    if metadata.get("is_read_only"):
        return "read-only"
    return "safe"


def _preview(tool: BaseTool, args: dict[str, Any]) -> str:
    """One-line preview of this call's args; falls back to the tool name.

    Capped at ``_PREVIEW_MAX_CHARS`` so a bash command with thousands of
    characters can't blow out the terminal. Cap is visual only — the
    full args still reach the tool.
    """
    preview_fn = (tool.metadata or {}).get("args_preview")
    if callable(preview_fn):
        try:
            out = preview_fn(args)
        except Exception:  # noqa: BLE001 — preview must never break the prompt
            return tool.name
        if isinstance(out, str) and out:
            if len(out) > _PREVIEW_MAX_CHARS:
                return out[: _PREVIEW_MAX_CHARS - 1] + "…"
            return out
    return tool.name


def _compose_option_two(
    tool: BaseTool, args: dict[str, Any],
) -> tuple[str, Rule, Literal["project", "session"]]:
    """Return ``(label, rule, scope)`` for the "yes, always" option.

    - Precise rule (matcher present) → project scope, specific pattern
    - No matcher → session scope, tool-wide fallback
    """
    derived = derive_rule_hint(tool, args)
    if derived is not None:
        return (
            f"Yes, and always allow `{derived.to_string()}` in this project",
            derived,
            "project",
        )
    return (
        f"Yes, and always allow `{tool.name}` for this session",
        Rule(tool=tool.name, content=None),
        "session",
    )


def _render_prompt_block(
    console: Console,
    *,
    tool: BaseTool,
    preview: str,
    tag: Literal["destructive", "read-only", "safe"],
    option_two_label: str,
    default_choice: int,
) -> None:
    """Print the inline permission block to ``console``.

    Layout (top → bottom):

        ● <tool>  ·  <tag>
          <preview>          (omitted when preview == tool.name)

          ❯ 1. Yes
            2. <option-two label>
            3. No

          Enter = default (N) · Ctrl+C to cancel

    Rendered with rich markup — no box, no border. The only color
    accent is the tag marker on the header (red / green / yellow for
    destructive / read-only / safe). The rest is dim to sit quietly
    against the surrounding conversation.
    """
    color, marker = _TAG_STYLE[tag]
    console.print()
    console.print(
        f"[{color}]{marker}[/{color}] [bold]{tool.name}[/bold]"
        f"  [dim]·  {tag}[/dim]"
    )
    if preview and preview != tool.name:
        console.print(f"  [dim]{preview}[/dim]")
    console.print()
    for n, label in ((1, "Yes"), (2, option_two_label), (3, "No")):
        cursor = "❯" if n == default_choice else " "
        console.print(f"  {cursor} {n}. {label}")
    console.print()
    console.print(
        f"  [dim]Enter = default ({default_choice}) · Ctrl+C to cancel[/dim]"
    )


def _parse_choice(
    raw: str, *, default: int,
) -> int | None:
    """Return parsed choice (1/2/3), or ``None`` to signal "reprompt".

    Accepted tokens (case-insensitive, stripped):
      - Empty line → default (1 for safe, 3 for destructive)
      - ``1`` / ``y`` / ``yes`` → 1 (once)
      - ``2`` / ``a`` / ``always`` → 2 (always, with rule)
      - ``3`` / ``n`` / ``no`` → 3 (deny)
      - anything else → ``None`` (caller prints a hint and re-reads)
    """
    tok = raw.strip().lower()
    if tok == "":
        return default
    if tok in {"1", "y", "yes"}:
        return 1
    if tok in {"2", "a", "always"}:
        return 2
    if tok in {"3", "n", "no"}:
        return 3
    return None


async def _read_choice(prompt: str = "  ❯ ") -> str:
    """Read a single line from the terminal.

    Uses a transient ``prompt_toolkit.PromptSession`` rather than raw
    ``input()`` for two reasons:
      1. Ctrl+C / Ctrl+D raise ``KeyboardInterrupt`` / ``EOFError``
         without leaving the tty in a weird state (input() via
         asyncio.to_thread can't be cancelled cleanly).
      2. Arrow-key history and terminal line editing "just work", so
         answering "1" feels the same as typing into the main REPL.

    Each ask creates its own session (cheap) — keeps state from
    leaking between permission prompts and avoids a module-level
    mutable singleton.
    """
    session: PromptSession[str] = PromptSession()
    return await session.prompt_async(prompt)


def make_cli_asker(console: Console | None = None) -> PermissionAsker:
    """Return a ``PermissionAsker`` backed by an inline prompt.

    ``console`` — optional rich Console (tests inject a StringIO-backed
    one for output capture). Production path creates a fresh Console.
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

        _render_prompt_block(
            _console,
            tool=tool,
            preview=preview,
            tag=tag,
            option_two_label=option_two_label,
            default_choice=default_choice,
        )

        try:
            while True:
                raw = await _read_choice()
                choice = _parse_choice(raw, default=default_choice)
                if choice is not None:
                    break
                _console.print(
                    "  [yellow]Please enter 1, 2, or 3 (Enter for default).[/yellow]"
                )
        except (KeyboardInterrupt, SystemExit):
            # Ctrl+C on the permission prompt is a first-class deny —
            # matches claude-code's "Esc = deny" convention but without
            # propagating cancellation up to the agent turn (the user
            # said "no to this tool call", not "kill the whole session").
            _console.print()
            journal.write("permission_answered", tool=tool.name, choice="deny")
            return AskerResponse(choice="deny")
        except EOFError:
            # Piped stdin hit end (or Ctrl+D) → fail closed to deny.
            journal.write("permission_answered", tool=tool.name, choice="deny")
            return AskerResponse(choice="deny")
        except Exception as exc:  # noqa: BLE001 — no-TTY / prompt failures
            journal.write(
                "permission_prompt_unavailable",
                tool=tool.name,
                detail=repr(exc),
            )
            return AskerResponse(choice="deny")

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
        # choice == 3
        journal.write("permission_answered", tool=tool.name, choice="deny")
        return AskerResponse(choice="deny")

    return _ask


def print_bypass_banner(console: Console) -> None:
    """Print the bypass-mode startup warning (spec §8.5)."""
    console.print(
        "[bold red]⚠  PERMISSION CHECKS DISABLED — "
        "all tool calls will run without asking[/bold red]",
    )
