"""CLI permission asker — list-select prompt via ``prompt_toolkit``.

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §8.1–§8.5.

One job: present the choice, capture the answer. The asker does NOT decide
(that's the hook), does NOT persist (that's the store), and does NOT emit
domain events beyond the two I/O-boundary journal lines it's responsible for
(``permission_asked`` / ``permission_answered``).

Rule derivation for option 2 delegates to :func:`derive_rule_hint`:

- precise pattern rule returned → label says "in this project", ``scope="project"``
- ``None`` → label says "for this session", ``scope="session"`` with a tool-wide rule
"""

from __future__ import annotations

from html import escape as html_escape
from typing import Any, Literal

from langchain_core.tools import BaseTool
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import radiolist_dialog
from rich.console import Console

from aura.core.hooks.permission import AskerResponse, PermissionAsker
from aura.core.permissions.rule import Rule
from aura.core.permissions.rule_hint import derive_rule_hint
from aura.core.persistence import journal

# Visual cap on prompt preview text so a huge ``bash`` command (or similar)
# doesn't blow out the radiolist dialog layout. Visual only — the tool still
# receives the full args.
_PREVIEW_MAX_CHARS = 200

_FOOTER_SHORTCUTS = "↑/↓ select · Enter confirm · Esc deny"


def _tag(tool: BaseTool) -> str:
    """Classification tag for the title line (spec §8.1).

    Precedence: destructive > read-only > safe. Mutually exclusive by how
    tools are tagged today; the explicit precedence here guards against
    tools that (wrongly) set both flags.
    """
    metadata = tool.metadata or {}
    if metadata.get("is_destructive"):
        return "destructive"
    if metadata.get("is_read_only"):
        return "read-only"
    return "safe"


def _preview(tool: BaseTool, args: dict[str, Any]) -> str:
    """One-line preview of this call's args; falls back to the tool name.

    Preview output is capped at ``_PREVIEW_MAX_CHARS`` so a tool returning a
    huge string (e.g. ``bash`` with a 5000-char command) doesn't blow out the
    radiolist dialog layout. The cap is visual only — the full args still
    reach the tool; this just keeps the prompt readable.
    """
    preview_fn = (tool.metadata or {}).get("args_preview")
    if callable(preview_fn):
        try:
            out = preview_fn(args)
        except Exception:  # noqa: BLE001 — preview must never break the prompt
            return tool.name
        if isinstance(out, str) and out:
            if len(out) > _PREVIEW_MAX_CHARS:
                return out[: _PREVIEW_MAX_CHARS - 1] + "\u2026"
            return out
    return tool.name


def _risk_indicator_html(tool: BaseTool) -> str:
    """Colored risk indicator for the dialog header (prompt_toolkit HTML).

    Mapping:
      - destructive → red "⚠ destructive"
      - read-only   → green "● read-only"
      - neither     → yellow "? unknown"

    The surrounding ``<style ...>`` tags are prompt_toolkit HTML syntax so the
    terminal renders color; if HTML isn't supported the tag still reads as
    plain text.
    """
    metadata = tool.metadata or {}
    if metadata.get("is_destructive"):
        return '<style fg="ansired">⚠ destructive</style>'
    if metadata.get("is_read_only"):
        return '<style fg="ansigreen">● read-only</style>'
    return '<style fg="ansiyellow">? unknown</style>'


def _build_dialog_text(tool: BaseTool, args: dict[str, Any]) -> str:
    """Compose the body text for the permission dialog.

    Pure function — no I/O, no journal writes. Tested in isolation; the live
    ``_ask`` coroutine calls this to build the ``text`` kwarg handed to
    ``radiolist_dialog``. The output is a prompt_toolkit HTML source string.

    Layout (top to bottom):

        <preview line, truncated to _PREVIEW_MAX_CHARS>
        <risk indicator: destructive / read-only / unknown>
        If you choose "allow always", <rule hint>:
            <rule string OR "all <tool> calls will be auto-allowed">
        <dim footer with keyboard shortcuts>
    """
    preview = _preview(tool, args)
    indicator = _risk_indicator_html(tool)
    derived = derive_rule_hint(tool, args)
    if derived is not None:
        rule_explanation = (
            'If you choose "allow always", this rule will be saved:\n'
            f"    {derived.to_string()}"
        )
    else:
        rule_explanation = (
            f'If you choose "allow always", all {tool.name} '
            "calls will be auto-allowed (tool-wide)"
        )
    # Preview may contain ``<`` / ``>`` / ``&`` which prompt_toolkit's HTML
    # parser would otherwise treat as markup — escape it. The indicator is
    # our own trusted HTML; don't escape it.
    safe_preview = html_escape(preview)
    safe_rule_explanation = html_escape(rule_explanation)
    footer = f'<style fg="ansibrightblack">{html_escape(_FOOTER_SHORTCUTS)}</style>'
    return (
        f"{safe_preview}\n"
        f"{indicator}\n"
        f"{safe_rule_explanation}\n\n"
        f"{footer}"
    )


def _compose_option_two(
    tool: BaseTool, args: dict[str, Any],
) -> tuple[str, Rule, Literal["project", "session"]]:
    """Return ``(label, rule, scope)`` for prompt option 2 (spec §8.2)."""
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


def make_cli_asker() -> PermissionAsker:
    """Return a ``PermissionAsker`` backed by ``prompt_toolkit``'s radiolist.

    Stateless — no closures beyond the prompt library's own internals. The
    returned coroutine is safe to reuse across calls.
    """

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

        journal.write(
            "permission_asked",
            tool=tool.name,
            args_preview=preview,
            rule_hint=option_two_rule.to_string(),
        )

        default_cursor = 3 if (tool.metadata or {}).get("is_destructive") else 1

        # HTML formatted text gives us inline colour hints (destructive red,
        # read-only/safe dim) without pulling a full Style object. Windows
        # terminals without ANSI just render the tag as plain text.
        title = HTML(f"<b>{html_escape(tool.name)}</b>  <i>{html_escape(tag)}</i>")
        text = HTML(_build_dialog_text(tool, args))

        try:
            app = radiolist_dialog(
                title=title,
                text=text,
                values=[
                    (1, "Yes"),
                    (2, option_two_label),
                    (3, "No"),
                ],
                default=default_cursor,
            )
            choice = await app.run_async()
        except (KeyboardInterrupt, SystemExit):
            # Ctrl+C / hard exit — let the upper loop handle cancellation.
            # Spec §8.3: "Ctrl+C → deny" is enforced by the caller treating
            # cancellation as deny; this layer must not swallow it into a
            # soft-deny response.
            raise
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
        # choice == 3 or None (Esc / cancelled)
        journal.write("permission_answered", tool=tool.name, choice="deny")
        return AskerResponse(choice="deny")

    return _ask


def print_bypass_banner(console: Console) -> None:
    """Print the bypass-mode startup warning (spec §8.5).

    Task 9 wires the call site in ``aura/cli/__main__.py``; this module only
    owns the rendering.
    """
    console.print(
        "[bold red]⚠  PERMISSION CHECKS DISABLED — "
        "all tool calls will run without asking[/bold red]",
    )
