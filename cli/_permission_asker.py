"""CLI ``PermissionAsker`` — translates tool prompts into form questions.

Renders permission prompts through the unified :func:`cli.forms.render_form`
widget so the permission asker, the user-question asker, and any future form
share one keymap and one style sheet. Per-tool-family option labels follow
claude-code: bash gets a command-prefix install, write gets a directory
install, every other tool gets a tool-wide install.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from rich.console import Console

from aura.application.permission.asker import AskerResponse, PermissionAsker
from aura.application.permission.rule_hint import derive_rule_hint
from aura.domain.permission.rule import Rule
from aura.infrastructure.persistence import journal
from aura.tools.ask_user import FormOptionDict, FormQuestionDict
from cli.forms.widget import FormCancelled, render_form

_Scope = Literal["project", "session"]

_BASH_TOOLS: frozenset[str] = frozenset({"bash", "bash_background"})
_WRITE_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file"})

_PERMISSION_QUESTION = "Allow this tool call?"
_FEEDBACK_QUESTION = "Feedback (optional)"
_FEEDBACK_PLACEHOLDER = "Press Enter to skip, or type a note for the model…"

# Stable labels for choice mapping; UI strings come from per-family option lists.
_ALLOW_ONCE = "Allow once"
_DENY = "Deny"


def _bash_options(command: str) -> list[FormOptionDict]:
    head = command.split(maxsplit=1)[0] if command.strip() else ""
    prefix = head or command
    return [
        {"label": _ALLOW_ONCE, "description": "Run this command this time."},
        {
            "label": "Allow always for this command",
            "description": f"Install session rule for exactly `{command}`.",
        },
        {
            "label": "Allow always for the prefix",
            "description": f"Install project rule for any `{prefix} …` command.",
        },
        {"label": _DENY, "description": "Block this call; the LLM sees the deny."},
    ]


def _write_options(path: str) -> list[FormOptionDict]:
    parent = path.rsplit("/", 1)[0] if "/" in path else "."
    return [
        {"label": _ALLOW_ONCE, "description": "Write this file this time."},
        {
            "label": "Allow always for this path",
            "description": f"Install project rule for `{path}`.",
        },
        {
            "label": "Allow always for this dir",
            "description": f"Install project rule for `{parent}/*`.",
        },
        {"label": _DENY, "description": "Block this write; the LLM sees the deny."},
    ]


def _generic_options() -> list[FormOptionDict]:
    return [
        {"label": _ALLOW_ONCE, "description": "Run this tool call this time."},
        {
            "label": "Allow always",
            "description": "Install a rule so this tool runs without asking.",
        },
        {"label": _DENY, "description": "Block this call; the LLM sees the deny."},
    ]


def _build_questions(
    tool: BaseTool, args: dict[str, Any],
) -> list[FormQuestionDict]:
    """Return the form questions to render for this tool call."""
    if tool.name in _BASH_TOOLS:
        header = "Allow bash?"
        options = _bash_options(str(args.get("command", "") or ""))
    elif tool.name in _WRITE_TOOLS:
        header = "Allow write?"
        options = _write_options(str(args.get("path", "") or ""))
    else:
        header = "Allow tool?"
        options = _generic_options()
    return [
        {
            "question": _PERMISSION_QUESTION,
            "header": header,
            "multi_select": False,
            "options": options,
        },
        {
            "question": _FEEDBACK_QUESTION,
            "header": header,
            "multi_select": False,
            "free_text_placeholder": _FEEDBACK_PLACEHOLDER,
        },
    ]


def _translate(
    label: str,
    *,
    tool: BaseTool,
    args: dict[str, Any],
    feedback: str,
) -> AskerResponse:
    """Map a chosen option label to an :class:`AskerResponse`.

    Bash "prefix" and write "dir" choices fall back to a tool-wide session rule
    when the matcher can't derive a precise pattern (e.g. empty command).
    """
    if label == _ALLOW_ONCE:
        return AskerResponse(choice="accept", feedback=feedback)
    if label == _DENY:
        return AskerResponse(choice="deny", feedback=feedback)
    if label in {"Allow always", "Allow always for this path"}:
        derived = derive_rule_hint(tool, args)
        rule = derived or Rule(tool=tool.name, content=None)
        scope: _Scope = "project" if derived is not None else "session"
        return AskerResponse(
            choice="always", scope=scope, rule=rule, feedback=feedback,
        )
    if label == "Allow always for this command":
        rule = Rule(tool=tool.name, content=str(args.get("command", "") or "") or None)
        return AskerResponse(
            choice="always", scope="session", rule=rule, feedback=feedback,
        )
    if label == "Allow always for the prefix":
        command = str(args.get("command", "") or "")
        head = command.split(maxsplit=1)[0] if command.strip() else ""
        rule = Rule(tool=tool.name, content=head or None)
        prefix_scope: _Scope = "project" if head else "session"
        return AskerResponse(
            choice="always", scope=prefix_scope, rule=rule, feedback=feedback,
        )
    if label == "Allow always for this dir":
        path = str(args.get("path", "") or "")
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        rule = Rule(tool=tool.name, content=f"{parent}/*" if parent else None)
        dir_scope: _Scope = "project" if parent else "session"
        return AskerResponse(
            choice="always", scope=dir_scope, rule=rule, feedback=feedback,
        )
    # Unknown label is a programming error in this module — surface loudly.
    raise ValueError(f"unrecognised permission-asker label: {label!r}")


def make_cli_asker(
    console: Console | None = None,  # noqa: ARG001 — reserved for future audit-line printing
    *,
    timeout: float | None = None,  # noqa: ARG001 — render_form does not currently honor a timeout
) -> PermissionAsker:
    """Return a ``PermissionAsker`` backed by the unified form widget."""

    async def _ask(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,  # noqa: ARG001 — derivation is local
    ) -> AskerResponse:
        questions = _build_questions(tool, args)
        journal.write("permission_asked", tool=tool.name)
        try:
            answers = await render_form(questions)
        except FormCancelled:
            journal.write("permission_answered", tool=tool.name, choice="deny")
            return AskerResponse(choice="deny")
        label = answers.get(_PERMISSION_QUESTION, "")
        feedback = answers.get(_FEEDBACK_QUESTION, "")
        response = _translate(label, tool=tool, args=args, feedback=feedback)
        journal.write(
            "permission_answered", tool=tool.name, choice=response.choice,
        )
        return response

    return _ask


def print_bypass_banner(console: Console) -> None:
    """Print the bypass-mode startup warning (spec §8.5)."""
    console.print(
        "[bold red]⚠  PERMISSION CHECKS DISABLED — "
        "all tool calls will run without asking[/bold red]",
    )
