"""CLI permission asker — dispatch router over per-tool-type widgets.

Matches claude-code's permission UX: different specialized dialogs for
different tool families. Three specialized widgets live in sibling
modules — this file routes to them:

  * :mod:`aura.cli.permission_bash`    — ``bash`` / ``bash_background``
    (syntax-highlighted command + dangerous-arg banner)
  * :mod:`aura.cli.permission_write`   — ``write_file`` / ``edit_file``
    (diff preview for edits, size + head preview for writes)
  * :mod:`aura.cli.permission_generic` — everything else (the original
    inline 4-option pt.Application widget)

The router doesn't re-implement pt / rule derivation / journal plumbing
— that stays here. What it does:

1. Inspect tool + args.
2. Pick the correct specialized widget.
3. Hand it the common pieces (option_two_label, default_choice, tag).
4. Collect ``(choice, feedback)`` and map onto ``AskerResponse``.

Spec alignment: ``docs/specs/2026-04-19-aura-permission.md`` §8.1–§8.5.

One job: present the choice, capture the answer. The asker does NOT
decide (that's the hook), does NOT persist (that's the store), and does
NOT emit domain events beyond the two I/O-boundary journal lines
(``permission_asked`` / ``permission_answered``).
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from rich.console import Console

from aura.cli.permission_bash import run_bash_permission
from aura.cli.permission_generic import (
    _TOOL_VERB,
    _build_explanation,
    _tool_title,
    _tool_verb,
    run_generic_permission,
)
from aura.cli.permission_write import run_write_permission
from aura.core.hooks.permission import AskerResponse, PermissionAsker
from aura.core.permissions.rule import Rule
from aura.core.permissions.rule_hint import derive_rule_hint
from aura.core.persistence import journal

# Re-export the preview cap so external tests / callers that poked the
# old module-level constant still find it.
_PREVIEW_MAX_CHARS = 200


# Tool-name sets that pick the specialized widget. Closed sets —
# adding a new bash-family / write-family tool requires an explicit
# entry, which is what we want (unknown tools go through the generic
# widget, not a best-guess specialized one).
_BASH_TOOLS: frozenset[str] = frozenset({"bash", "bash_background"})
_WRITE_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file"})


def _tag(tool: BaseTool) -> Literal["destructive", "read-only", "safe"]:
    """Classification tag. Kept for journal/telemetry; NOT rendered in
    the widget header — claude-code's design uses a clean title and
    lets the command preview carry the risk signal."""
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
    tag: Literal["destructive", "read-only", "safe"],
    option_two_label: str,
    default_choice: int,
    args: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> tuple[int | None, str]:
    """Dispatch to the specialized widget based on ``tool.name``.

    Kept as a single-entry function (matching the pre-refactor shape)
    so existing tests that monkey-patch this symbol keep working. All
    it does now is route + forward; the real rendering lives in the
    three ``permission_*`` sibling modules.
    """
    _args = args or {}
    if tool.name in _BASH_TOOLS:
        command = str(_args.get("command", "") or "")
        return await run_bash_permission(
            tool=tool,
            command=command,
            args_preview=preview,
            args=_args,
            tag=tag,
            option_two_label=option_two_label,
            default_choice=default_choice,
            timeout=timeout,
        )
    if tool.name in _WRITE_TOOLS:
        return await run_write_permission(
            tool=tool,
            args=_args,
            tag=tag,
            option_two_label=option_two_label,
            default_choice=default_choice,
            timeout=timeout,
        )
    return await run_generic_permission(
        tool=tool,
        preview=preview,
        tag=tag,
        option_two_label=option_two_label,
        default_choice=default_choice,
        args=_args,
        timeout=timeout,
    )


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
    """Return a ``PermissionAsker`` backed by the dispatch router.

    ``console`` — optional rich Console for tests (StringIO-backed for
    capture). Production path creates a fresh one.

    ``timeout`` — seconds to wait for the user to respond before
    treating the non-response as a denial (fail-safe for unattended /
    headless sessions). ``None`` preserves legacy "wait forever"
    behavior.
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
        # press Tab); non-empty flows on to the hook.
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


__all__ = [
    "AskerResponse",
    "PermissionAsker",
    "_TOOL_VERB",
    "_build_explanation",
    "_compose_option_two",
    "_pick_choice_interactive",
    "_preview",
    "_render_decision_audit_line",
    "_tag",
    "_tool_title",
    "_tool_verb",
    "make_cli_asker",
    "print_bypass_banner",
]
