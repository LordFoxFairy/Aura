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

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.tools import BaseTool
from rich.console import Console

from aura.cli.permission_bash import run_bash_permission
from aura.cli.permission_generic import (
    _TOOL_VERB,
    _build_explanation,
    _run_widget,
    _tool_title,
    _tool_verb,
    run_generic_permission,
)
from aura.cli.permission_write import run_write_permission
from aura.core.hooks.permission import AskerResponse, PermissionAsker
from aura.core.permissions.rule import Rule
from aura.core.permissions.rule_hint import derive_rule_hint
from aura.core.persistence import journal
from aura.schemas.permissions import AskerPrompt
from aura.schemas.permissions import AskerResponse as AskerResponseV2
from aura.schemas.tool_meta_access import meta_dict

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
    metadata = meta_dict(tool)
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
    preview_fn = meta_dict(tool).get("args_preview")
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


# ---------------------------------------------------------------------------
# Phase 5 Task 4 — new ``AskerPrompt`` / ``AskerResponse`` shape.
#
# The v2 asker takes a fully-rendered :class:`AskerPrompt` (display strings
# + ``request_id``) and returns the four-state
# :class:`aura.schemas.permissions.AskerResponse` (yes / yes-always / no /
# no-always). Same pt.Application driver as legacy — UI behavior matches
# the legacy generic widget (3 visible buttons today; ``no-always`` is
# reserved for the gate to emit programmatically once Stage 9 lands).
#
# Legacy callers (``make_permission_hook``) keep using ``make_cli_asker``
# above; the two coexist until Phase 5 collapses the asker boundary.
# ---------------------------------------------------------------------------


# Map of internal picker int → new-shape choice string. ``None``
# (Ctrl+C / Esc cancel) resolves to ``"no"`` so the loop never hangs on
# an unresolved prompt — same fail-safe contract as the legacy asker's
# ``deny`` path.
_INT_TO_NEW_CHOICE: dict[int | None, Literal["yes", "yes-always", "no"]] = {
    1: "yes",
    2: "yes-always",
    3: "no",
    None: "no",
}


def _v2_header_frags(prompt: AskerPrompt) -> list[tuple[str, str]]:
    """Build the widget header fragments from an :class:`AskerPrompt`.

    The v2 asker has no :class:`BaseTool` to consult (the new Protocol
    intentionally decouples the asker from the registry — IPC + subagent
    askers have only the prompt strings). We render the same generic
    layout as :func:`run_generic_permission` but sourced from the
    prompt's strings: ``tool`` becomes the title, ``args_preview`` is
    the body, and the per-tool verb falls back to a generic phrase.
    """
    title = prompt.tool.replace("_", " ")
    if " " in title:
        head, *rest = title.split(" ")
        title = " ".join([head.capitalize(), *rest])
    else:
        title = f"{title.capitalize()} command"
    verb = _TOOL_VERB.get(prompt.tool, "")
    header: list[tuple[str, str]] = [
        ("bold", f"  {title}\n"),
        ("", "\n"),
    ]
    if prompt.args_preview:
        header.append(("", f"    {prompt.args_preview}\n"))
    if verb:
        header.append(("class:dim", f"  {verb}\n"))
    header.append(("", "\n"))
    return header


def _v2_explanation_frags(prompt: AskerPrompt) -> list[tuple[str, str]]:
    """Ctrl+E panel for the v2 (string-only) widget.

    The legacy explanation pulls from ``tool.description`` + arg dict;
    the v2 prompt only carries display strings, so we emit a slimmed
    panel that names the tool, surfaces the rule_hint, and flags
    destructiveness — enough context for an operator deciding whether
    to approve, without inventing fake docstring content.
    """
    risk = (
        "⚠ This tool can modify or delete data."
        if prompt.is_destructive
        else "● Standard tool — see preview above for what will run."
    )
    frags: list[tuple[str, str]] = [
        ("class:dim bold", "  ┌ Explanation\n"),
        ("class:dim bold", "  │ Tool:\n"),
        ("class:dim", f"  │     {prompt.tool}\n"),
        ("class:dim bold", "  │ Preview:\n"),
        ("class:dim", f"  │     {prompt.args_preview or '(no preview)'}\n"),
        ("class:dim bold", "  │ Rule hint:\n"),
        ("class:dim", f"  │     {prompt.rule_hint or '(none)'}\n"),
        ("class:dim bold", "  │ Risk:\n"),
        ("class:dim", f"  │     {risk}\n"),
        ("class:dim bold", "  └\n"),
    ]
    return frags


def _option_two_label_from_prompt(prompt: AskerPrompt) -> str:
    """Compose the "yes, and don't ask again" label from a prompt.

    Mirrors :func:`_compose_option_two`'s wording. The v2 prompt carries
    a precomputed ``rule_hint`` string; we just embed it. Empty
    ``rule_hint`` falls back to a tool-wide session label (matching the
    legacy "no matcher" path).
    """
    if prompt.rule_hint:
        return (
            f"Yes, and don't ask again for `{prompt.rule_hint}` in this project"
        )
    return f"Yes, and don't ask again for `{prompt.tool}` this session"


def make_cli_asker_v2(
    console: Console | None = None,
    *,
    timeout: float | None = None,
) -> Callable[[AskerPrompt], Awaitable[AskerResponseV2]]:
    """Return a v2 CLI asker that consumes :class:`AskerPrompt`.

    Spec: ``docs/superpowers/specs/2026-05-10-aura-phase-5-permissions.md``
    §6. The returned callable is the new-shape asker the
    :class:`PermissionGate` (Phase 5 Task 8) will wire up. Coexists with
    the legacy :func:`make_cli_asker` until the gate fully replaces the
    permission_hook factory; both share the same pt.Application driver,
    so UI behavior is consistent.

    Mapping from picker int → new choice:

    - 1 (Yes)             → ``"yes"``
    - 2 (Yes, always)     → ``"yes-always"``
    - 3 (No)              → ``"no"``
    - None (Esc / Ctrl+C) → ``"no"``    (fail-safe deny)

    ``"no-always"`` is not yet emitted by this widget (CLI today shows 3
    buttons matching claude-code's three-button dialog); the gate will
    synthesize it programmatically when an operator chooses to install a
    deny rule. The shape is reserved here so the union stays exhaustive.
    """
    _ = console or Console()  # parity with legacy factory; reserved for audit-line

    async def _ask(prompt: AskerPrompt) -> AskerResponseV2:
        default_choice = 3 if prompt.is_destructive else 1
        option_two_label = _option_two_label_from_prompt(prompt)

        journal.write(
            "permission_asked",
            tool=prompt.tool,
            args_preview=prompt.args_preview,
            rule_hint=prompt.rule_hint,
            request_id=prompt.request_id,
        )

        try:
            choice, _feedback = await _run_widget(
                header_frags=_v2_header_frags(prompt),
                option_two_label=option_two_label,
                default_choice=default_choice,
                explanation_frags=_v2_explanation_frags(prompt),
                timeout=timeout,
            )
        except TimeoutError:
            journal.write(
                "permission_prompt_timeout",
                tool=prompt.tool,
                timeout_sec=timeout,
                request_id=prompt.request_id,
            )
            journal.write(
                "permission_answered",
                tool=prompt.tool,
                choice="no",
                reason="timeout",
                request_id=prompt.request_id,
            )
            return AskerResponseV2(choice="no", request_id=prompt.request_id)
        except (KeyboardInterrupt, SystemExit):
            journal.write(
                "permission_answered",
                tool=prompt.tool,
                choice="no",
                request_id=prompt.request_id,
            )
            return AskerResponseV2(choice="no", request_id=prompt.request_id)
        except Exception as exc:  # noqa: BLE001 — no-TTY / pt failures
            journal.write(
                "permission_prompt_unavailable",
                tool=prompt.tool,
                detail=repr(exc),
                request_id=prompt.request_id,
            )
            return AskerResponseV2(choice="no", request_id=prompt.request_id)

        new_choice = _INT_TO_NEW_CHOICE[choice]
        journal.write(
            "permission_answered",
            tool=prompt.tool,
            choice=new_choice,
            request_id=prompt.request_id,
        )
        return AskerResponseV2(choice=new_choice, request_id=prompt.request_id)

    return _ask


def legacy_asker_from_v2(
    v2_asker: Callable[[AskerPrompt], Awaitable[AskerResponseV2]],
) -> PermissionAsker:
    """Adapter — expose a v2 asker as the legacy :class:`PermissionAsker`.

    The legacy permission hook (``make_permission_hook``) calls askers
    with ``(tool, args, rule_hint)`` kwargs and expects the legacy
    :class:`AskerResponse` (``accept`` / ``always`` / ``deny`` + rule +
    scope + feedback). Phase 5 transitions everything to v2 in one shot
    (Task 8); this adapter is the bridge that lets the legacy hook keep
    working when wired to a v2-shape asker mid-migration.

    The adapter:

    1. Composes an :class:`AskerPrompt` from the legacy kwargs (using
       :func:`_compose_option_two` to derive ``rule_hint``).
    2. Generates a fresh ``request_id`` per call (legacy callers don't
       carry one — IPC correlation only matters for the desktop asker).
    3. Calls the v2 asker.
    4. Maps the four-state :class:`AskerResponseV2` back onto the
       three-state legacy :class:`AskerResponse`:

       - ``yes``        → ``accept``
       - ``yes-always`` → ``always`` + rule from local derivation
       - ``no``         → ``deny``
       - ``no-always``  → ``deny`` (no legacy equivalent for "deny rule";
         the gate will install the deny rule out-of-band when it owns
         the flow — for now treat it as a one-shot deny so legacy
         callers don't see an unrepresentable choice).

    Feedback is dropped on the v2 boundary (the new shape doesn't carry
    free-text); legacy callers that wired feedback through this adapter
    therefore see ``feedback=""``. This is a known regression of the
    transitional adapter, not a permanent loss — Task 8 retires both
    the adapter and the legacy AskerResponse together.
    """
    import uuid

    async def _legacy(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,  # noqa: ARG001 — derived locally from tool/args
    ) -> AskerResponse:
        _option_two_label, option_two_rule, option_two_scope = (
            _compose_option_two(tool, args)
        )
        prompt = AskerPrompt(
            tool=tool.name,
            args_preview=_preview(tool, args),
            rule_hint=option_two_rule.to_string(),
            is_destructive=_tag(tool) == "destructive",
            request_id=str(uuid.uuid4()),
        )
        v2_resp = await v2_asker(prompt)
        match v2_resp.choice:
            case "yes":
                return AskerResponse(choice="accept")
            case "yes-always":
                return AskerResponse(
                    choice="always",
                    scope=option_two_scope,
                    rule=option_two_rule,
                )
            case "no" | "no-always":
                return AskerResponse(choice="deny")

    return _legacy


__all__ = [
    "AskerPrompt",
    "AskerResponse",
    "AskerResponseV2",
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
    "legacy_asker_from_v2",
    "make_cli_asker",
    "make_cli_asker_v2",
    "print_bypass_banner",
]
