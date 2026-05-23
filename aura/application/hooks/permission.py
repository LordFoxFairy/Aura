"""Permission PreToolHook — gates every tool call (spec §5).

Decision order, short-circuits at first match:
  0. restrict-tools lease — whitelist override fires first.
  0.5 deny rules — bypass-immune.
  1. mode=bypass → allow.
  2. Safety — path tools w/ ``args["path"]``.
  3. mode=plan → dry-run deny unless tool is in plan-mode allow-list.
  4. mode=accept_edits → auto-allow edit-family tools.
  4.5 ask rules — force prompt.
  5. Rule match — project then session.
  6. Ask — per-turn dedup, then user.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from aura.application.hooks import PreToolHook
from aura.application.permission.asker import AskerResponse, PermissionAsker
from aura.application.permission.decision import Decision
from aura.application.permission.denials import PermissionDenial
from aura.application.permission.safety import is_protected
from aura.domain.permission.mode import DEFAULT_MODE, Mode
from aura.domain.permission.rule import Rule
from aura.domain.permission.safety import DEFAULT_SAFETY, SafetyPolicy
from aura.domain.permission.session import RuleSet, SessionRuleSet
from aura.infrastructure.permission_store import PermissionStoreError, save_rule
from aura.infrastructure.persistence import journal
from aura.infrastructure.skills.restrict import has_active_lease, tool_allowed_by_lease
from aura.schemas.permissions import Allow, Replace
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult, resolve_is_destructive
from aura.schemas.tool_meta_access import meta_dict

_EMPTY_RULESET = RuleSet()

_ACCEPT_EDITS_TOOLS: frozenset[str] = frozenset({"read_file", "write_file", "edit_file"})

_PLAN_MODE_READ_TOOLS: frozenset[str] = frozenset(
    {"read_file", "grep", "glob", "task_get", "task_list"}
)

_PLAN_MODE_EXEMPT_TOOLS: frozenset[str] = frozenset(
    {"enter_plan_mode", "exit_plan_mode"}
)

_PLAN_PREVIEW_MAX_CHARS = 200


def _dedup_key(tool_name: str, args: dict[str, Any]) -> str:
    return f"{tool_name}::{json.dumps(args, sort_keys=True, default=str)}"


def _plan_args_preview(tool: BaseTool, args: dict[str, Any]) -> str:
    preview_fn = meta_dict(tool).get("args_preview")
    if callable(preview_fn):
        try:
            out = preview_fn(args)
        except Exception:  # noqa: BLE001
            out = None
        if isinstance(out, str) and out:
            return _truncate(out, _PLAN_PREVIEW_MAX_CHARS)
    joined = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _truncate(joined, _PLAN_PREVIEW_MAX_CHARS)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _plan_error_message(tool: BaseTool, args: dict[str, Any]) -> str:
    return (
        f"plan mode: would have called {tool.name}({_plan_args_preview(tool, args)})"
    )


def _deny_message(decision: Decision, *, feedback: str = "") -> str:
    match decision.reason:
        case "safety_blocked":
            return "denied: protected path (safety policy)"
        case "restrict_tools_blocked":
            return (
                "denied: tool not in active skill's restrict-tools whitelist"
            )
        case "rule_deny":
            if decision.rule is not None:
                return f"denied: deny rule `{decision.rule.to_string()}`"
            return "denied: deny rule"
        case "user_deny":
            if feedback:
                return f"denied: user — note: {feedback}"
            return "denied: user"
        case _:
            return "denied"


def _safety_target(args: dict[str, Any]) -> str | None:
    path = args.get("path")
    if not isinstance(path, str) or not path:
        return None
    return path


def _install_always(
    response: AskerResponse,
    *,
    session: SessionRuleSet,
    project_root: Path,
    tool_name: str,
) -> None:
    """Install ``response.rule`` per ``response.scope``.

    Project-scope save failure: journal, degrade to session.
    """
    assert response.rule is not None
    match response.scope:
        case "project":
            try:
                save_rule(project_root, response.rule, scope="project")
            except PermissionStoreError as exc:
                journal.write(
                    "permission_save_failed",
                    tool=tool_name,
                    rule=response.rule.to_string(),
                    detail=str(exc),
                )
                session.add(response.rule)
        case "session":
            session.add(response.rule)


def make_permission_hook(
    *,
    asker: PermissionAsker,
    session: SessionRuleSet,
    rules: RuleSet = _EMPTY_RULESET,
    deny_rules: RuleSet = _EMPTY_RULESET,
    ask_rules: RuleSet = _EMPTY_RULESET,
    project_root: Path,
    mode: Mode | Callable[[], Mode] = DEFAULT_MODE,
    safety: SafetyPolicy = DEFAULT_SAFETY,
    disable_bypass: bool = False,
) -> PreToolHook:
    # mode is a Callable so Agent.set_mode mid-session is honored.
    if callable(mode):
        _raw_mode_provider: Callable[[], Mode] = mode
    else:
        _frozen_mode: Mode = mode
        def _raw_mode_provider() -> Mode:
            return _frozen_mode

    _bypass_clamp_warned: list[bool] = [False]

    def _mode_provider() -> Mode:
        live = _raw_mode_provider()
        if disable_bypass and live == "bypass":
            if not _bypass_clamp_warned[0]:
                journal.write(
                    "bypass_clamped",
                    reason="disable_bypass",
                    requested="bypass",
                    effective="default",
                )
                _bypass_clamp_warned[0] = True
            return "default"
        return live

    async def _hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        tool_call_id: str = "",
        ask_pending: bool = False,
        **_: Any,
    ) -> Allow | Replace:
        decision, feedback = await _decide(
            tool=tool,
            args=args,
            asker=asker,
            session=session,
            rules=rules,
            deny_rules=deny_rules,
            ask_rules=ask_rules,
            project_root=project_root,
            mode=_mode_provider(),
            safety=safety,
            state=state,
            ask_demote=ask_pending,
        )
        extra: dict[str, Any] = {}
        if feedback:
            extra["feedback"] = feedback
        journal.write(
            "permission_decision",
            tool=tool.name,
            reason=decision.reason,
            rule=decision.rule.to_string() if decision.rule is not None else None,
            rule_kind=decision.rule.kind if decision.rule is not None else None,
            mode=_mode_provider(),
            target=decision.target,
            **extra,
        )
        if decision.allow:
            return Allow(decision=decision)
        state.slots.turn_denials.append(
            PermissionDenial(
                tool_name=tool.name,
                tool_use_id=tool_call_id,
                tool_input=dict(args),
                reason=decision.reason,
                target=decision.target,
            )
        )
        if decision.reason == "plan_mode_blocked":
            short_circuit = ToolResult(
                ok=False, error=_plan_error_message(tool, args),
            )
        else:
            short_circuit = ToolResult(
                ok=False, error=_deny_message(decision, feedback=feedback),
            )
        return Replace(result=short_circuit, decision=decision)

    return _hook


async def _decide(
    *,
    tool: BaseTool,
    args: dict[str, Any],
    asker: PermissionAsker,
    session: SessionRuleSet,
    rules: RuleSet,
    deny_rules: RuleSet = _EMPTY_RULESET,
    ask_rules: RuleSet = _EMPTY_RULESET,
    project_root: Path,
    mode: Mode,
    safety: SafetyPolicy,
    state: LoopState,
    ask_demote: bool = False,
) -> tuple[Decision, str]:
    """Pick an outcome + return ``(decision, feedback)``.

    ``ask_demote`` reflects whether an upstream hook in the same
    pre_tool chain returned Ask; when true, auto-allow paths
    (mode_bypass, mode_accept_edits, rule_allow, dedup-cache) are
    demoted to the asker so the user still confirms (F-04-002).
    """
    if has_active_lease(state) and not tool_allowed_by_lease(state, tool.name):
        return Decision(allow=False, reason="restrict_tools_blocked"), ""

    deny_match = deny_rules.matches(tool.name, args, tool)
    if deny_match is not None:
        return Decision(allow=False, reason="rule_deny", rule=deny_match), ""

    if mode == "bypass" and not ask_demote:
        journal.write("permission_bypass", tool=tool.name)
        return Decision(allow=True, reason="mode_bypass"), ""

    target = _safety_target(args)
    if target is not None:
        is_write = resolve_is_destructive(meta_dict(tool), args)
        if is_protected(target, safety, is_write=is_write):
            return Decision(allow=False, reason="safety_blocked", target=target), ""

    if (
        mode == "plan"
        and tool.name not in _PLAN_MODE_EXEMPT_TOOLS
        and tool.name not in _PLAN_MODE_READ_TOOLS
    ):
        return Decision(allow=False, reason="plan_mode_blocked"), ""

    if (
        mode == "accept_edits"
        and tool.name in _ACCEPT_EDITS_TOOLS
        and not ask_demote
    ):
        return Decision(allow=True, reason="mode_accept_edits"), ""

    ask_match = ask_rules.matches(tool.name, args, tool)
    if ask_match is not None:
        journal.write(
            "permission_ask_rule_forced",
            tool=tool.name,
            rule=ask_match.to_string(),
        )
        rule_hint = Rule(tool=tool.name, content=None)
        try:
            response = await asker(tool=tool, args=args, rule_hint=rule_hint)
        except Exception as exc:  # noqa: BLE001
            journal.write(
                "permission_asker_failed",
                tool=tool.name,
                detail=f"{type(exc).__name__}: {exc}",
            )
            return Decision(allow=False, reason="user_deny"), ""
        feedback = response.feedback
        match response.choice:
            case "accept":
                return Decision(allow=True, reason="user_accept"), feedback
            case "deny":
                return Decision(allow=False, reason="user_deny"), feedback
            case "always":
                _install_always(
                    response,
                    session=session,
                    project_root=project_root,
                    tool_name=tool.name,
                )
                return Decision(
                    allow=True,
                    reason="user_always",
                    rule=response.rule,
                ), feedback

    if not ask_demote:
        matched = rules.matches(tool.name, args, tool)
        if matched is None:
            matched = session.matches(tool.name, args, tool)
        if matched is not None:
            return Decision(allow=True, reason="rule_allow", rule=matched), ""

    cache_key = _dedup_key(tool.name, args)
    cache = state.slots.perm_dedup_cache
    cached = cache.get(cache_key)
    if cached is not None and not ask_demote:
        cached_decision, cached_feedback = cached
        journal.write(
            "permission_dedup_hit",
            tool=tool.name,
            reason=cached_decision.reason,
        )
        return cached_decision, cached_feedback

    rule_hint = Rule(tool=tool.name, content=None)
    try:
        response = await asker(tool=tool, args=args, rule_hint=rule_hint)
    except Exception as exc:  # noqa: BLE001
        # Catches Exception only — KeyboardInterrupt / CancelledError propagate.
        journal.write(
            "permission_asker_failed",
            tool=tool.name,
            detail=f"{type(exc).__name__}: {exc}",
        )
        return Decision(allow=False, reason="user_deny"), ""

    feedback = response.feedback
    match response.choice:
        case "accept":
            decision = Decision(allow=True, reason="user_accept")
            cache[cache_key] = (decision, feedback)
            return decision, feedback
        case "deny":
            decision = Decision(allow=False, reason="user_deny")
            cache[cache_key] = (decision, feedback)
            return decision, feedback
        case "always":
            _install_always(
                response,
                session=session,
                project_root=project_root,
                tool_name=tool.name,
            )
            decision = Decision(
                allow=True, reason="user_always", rule=response.rule,
            )
            return decision, feedback
