"""Permission layer — PreToolHook factory that gates every tool call.

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §5.

Decision order (short-circuits at first match):

1. ``mode == "bypass"`` → allow (loud journal).
2. Safety — only for destructive tools operating on a single file via
   ``args["path"]``. Convention: ``write_file`` / ``edit_file`` — tools
   that map to one concrete filesystem target. Other destructive tools
   (``bash``) have no arg-to-path mapping, so safety doesn't fire; the
   user's rule grant is the backstop there.
3. ``is_read_only`` → allow.
4. Rule match — project rules first, session rules second.
5. Ask — delegate to the CLI asker; install the returned rule if
   ``always``.

Layer boundaries (enforced by construction, not convention):

- **Hook** decides: assembles inputs into a ``Decision``, emits journal,
  returns ``ToolResult`` or ``None``.
- **Asker** asks: presents the choice to the user, returns
  ``AskerResponse`` (``accept`` / ``always`` / ``deny``).
- **Store** saves: atomic write to ``.aura/settings.json``.

Each does exactly one thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from langchain_core.tools import BaseTool

from aura.core.hooks import PreToolHook
from aura.core.permissions.decision import Decision
from aura.core.permissions.mode import DEFAULT_MODE, Mode
from aura.core.permissions.rule import Rule
from aura.core.permissions.safety import DEFAULT_SAFETY, SafetyPolicy, is_protected
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.core.permissions.store import PermissionStoreError, save_rule
from aura.core.persistence import journal
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult

# Shared empty immutable RuleSet — safe as a default (frozen, no mutable state).
_EMPTY_RULESET = RuleSet()


@dataclass(frozen=True)
class AskerResponse:
    """What the CLI asker returns to the hook.

    Invariants (enforced in ``__post_init__`` so bugs surface at
    construction, not when the hook tries to install a None rule):

    - ``choice == "always"`` iff ``rule is not None``. An ``always`` with
      no rule would be meaningless (nothing to install); an ``accept`` or
      ``deny`` with a rule would be a category error (the hook only
      installs on ``always``).
    """

    choice: Literal["accept", "always", "deny"]
    scope: Literal["project", "session"] = "session"
    rule: Rule | None = None

    def __post_init__(self) -> None:
        if self.choice == "always" and self.rule is None:
            raise ValueError("choice='always' requires a rule to install")
        if self.choice != "always" and self.rule is not None:
            raise ValueError(
                f"choice={self.choice!r} must not carry a rule "
                f"(only 'always' installs a rule)"
            )


@runtime_checkable
class PermissionAsker(Protocol):
    """I/O boundary — asks the user for a single permission choice.

    Keyword-only signature: callers cannot accidentally rely on
    positional order, and adding future kwargs (e.g. a safety hint) is
    non-breaking.
    """

    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,
    ) -> AskerResponse: ...


def _deny_message(decision: Decision) -> str:
    """Model-facing error string for denied decisions. Concise + actionable."""
    match decision.reason:
        case "safety_blocked":
            return "denied: write to protected path (safety policy)"
        case "user_deny":
            return "denied: user"
        case _:  # pragma: no cover — unreachable by construction
            return "denied"


def _safety_target(tool: BaseTool, args: dict[str, Any]) -> str | None:
    """Return the filesystem target for safety checks, or None if not applicable.

    Convention (spec §6, scoped to MVP): safety only fires on destructive
    tools that operate on a single file via ``args["path"]``. This covers
    ``write_file`` and ``edit_file``. Other destructive tools (``bash``)
    have no arg-to-path mapping, so they aren't safety-gated here — the
    user's rule grant for that tool is the backstop.
    """
    metadata = tool.metadata or {}
    if not metadata.get("is_destructive"):
        return None
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

    On project-scope save failure: journal, degrade to session scope. The
    user's consent stands — disk failure doesn't revoke it.
    """
    assert response.rule is not None  # invariant enforced by AskerResponse
    if response.scope == "project":
        try:
            save_rule(project_root, response.rule, scope="project")
            return
        except PermissionStoreError as exc:
            journal.write(
                "permission_save_failed",
                tool=tool_name,
                rule=response.rule.to_string(),
                detail=str(exc),
            )
            # Fall through to session-scope degradation.
    session.add(response.rule)


def make_permission_hook(
    *,
    asker: PermissionAsker,
    session: SessionRuleSet,
    rules: RuleSet = _EMPTY_RULESET,
    project_root: Path,
    mode: Mode = DEFAULT_MODE,
    safety: SafetyPolicy = DEFAULT_SAFETY,
) -> PreToolHook:
    async def _hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **_: Any,
    ) -> ToolResult | None:
        decision = await _decide(
            tool=tool,
            args=args,
            asker=asker,
            session=session,
            rules=rules,
            project_root=project_root,
            mode=mode,
            safety=safety,
        )
        journal.write(
            "permission_decision",
            tool=tool.name,
            reason=decision.reason,
            rule=decision.rule.to_string() if decision.rule is not None else None,
            mode=mode,
        )
        # Transient per-call stash: loop._plan_tool_calls reads this back
        # IMMEDIATELY after run_pre_tool returns (same tool call, same
        # event-loop turn, no await in between) and pops it. Single-slot
        # by design — tool plans are built sequentially so there's no race.
        state.custom["_aura_pending_decision"] = decision
        if decision.allow:
            return None
        return ToolResult(ok=False, error=_deny_message(decision))

    return _hook


async def _decide(
    *,
    tool: BaseTool,
    args: dict[str, Any],
    asker: PermissionAsker,
    session: SessionRuleSet,
    rules: RuleSet,
    project_root: Path,
    mode: Mode,
    safety: SafetyPolicy,
) -> Decision:
    # 1. Bypass mode — loud and first.
    if mode == "bypass":
        journal.write("permission_bypass", tool=tool.name)
        return Decision(allow=True, reason="mode_bypass")

    # 2. Safety — only for destructive tools with a resolvable path arg.
    target = _safety_target(tool, args)
    if target is not None and is_protected(target, safety):
        return Decision(allow=False, reason="safety_blocked")

    # 3. Read-only auto-allow.
    if (tool.metadata or {}).get("is_read_only", False):
        return Decision(allow=True, reason="read_only")

    # 4. Rule match — project first, session second.
    matched = rules.matches(tool.name, args, tool)
    if matched is None:
        matched = session.matches(tool.name, args, tool)
    if matched is not None:
        return Decision(allow=True, reason="rule_allow", rule=matched)

    # 5. Ask.
    rule_hint = Rule(tool=tool.name, content=None)
    try:
        response = await asker(tool=tool, args=args, rule_hint=rule_hint)
    except Exception as exc:  # noqa: BLE001 — asker bugs must not crash the loop
        # Note: we deliberately don't catch BaseException. KeyboardInterrupt,
        # asyncio.CancelledError, and SystemExit must propagate so Ctrl+C and
        # task cancellation reach Agent.astream's cancellation handling.
        journal.write(
            "permission_asker_failed",
            tool=tool.name,
            detail=f"{type(exc).__name__}: {exc}",
        )
        return Decision(allow=False, reason="user_deny")

    match response.choice:
        case "accept":
            return Decision(allow=True, reason="user_accept")
        case "deny":
            return Decision(allow=False, reason="user_deny")
        case "always":
            _install_always(
                response,
                session=session,
                project_root=project_root,
                tool_name=tool.name,
            )
            return Decision(allow=True, reason="user_always", rule=response.rule)
