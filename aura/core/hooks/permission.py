"""Permission layer — PreToolHook factory that gates every tool call.

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §5, extended 2026-04-21
with the ``plan`` and ``accept_edits`` modes.

Decision order (short-circuits at first match):

1. ``mode == "bypass"`` → allow (loud journal).
2. Safety — fires for any tool operating on a single file via
   ``args["path"]``. The direction flag (``is_write`` ≙
   ``metadata["is_destructive"]``) picks which protected list to consult:
   the writes list for destructive tools, the (narrower) reads list for
   everyone else. Tools without a path arg (``bash``) fall through; the
   user's rule grant is the backstop there. Runs BEFORE the plan /
   accept_edits mode branches so no mode can short-circuit safety.
3. ``mode == "plan"`` → dry-run deny with ``plan_mode_blocked``. Reports
   what the agent *would* have done without side effects.
4. ``mode == "accept_edits"`` AND tool name ∈ ``{read_file, write_file,
   edit_file}`` → auto-allow with ``mode_accept_edits``. Any other tool
   falls through to the rule / ask path.
5. Rule match — project rules first, session rules second. The project
   RuleSet carries built-in defaults from
   ``aura.core.permissions.defaults.DEFAULT_ALLOW_RULES`` composed with
   user rules at startup, so ``read_file`` / ``grep`` / ``glob`` hit
   ``rule_allow`` without a prompt while still passing through safety.
6. Ask — delegate to the CLI asker; install the returned rule if
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

# Tools whose invocation ``accept_edits`` mode auto-allows (once safety has
# cleared). Deliberately a tight closed set: the user opts into "yes, keep
# editing files" without also opting into shell commands, web fetches, or
# custom tools. Matches claude-code's "accept edits" mode behavior.
_ACCEPT_EDITS_TOOLS: frozenset[str] = frozenset({"read_file", "write_file", "edit_file"})

# Plan mode allow-list. Pure read / information-gathering tools the planner
# needs in order to draft a useful plan (read the code, search it, fetch
# docs, inspect running subagents). These fall through to the normal
# rule / ask pipeline so safety + default-allow rules still apply; plan
# mode only removes the blanket dry-run for them.
_PLAN_MODE_READ_TOOLS: frozenset[str] = frozenset(
    {
        "read_file", "grep", "glob",
        "web_fetch", "web_search",
        "task_get", "task_list",
    }
)

# Tools that control plan mode itself MUST be reachable from plan mode,
# otherwise the LLM could never exit. Bootstrap safety — these bypass
# the plan-mode dry-run branch entirely and fall through to the usual
# rule/ask pipeline. Kept separate from the read-allow list because the
# intent is different: "mode control" vs "read-only browsing".
_PLAN_MODE_EXEMPT_TOOLS: frozenset[str] = frozenset(
    {"enter_plan_mode", "exit_plan_mode"}
)

# Cap for the plan-mode "would have called" error string. Tool args can be
# multi-kilobyte (write_file.content, bash.command). We truncate so a
# dry-run error doesn't flood the model's context window, but the full
# args are still emitted to the journal for the operator's audit trail.
_PLAN_PREVIEW_MAX_CHARS = 200


@dataclass(frozen=True)
class AskerResponse:
    """What the CLI asker returns to the hook.

    Invariants (enforced in ``__post_init__`` so bugs surface at
    construction, not when the hook tries to install a None rule):

    - ``choice == "always"`` iff ``rule is not None``. An ``always`` with
      no rule would be meaningless (nothing to install); an ``accept`` or
      ``deny`` with a rule would be a category error (the hook only
      installs on ``always``).

    ``feedback`` carries the free-text note the user typed via the
    widget's Tab-to-amend UX. Empty string means "no feedback" (the
    common case); when non-empty it gets threaded into the journal
    event AND — for denials — appended to the model-facing error
    string so the LLM can read the user's reason for saying no.
    """

    choice: Literal["accept", "always", "deny"]
    scope: Literal["project", "session"] = "session"
    rule: Rule | None = None
    feedback: str = ""

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


def _plan_args_preview(tool: BaseTool, args: dict[str, Any]) -> str:
    """One-line preview of a planned call's args for plan-mode dry-run output.

    Prefers the tool's own ``args_preview`` (same function used by the
    CLI prompt renderer — keeps the two surfaces consistent) and falls
    back to a compact ``key=repr(value)`` list when a tool doesn't ship
    one. Truncated to ``_PLAN_PREVIEW_MAX_CHARS`` with a single ellipsis
    character so very large args (a bash command embedding a script,
    a write_file body) don't explode the model-facing error.
    """
    preview_fn = (tool.metadata or {}).get("args_preview")
    if callable(preview_fn):
        try:
            out = preview_fn(args)
        except Exception:  # noqa: BLE001 — preview must never break the deny path
            out = None
        if isinstance(out, str) and out:
            return _truncate(out, _PLAN_PREVIEW_MAX_CHARS)
    # Fallback — compact "k=repr(v)" join. Uses ``repr`` so strings show
    # their quoting (easy to spot the boundary in an error message).
    joined = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _truncate(joined, _PLAN_PREVIEW_MAX_CHARS)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _plan_error_message(tool: BaseTool, args: dict[str, Any]) -> str:
    """Compose the model-facing error string for ``plan_mode_blocked``.

    Format: ``plan mode: would have called {tool}({args_preview})`` — it
    tells the model to stop executing AND gives the planner enough context
    to iterate on the plan without re-deriving the args.
    """
    return (
        f"plan mode: would have called {tool.name}({_plan_args_preview(tool, args)})"
    )


def _deny_message(decision: Decision, *, feedback: str = "") -> str:
    """Model-facing error string for denied decisions. Concise + actionable.

    When the user provided free-text feedback via the widget's
    Tab-to-amend path, append it to ``user_deny`` so the LLM can read
    *why* the user said no. ``safety_blocked`` is generated mechanically
    and has no user-authored reason — feedback is ignored there.

    Note: ``plan_mode_blocked`` is handled out-of-band by the hook so the
    message can embed the tool name + args preview without threading them
    through Decision.
    """
    match decision.reason:
        case "safety_blocked":
            return "denied: protected path (safety policy)"
        case "user_deny":
            if feedback:
                return f"denied: user — note: {feedback}"
            return "denied: user"
        case _:  # pragma: no cover — unreachable by construction
            return "denied"


def _safety_target(args: dict[str, Any]) -> str | None:
    """Return the filesystem target for safety checks, or None if not applicable.

    Convention (spec §6, scoped to MVP): safety fires for any tool whose
    args carry a non-empty ``path`` string. The *direction* — writes vs
    reads — is picked by the caller from the tool's ``is_destructive``
    metadata flag. Tools without a path arg (``bash``, ``web_fetch``)
    have no arg-to-path mapping, so they aren't safety-gated here — the
    user's rule grant for that tool is the backstop.
    """
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

    Explicit ``match`` exhausts every ``AskerResponse.scope`` literal so
    that adding a new scope value to the Literal (e.g. ``"user"``) forces
    an editor/mypy failure here rather than silently dropping the rule
    into session.
    """
    assert response.rule is not None  # invariant enforced by AskerResponse
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
                # Consent stands; disk failure doesn't revoke it.
                session.add(response.rule)
        case "session":
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
        decision, feedback = await _decide(
            tool=tool,
            args=args,
            asker=asker,
            session=session,
            rules=rules,
            project_root=project_root,
            mode=mode,
            safety=safety,
        )
        # Journal the decision. ``feedback`` is only ever non-empty when
        # the user typed a note via the widget's Tab-to-amend flow;
        # otherwise the field is omitted so existing audit scrapers
        # don't see a spurious empty string on every event.
        extra: dict[str, Any] = {}
        if feedback:
            extra["feedback"] = feedback
        journal.write(
            "permission_decision",
            tool=tool.name,
            reason=decision.reason,
            rule=decision.rule.to_string() if decision.rule is not None else None,
            mode=mode,
            target=decision.target,
            **extra,
        )
        # Transient per-call stash: loop._plan_tool_calls reads this back
        # IMMEDIATELY after run_pre_tool returns (same tool call, same
        # event-loop turn, no await in between) and pops it. Single-slot
        # by design — tool plans are built sequentially so there's no race.
        state.custom["_aura_pending_decision"] = decision
        if decision.allow:
            return None
        if decision.reason == "plan_mode_blocked":
            # The plan-mode error needs tool name + args preview, which
            # aren't on Decision. Compose here where they're in scope.
            return ToolResult(
                ok=False, error=_plan_error_message(tool, args),
            )
        return ToolResult(ok=False, error=_deny_message(decision, feedback=feedback))

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
) -> tuple[Decision, str]:
    """Decide whether to allow + return (decision, feedback).

    ``feedback`` is the free-text note the user typed via the widget's
    Tab-to-amend flow, or ``""`` when unavailable / not applicable. Only
    the user-interactive branches (accept / always / deny via the asker)
    can produce feedback; rule-allow / safety-blocked / mode-bypass
    short-circuits always return ``""``.
    """
    # 1. Bypass mode — loud and first. Note: bypass deliberately does NOT
    # run through safety here; the safety floor for bypass lives at the
    # tool level (bash_safety_hook) and for path tools is enforced by
    # the OS (write to /etc/passwd would fail anyway). This matches the
    # existing spec contract and the pre-existing test
    # ``test_bypass_mode_short_circuits_even_on_protected_path``.
    if mode == "bypass":
        journal.write("permission_bypass", tool=tool.name)
        return Decision(allow=True, reason="mode_bypass"), ""

    # 2. Safety — any tool with a resolvable path arg, write-or-read
    # direction chosen by the tool's is_destructive flag. Runs BEFORE
    # plan/accept_edits so no mode can sneak past the protected list.
    target = _safety_target(args)
    if target is not None:
        is_write = bool((tool.metadata or {}).get("is_destructive", False))
        if is_protected(target, safety, is_write=is_write):
            return Decision(allow=False, reason="safety_blocked", target=target), ""

    # 3. Plan mode — dry-run deny for every safety-cleared tool call,
    # EXCEPT the read-tool allow-list (reads are how the planner does its
    # job) and the plan-mode control tools (the LLM needs to be able to
    # exit plan mode; blocking them would be a deadlock). Both exemptions
    # fall through to the normal rule/ask pipeline below, so safety +
    # default-allow rules still apply. Deliberately placed AFTER safety
    # so a plan-mode session still surfaces safety_blocked decisions for
    # protected paths (the user sees the real reason, not a generic
    # "plan would have called").
    if (
        mode == "plan"
        and tool.name not in _PLAN_MODE_EXEMPT_TOOLS
        and tool.name not in _PLAN_MODE_READ_TOOLS
    ):
        return Decision(allow=False, reason="plan_mode_blocked"), ""

    # 4. accept_edits mode — auto-allow the edit-family tools. Any other
    # tool (bash, web_fetch, custom) falls through to the rule/ask path,
    # so side-effecting calls still require explicit consent.
    if mode == "accept_edits" and tool.name in _ACCEPT_EDITS_TOOLS:
        return Decision(allow=True, reason="mode_accept_edits"), ""

    # 5. Rule match — project first (includes built-in defaults), session second.
    matched = rules.matches(tool.name, args, tool)
    if matched is None:
        matched = session.matches(tool.name, args, tool)
    if matched is not None:
        return Decision(allow=True, reason="rule_allow", rule=matched), ""

    # 6. Ask.
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
            return (
                Decision(allow=True, reason="user_always", rule=response.rule),
                feedback,
            )
