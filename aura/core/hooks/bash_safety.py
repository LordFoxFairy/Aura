"""Bash-command safety hook — Tier A hard floor.

Factory returns a ``PreToolHook`` that short-circuits bash-family
commands flagged by
:func:`aura.core.permissions.bash_safety.check_bash_safety`. The hook is
tool-name aware: it covers BOTH the blocking ``bash`` tool and the
fire-and-forget ``bash_background`` tool so the two share a single
source of truth for safety policy + audit surface.

Wiring: inserted at ``pre_tool[0]`` in ``Agent.__init__`` so it runs
BEFORE any caller-supplied permission hook. Permission is "is the user
OK with this?"; safety is "this class of command can't be safe
regardless of opinion". Rules cannot override — that's the whole point.

Bypass semantics
----------------

``mode == "bypass"`` is honored: under bypass the user has explicitly
opted into running *any* command the agent emits, so this hook becomes
a no-op. This matches the contract the permission layer enforces for
every other policy; giving the two bash tools a SPECIAL bypass rule
(safety still applies) would be a surprise and a cross-tool drift.
The OS remains the final floor for genuinely catastrophic commands
(e.g. writing to ``/etc/passwd`` still fails without root).

Audit surface
-------------

Every safety-blocked call emits TWO journal events:

1. ``bash_safety_blocked`` — legacy event, kept so existing audit
   scrapers continue to work. Carries the full command + reason + detail.
2. ``permission_decision`` with ``reason="safety_blocked"`` — same event
   the permission hook emits for path-based safety violations, so
   downstream consumers that filter on ``permission_decision`` see a
   uniform denial record regardless of which safety axis fired.

Additionally, the hook populates the per-turn denials sink
(``state.custom[DENIALS_SINK_KEY]``) with a :class:`PermissionDenial`,
so :meth:`aura.core.agent.Agent.last_turn_denials` surfaces
bash/bash_background blocks to SDK consumers without them having to
re-parse ``events.jsonl``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool

from aura.core.hooks import PRE_TOOL_PASSTHROUGH, PreToolHook, PreToolOutcome
from aura.core.permissions.bash_safety import check_bash_safety
from aura.core.permissions.decision import Decision
from aura.core.permissions.denials import DENIALS_SINK_KEY, PermissionDenial
from aura.core.permissions.mode import DEFAULT_MODE, Mode
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult

# Tools the hook covers. ``bash`` is the blocking tool; ``bash_background``
# is its fire-and-forget counterpart. Same safety policy for both — hence
# a single hook matching a frozenset of names rather than two separate
# closures. Adding a new bash-family tool (e.g. ``bash_pipe``) only
# requires extending this set.
_BASH_TOOL_NAMES: frozenset[str] = frozenset({"bash", "bash_background"})


def make_bash_safety_hook(
    *,
    mode_provider: Callable[[], Mode] | None = None,
    tool_names: frozenset[str] = _BASH_TOOL_NAMES,
) -> PreToolHook:
    """Build a pre-tool hook that enforces Tier A bash safety.

    Parameters
    ----------
    mode_provider:
        Zero-arg callable returning the live permission mode. When it
        resolves to ``"bypass"`` the hook short-circuits as a no-op,
        matching the permission-hook contract. ``None`` (the default)
        means "assume ``default`` mode" — used by tests that build the
        hook standalone without an Agent; there the bypass branch is
        irrelevant.
    tool_names:
        Set of ``tool.name`` values the hook fires on. Defaults to
        ``{"bash", "bash_background"}``; callers wanting to gate extra
        shell-like tools can extend the set.
    """
    if mode_provider is None:
        def _mode_provider() -> Mode:
            return DEFAULT_MODE
    else:
        _mode_provider = mode_provider

    async def _hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        tool_call_id: str = "",
        **_: Any,
    ) -> PreToolOutcome:
        if tool.name not in tool_names:
            return PRE_TOOL_PASSTHROUGH

        # Bypass mode — loud and first. Parity with the permission hook
        # which skips its own safety check under bypass: the user has
        # opted into "run anything", and re-applying safety here would
        # create a one-off drift between policy layers.
        if _mode_provider() == "bypass":
            return PRE_TOOL_PASSTHROUGH

        command = args.get("command")
        if not isinstance(command, str) or not command:
            # Tool's own arg-validation rejects; don't pre-empt.
            return PRE_TOOL_PASSTHROUGH

        violation = check_bash_safety(command)
        if violation is None:
            return PRE_TOOL_PASSTHROUGH

        # Lazy import — mirrors must_read_first.py — keeps the journal
        # dependency out of the module-load path.
        from aura.core.persistence import journal

        # 1. Legacy event kept so existing audit scrapers continue to
        #    work. Carries the full command (local audit trail only —
        #    ``events.jsonl`` is not shipped anywhere).
        journal.write(
            "bash_safety_blocked",
            reason=violation.reason,
            detail=violation.detail,
            command=command,
        )
        # 2. Parity event — same shape the permission hook emits for
        #    path-based safety blocks. Downstream consumers that filter
        #    on ``permission_decision`` see a uniform denial record.
        #    ``target`` stays ``None`` because bash safety is
        #    command-shape, not path-based; keeping the field in the
        #    event for consistency with the other ``safety_blocked``
        #    emits.
        journal.write(
            "permission_decision",
            tool=tool.name,
            reason="safety_blocked",
            rule=None,
            mode=_mode_provider(),
            target=None,
        )

        # G5: populate the per-turn denials sink so
        # ``Agent.last_turn_denials`` picks up bash/bash_background
        # safety blocks. Same shallow-copy defensive-snapshot semantics
        # the permission hook uses — a caller mutating ``args`` after
        # this hook returns cannot retroactively rewrite the audit.
        sink_obj = state.custom.get(DENIALS_SINK_KEY)
        if isinstance(sink_obj, list):
            sink_obj.append(
                PermissionDenial(
                    tool_name=tool.name,
                    tool_use_id=tool_call_id,
                    tool_input=dict(args),
                    reason="safety_blocked",
                    # ``target`` is reserved for path-based safety
                    # blocks; bash safety is command-shape, no single
                    # filesystem target to record.
                    target=None,
                )
            )

        return PreToolOutcome(
            short_circuit=ToolResult(
                ok=False,
                error=(
                    f"bash safety blocked: {violation.detail} "
                    f"(reason={violation.reason})"
                ),
            ),
            # G4 parity: surface the decision on the outcome so the Loop
            # can stamp it onto ``ToolStep.permission_decision`` and the
            # auditor emits a ``PermissionAudit`` event between
            # ``ToolCallStarted`` and ``ToolCallCompleted``.
            decision=Decision(allow=False, reason="safety_blocked"),
        )

    return _hook
