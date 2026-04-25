"""V14 ``restrict-tools`` lease — strict whitelist scoped to a single turn.

Companion to v0.13's :func:`aura.core.skills.command.install_skill_allow_rules`.
The two layers stack:

- ``allowed_tools`` (permissive): auto-allow declared tools for the rest of
  the session. Lives in :class:`aura.core.permissions.session.SessionRuleSet`
  as concrete :class:`aura.core.permissions.rule.Rule` entries; cleared by
  ``/clear``.
- ``restrict_tools`` (restrictive, this module): block every tool NOT in
  the union of declared sets, scoped to the model response chain that
  processed the skill body. Implemented as a turn-count sentinel + per-skill
  whitelist stored in :attr:`aura.schemas.state.LoopState.custom`.

Lease shape — why a transient slot, not :class:`SessionRuleSet`:

- Restriction is fundamentally turn-scoped (claude-code's contract:
  "while the skill body is being processed"). SessionRuleSet is
  session-scoped — wrong granularity, would need a parallel expiry
  mechanism anyway.
- A single skill can install multiple leases over a session (one per
  invocation); each must independently expire when its triggering turn
  ends. A list of ``(turn, frozenset)`` entries on state.custom captures
  this naturally; ``SessionRuleSet`` rules are flat with no per-rule TTL.
- Internal/asker tools (``ask_user_question``) need to bypass the
  restriction. Adding "but-not-this-tool" exemptions on top of a Rule
  matcher would muddy the rule path; a separate lease kept the
  decision-tree simple.

Contract:

- Empty ``skill.restrict_tools`` → no lease installed (parity with
  ``allowed_tools=[]``).
- Multiple active leases on the same turn → union (tool allowed if ANY
  active skill declares it).
- Lease expires when ``state.turn_count`` advances past the install-time
  sentinel. Expired entries are pruned lazily on the next consult call.
- Internal tools bypass the lease entirely — the model-facing tool surface
  is what restrict_tools governs, not infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass

from aura.core.skills.types import Skill
from aura.schemas.state import LoopState

# state.custom slot. Exported so the permission hook (reader) and skill
# install path (writer) can reference one source of truth.
RESTRICT_LEASE_KEY = "_skill_restrict_lease"

# Tools the restrict-tools lease never blocks. ``ask_user_question`` is
# internal infrastructure — the agent itself uses it for clarification
# prompts and the permission asker is built on the same widget. Allowing
# the restrict lease to block it would break the user-experience layer.
# Plan-mode controls are similarly mode-controlling tools; restrict is a
# session-time whitelist orthogonal to plan-mode dynamics, so we keep
# those reachable. Kept tight (closed set) so a future "internal" tool
# doesn't silently inherit exemption.
_INTERNAL_EXEMPT_TOOLS: frozenset[str] = frozenset({
    "ask_user_question",
    "enter_plan_mode",
    "exit_plan_mode",
})


@dataclass(frozen=True)
class _RestrictEntry:
    """One installed lease — captures install turn + the declared whitelist."""

    install_turn: int
    tools: frozenset[str]


def install_restrict_lease(skill: Skill, state: LoopState) -> None:
    """Install a turn-scoped restrict lease for ``skill`` on ``state``.

    No-op when ``skill.restrict_tools`` is empty (matches the
    ``allowed_tools=[]`` "no special config" semantic).

    Idempotent within the same turn for the same skill: re-invoking the
    skill in turn N does not stack a second lease entry that expires at a
    different sentinel; the install-turn is simply re-asserted.
    """
    if not skill.restrict_tools:
        return
    leases: list[_RestrictEntry] = state.custom.setdefault(
        RESTRICT_LEASE_KEY, [],
    )
    install_turn = state.turn_count
    new_entry = _RestrictEntry(
        install_turn=install_turn, tools=frozenset(skill.restrict_tools),
    )
    # De-dup: drop any existing entry with same (install_turn, tools).
    if new_entry in leases:
        return
    leases.append(new_entry)


def _active_leases(state: LoopState) -> list[_RestrictEntry]:
    """Return non-expired leases, pruning expired ones in place.

    Lease expiry rule: a lease installed on turn N is active for turn N
    only — once ``state.turn_count`` advances past N (i.e. >= N+1), the
    lease is considered expired and removed. This matches the spec
    contract "scope by turn count: install at invocation, expire when
    state.turn_count advances past a recorded sentinel".
    """
    raw = state.custom.get(RESTRICT_LEASE_KEY)
    if not isinstance(raw, list) or not raw:
        return []
    current = state.turn_count
    active: list[_RestrictEntry] = []
    expired_any = False
    for entry in raw:
        if not isinstance(entry, _RestrictEntry):
            # Defensive: foreign payload (someone else used the slot).
            # Skip rather than crash.
            expired_any = True
            continue
        if entry.install_turn >= current:
            active.append(entry)
        else:
            expired_any = True
    if expired_any:
        # Prune in place so future calls don't re-walk dead entries.
        state.custom[RESTRICT_LEASE_KEY] = active
    return active


def tool_allowed_by_lease(state: LoopState, tool_name: str) -> bool:
    """True iff ``tool_name`` may run under the current lease set.

    Decision:

    - No active leases → True (no restriction in effect).
    - Tool in :data:`_INTERNAL_EXEMPT_TOOLS` → True (infrastructure
      bypass).
    - Otherwise → tool must appear in the union of declared whitelists
      across all active leases.
    """
    active = _active_leases(state)
    if not active:
        return True
    if tool_name in _INTERNAL_EXEMPT_TOOLS:
        return True
    union: set[str] = set()
    for entry in active:
        union.update(entry.tools)
    return tool_name in union


def has_active_lease(state: LoopState) -> bool:
    """True iff at least one non-expired lease is installed.

    Used by the permission hook to decide whether to consult the lease at
    all — when none exists the hook bypasses the whole restrict branch
    instead of paying the dict-lookup cost on every tool call.
    """
    return bool(_active_leases(state))


def expire_lease(state: LoopState) -> None:
    """Force-clear all leases (test hook + ``/clear`` integration point)."""
    state.custom.pop(RESTRICT_LEASE_KEY, None)
