"""``restrict-tools`` lease — strict per-turn whitelist blocking every undeclared tool."""

from __future__ import annotations

from aura.application.loop_state import LoopState
from aura.domain.skill import Skill
from aura.domain.state_values import SkillRestrictLease

# Mode controls and the permission-asker UX the lease must never block.
_INTERNAL_EXEMPT_TOOLS: frozenset[str] = frozenset(
    {
        "ask_user_question",
        "enter_plan_mode",
        "exit_plan_mode",
    }
)


def install_restrict_lease(skill: Skill, state: LoopState) -> None:
    """Install a turn-scoped restrict lease; no-op when empty, idempotent within a turn."""
    if not skill.restrict_tools:
        return
    leases = state.slots.skill_restrict_leases
    new_entry = SkillRestrictLease(
        install_turn=state.turn_count,
        tools=frozenset(skill.restrict_tools),
    )
    if new_entry in leases:
        return
    leases.append(new_entry)


def _active_leases(state: LoopState) -> list[SkillRestrictLease]:
    """Return non-expired leases (active for their install turn only), pruning in place."""
    raw = state.slots.skill_restrict_leases
    if not raw:
        return []
    current = state.turn_count
    active = [e for e in raw if e.install_turn >= current]
    if len(active) != len(raw):
        raw[:] = active
    return active


def tool_allowed_by_lease(state: LoopState, tool_name: str) -> bool:
    """True iff ``tool_name`` may run under the current lease set."""
    active = _active_leases(state)
    if not active:
        return True
    if tool_name in _INTERNAL_EXEMPT_TOOLS:
        return True
    return any(tool_name in entry.tools for entry in active)


def has_active_lease(state: LoopState) -> bool:
    """True iff at least one non-expired lease is installed."""
    return bool(_active_leases(state))
