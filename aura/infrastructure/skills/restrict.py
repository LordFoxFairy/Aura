"""V14 ``restrict-tools`` lease — strict whitelist scoped to a single turn.

Companion to :func:`install_skill_allow_rules` in ``command.py``. The two
layers stack: ``allowed_tools`` auto-allows declared tools for the session
(``SessionRuleSet``); ``restrict_tools`` blocks every tool NOT in the union
of declared sets for the response chain processing the skill body.
"""

from __future__ import annotations

from aura.domain.skill import Skill
from aura.domain.state_values import SkillRestrictLease
from aura.schemas.state import LoopState

# Internal tools the restrict-tools lease never blocks: ``ask_user_question``
# powers the permission asker UX; enter/exit_plan_mode are mode controls.
_INTERNAL_EXEMPT_TOOLS: frozenset[str] = frozenset({
    "ask_user_question",
    "enter_plan_mode",
    "exit_plan_mode",
})


def install_restrict_lease(skill: Skill, state: LoopState) -> None:
    """Install a turn-scoped restrict lease for ``skill`` on ``state``.

    No-op when ``skill.restrict_tools`` is empty. Idempotent within the
    same turn — re-invoking the same skill does not stack duplicate
    entries.
    """
    if not skill.restrict_tools:
        return
    leases = state.slots.skill_restrict_leases
    new_entry = SkillRestrictLease(
        install_turn=state.turn_count, tools=frozenset(skill.restrict_tools),
    )
    if new_entry in leases:
        return
    leases.append(new_entry)


def _active_leases(state: LoopState) -> list[SkillRestrictLease]:
    """Return non-expired leases, pruning expired ones in place.

    A lease installed on turn N is active for turn N only; once
    ``state.turn_count`` advances past N the lease is dropped.
    """
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
