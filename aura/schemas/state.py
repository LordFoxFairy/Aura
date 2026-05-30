"""Cumulative loop state carried across turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

from aura.domain.state_values import (
    BuddyState,
    PermissionKey,
    SkillRestrictLease,
    TokenStats,
)
from aura.domain.todos import TodoItem

if TYPE_CHECKING:
    from aura.application.permission.decision import Decision
    from aura.domain.permission.denials import PermissionDenial as Denial
    from aura.domain.skill import Skill
else:
    # Runtime aliases so pydantic field-type introspection resolves without
    # dragging the real modules into aura.schemas at import time.
    Denial = Any
    Decision = Any
    Skill = Any


PermissionDedupEntry: TypeAlias = "tuple[Decision, str]"


@dataclass(frozen=True)
class LoopSlots:
    token_stats: TokenStats = field(default_factory=TokenStats)
    turn_denials: list[Denial] = field(default_factory=list)
    todos: list[TodoItem] = field(default_factory=list)
    perm_dedup_cache: dict[PermissionKey, PermissionDedupEntry] = field(default_factory=dict)
    preserved_invoked_skills: list[Skill] = field(default_factory=list)
    invoked_skills: list[Skill] = field(default_factory=list)
    consecutive_compact_failures: int = 0
    active_team: str | None = None
    buddy: BuddyState = field(default_factory=BuddyState)
    skill_restrict_leases: list[SkillRestrictLease] = field(default_factory=list)


@dataclass
class LoopState:
    turn_count: int = 0
    total_tokens_used: int = 0
    slots: LoopSlots = field(default_factory=LoopSlots)

    def reset(self) -> None:
        # In-place: AgentLoop holds the same ref; slots are owned by writers.
        self.turn_count = 0
        self.total_tokens_used = 0
