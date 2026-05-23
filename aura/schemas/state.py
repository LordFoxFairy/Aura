"""Cumulative loop state carried across turns."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias

from aura.schemas.todos import TodoItem

if TYPE_CHECKING:
    from aura.application.permission.decision import Decision
    from aura.application.permission.denials import PermissionDenial as Denial
    from aura.infrastructure.skills.types import Skill
else:
    # Runtime fallbacks so pydantic field-type introspection on LoopState
    # resolves without dragging the real modules into aura.schemas.
    Denial = Any
    Decision = Any
    Skill = Any


PermissionKey: TypeAlias = str

PermissionDedupEntry: TypeAlias = "tuple[Decision, str]"


@dataclass(frozen=True)
class TokenStats:
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    last_cache_read_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    turn_count: int = 0


@dataclass(frozen=True)
class SkillRestrictLease:
    install_turn: int
    tools: frozenset[str]


@dataclass(frozen=True)
class BuddyState:
    mood: str = "idle"
    last_event_ts: float = 0.0
    had_recent_error: bool = False


@dataclass(frozen=True)
class ReadRecord:
    path: Path
    mtime_at_read: float
    size_at_read: int
    read_at_turn: int


@dataclass(frozen=True)
class ReadCarryover:
    records: Mapping[Path, ReadRecord]
    source_session_id: str | None
    generated_at_turn: int

    def __post_init__(self) -> None:
        # object.__setattr__ is the only way to rebind on a frozen dataclass.
        if not isinstance(self.records, MappingProxyType):
            normalized = {
                path.expanduser().resolve(strict=False): record
                for path, record in self.records.items()
            }
            object.__setattr__(self, "records", MappingProxyType(normalized))

    def is_fresh(self, path: Path) -> bool:
        resolved = path.expanduser().resolve(strict=False)
        record = self.records.get(resolved)
        if record is None:
            return False
        try:
            stat = os.stat(resolved)
        except OSError:
            return False
        return (
            stat.st_mtime == record.mtime_at_read
            and stat.st_size == record.size_at_read
        )


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
        # In-place mutation — AgentLoop holds the same reference; slots
        # are owned by individual writers and not reset here.
        self.turn_count = 0
        self.total_tokens_used = 0
