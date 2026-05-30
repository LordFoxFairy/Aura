"""Pure value types carried inside loop state."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

PermissionKey: TypeAlias = str


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
        if not isinstance(self.records, MappingProxyType):
            normalized = {
                path.expanduser().resolve(strict=False): record
                for path, record in self.records.items()
            }
            # object.__setattr__: only way to rebind a frozen dataclass field.
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
