"""Loop-level state shared across turns within one AgentLoop."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoopState:
    turn_count: int = 0
    total_tokens_used: int = 0
    custom: dict[str, object] = field(default_factory=dict)
