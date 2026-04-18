"""Loop-level state shared across turns within one AgentLoop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopState:
    turn_count: int = 0
    total_tokens_used: int = 0
    custom: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.turn_count = 0
        self.total_tokens_used = 0
        self.custom.clear()
