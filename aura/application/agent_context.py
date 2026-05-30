"""AgentContext — per-turn capability bag injected into run_agent."""

from __future__ import annotations

from dataclasses import dataclass

from aura.application.loop import AgentLoop
from aura.domain.abort import AbortController


@dataclass(frozen=True)
class AgentContext:
    loop: AgentLoop
    abort: AbortController
