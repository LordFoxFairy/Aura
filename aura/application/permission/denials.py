"""Structured permission-denial records (Workstream G5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class PermissionDenial:
    """One non-allow permission decision, captured at decision time.

    - ``tool_use_id``: matches LangChain's ``tool_call.id``; empty when
      the hook ran outside a real loop (unit-test path).
    - ``tool_input``: shallow copy of args (downstream mutation cannot
      retroactively rewrite the audit record).
    - ``reason``: a deny variant of
      :data:`aura.application.permission.decision.DecisionReason`.
    - ``target``: mirror of :attr:`Decision.target` — populated only for
      ``safety_blocked`` today.
    """

    tool_name: str
    tool_use_id: str
    tool_input: dict[str, Any]
    reason: str
    target: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
