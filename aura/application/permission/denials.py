"""Structured permission-denial records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class PermissionDenial:
    tool_name: str
    tool_use_id: str
    # Shallow copy of args at decision time so downstream mutation can't rewrite the audit record.
    tool_input: dict[str, Any]
    reason: str
    target: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
