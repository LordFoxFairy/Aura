"""Canonical Aura protocol event payload contracts."""

from __future__ import annotations

from typing import Any, TypedDict


class WireEvent(TypedDict, total=False):
    event: str
    id: str
    name: str
    input: dict[str, Any]
    stream: str
    chunk: str
    content: dict[str, Any]
    tool: str
    text: str
    message: str
    reason: str
    args: dict[str, Any]
    rule_hint: str
    is_destructive: bool
    trigger: str
    tokens_before: int
    tokens_after: int
    outcome: str
    duration_ms: float
    model: str
    mode: str
    cwd: str
    tokens: dict[str, int]
    pinned: int
    window: int
    last_turn_seconds: float
    type: str


__all__ = ["WireEvent"]
