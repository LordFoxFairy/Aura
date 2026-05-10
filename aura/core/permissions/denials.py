"""Structured permission-denial records (Workstream G5).

Surface for SDK / plugin / UI consumers who want to inspect which tool
calls the permission layer denied within the most recent turn without
re-parsing ``events.jsonl``. The hook (``aura/core/hooks/permission.py``)
appends one :class:`PermissionDenial` per non-allow decision to a
per-turn sink; the Agent exposes it via
:meth:`aura.core.agent.Agent.last_turn_denials`.

Contract invariants:

- :class:`PermissionDenial` is frozen — once appended the record must not
  mutate. Consumers see a stable snapshot of the decision at deny time.
- ``reason`` values mirror :data:`aura.core.permissions.decision.DecisionReason`
  deny variants (``safety_blocked`` / ``user_deny`` / ``plan_mode_blocked``)
  — we do NOT invent a new taxonomy. If a new deny reason lands on
  ``Decision``, plumb it through here by adding it to the DecisionReason
  literal; mypy will force this file into sync.
- ``tool_input`` is a shallow copy of the tool's args at decision time so
  a downstream mutation of ``args`` after the hook returns cannot
  retroactively rewrite the audit record.

Phase 1 Task 4 migrated the per-turn denials sink off the legacy
``LoopState.custom["_aura_denials_sink"]`` scratchpad onto the typed
``LoopState.slots.turn_denials`` slot. The permission hook (writer),
:class:`aura.core.loop.AgentLoop` (turn-start clear), and
:class:`aura.core.agent.Agent` (read view via ``last_turn_denials``)
all coordinate through that single typed list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class PermissionDenial:
    """One non-allow permission decision, captured at decision time.

    Fields:

    - ``tool_name``: the LangChain BaseTool.name the model tried to call.
    - ``tool_use_id``: the LangChain ``tool_call.id`` — matches the
      ``ToolMessage.tool_call_id`` that Aura will (or would have) written
      back for this call. Empty string when the hook was invoked without
      a tool-call id (unit-test path where callers build a fake hook
      invocation without going through the Loop).
    - ``tool_input``: shallow copy of the args dict at decision time.
    - ``reason``: the :class:`aura.core.permissions.decision.Decision`
      reason string for the deny branch (``safety_blocked`` /
      ``user_deny`` / ``plan_mode_blocked``).
    - ``target``: mirror of ``Decision.target`` — populated only for
      ``safety_blocked`` today; reserved for future path-carrying deny
      reasons. None otherwise.
    - ``timestamp``: UTC wall-clock at decision time.
    """

    tool_name: str
    tool_use_id: str
    tool_input: dict[str, Any]
    reason: str
    target: str | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )
