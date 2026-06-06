"""Stream an agent run as a sequence of Aura wire events."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, runtime_checkable

from aura.infrastructure.wire.events import WireEvent
from aura.infrastructure.wire.serialize import agent_state_to_wire, event_to_wire


@runtime_checkable
class _DrainsProtocolEvents(Protocol):
    def drain_protocol_events(self) -> list[WireEvent]: ...


@runtime_checkable
class _HasPendingProtocolEvents(Protocol):
    @property
    def pending_protocol_events(self) -> tuple[WireEvent, ...]: ...


async def stream_agent_wire(
    agent: Any,
    prompt: str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[WireEvent]:
    """Run ``agent.astream`` and yield Aura wire events.

    Invariant: pre-turn coordination drains first, per-event coordination
    drains after each astream yield, ``aura_state`` is always the final
    event of the turn.
    """
    turn_start = clock()
    for payload in _drain_coordination_events(agent):
        yield payload
    async for event in agent.astream(prompt):
        for payload in _drain_coordination_events(agent):
            yield payload
        wire_event = event_to_wire(event)
        if _is_skipped_no_op_compact(wire_event):
            continue
        yield wire_event
    for payload in _drain_coordination_events(agent):
        yield payload
    yield agent_state_to_wire(agent, clock() - turn_start)


def _drain_coordination_events(agent: Any) -> list[WireEvent]:
    if isinstance(agent, _DrainsProtocolEvents):
        return list(agent.drain_protocol_events())
    if isinstance(agent, _HasPendingProtocolEvents):
        return list(agent.pending_protocol_events)
    return []


def _is_skipped_no_op_compact(payload: WireEvent) -> bool:
    # Skipped no-delta compacts: journal-useful, wire-noise.
    if payload.get("event") != "compact_event":
        return False
    if payload.get("outcome") != "skipped":
        return False
    return payload.get("tokens_before") == payload.get("tokens_after")
