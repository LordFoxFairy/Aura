"""Stream an agent run as a sequence of Aura wire events."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from aura.infrastructure.wire.event_dto import WireEvent
from aura.infrastructure.wire.wire import agent_state_to_wire, event_to_wire
from aura.schemas.events import Final


async def stream_agent_wire(
    agent: Any,
    prompt: str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[WireEvent]:
    """Run ``agent.astream`` and yield Aura wire events."""
    turn_start = clock()
    for payload in _drain_coordination_events(agent):
        yield payload
    async for event in agent.astream(prompt):
        for payload in _drain_coordination_events(agent):
            yield payload
        yield event_to_wire(event)
        if isinstance(event, Final):
            yield agent_state_to_wire(agent, clock() - turn_start)
        for pending in _drain_coordination_events(agent):
            yield pending


def encode_sse(payload: WireEvent, *, event: str = "aura") -> str:
    """Encode one wire event as a single SSE frame."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_agent_wire_sse(
    agent: Any,
    prompt: str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    """Run ``agent.astream`` and yield encoded SSE frames."""
    async for wire_event in stream_agent_wire(agent, prompt, clock=clock):
        yield encode_sse(wire_event)


def _drain_coordination_events(agent: Any) -> list[WireEvent]:
    drain = getattr(agent, "drain_protocol_events", None)
    if callable(drain):
        return list(drain())
    return list(getattr(agent, "pending_protocol_events", ()))
