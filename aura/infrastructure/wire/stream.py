"""Stream an agent run as a sequence of Aura wire events."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from aura.infrastructure.wire.event_dto import WireEvent
from aura.infrastructure.wire.wire import agent_state_to_wire, event_to_wire


async def stream_agent_wire(
    agent: Any,
    prompt: str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[WireEvent]:
    """Run ``agent.astream`` and yield Aura wire events.

    Ordering contract: any pre-turn coordination buffer drains FIRST so
    cross-turn leftovers attribute to the turn they arrived in; per-event
    coordination drains follow each astream yield; ``aura_state`` is the
    final wire event of the turn, emitted after every event astream
    produced (including post-final auto-compact events).
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
        return list(cast("list[WireEvent]", drain()))
    pending = cast("tuple[WireEvent, ...]", getattr(agent, "pending_protocol_events", ()))
    return list(pending)


def _is_skipped_no_op_compact(payload: WireEvent) -> bool:
    # Skipped microcompact / auto checks fire 1-3x per turn with no
    # state delta — useful in the journal, pure noise on the wire.
    if payload.get("event") != "compact_event":
        return False
    if payload.get("outcome") != "skipped":
        return False
    return payload.get("tokens_before") == payload.get("tokens_after")
