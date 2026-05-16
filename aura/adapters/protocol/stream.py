"""Streaming helpers and SSE framing for Aura protocol adapters."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from aura.adapters.protocol.agui import AguiAdapter
from aura.adapters.protocol.wire import agent_state_to_wire, event_to_wire
from aura.domain.protocol.events import WireEvent
from aura.schemas.events import Final


def encode_sse(
    data: str,
    *,
    event: str | None = None,
    id: str | None = None,
    retry: int | None = None,
) -> str:
    """Encode a single SSE frame."""
    lines: list[str] = []
    if id is not None:
        lines.append(f"id: {id}")
    if event is not None:
        lines.append(f"event: {event}")
    if retry is not None:
        lines.append(f"retry: {retry}")
    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def encode_json_sse(
    payload: WireEvent | dict[str, Any],
    *,
    event: str | None = None,
    id: str | None = None,
    retry: int | None = None,
) -> str:
    """Encode a compact JSON payload as an SSE frame."""
    data = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return encode_sse(data, event=event, id=id, retry=retry)


async def stream_agent_wire(
    agent: Any,
    prompt: str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[WireEvent]:
    """Run ``agent.astream`` and yield Aura wire events."""
    turn_start = clock()
    async for event in agent.astream(prompt):
        payload = event_to_wire(event)
        yield payload
        if isinstance(event, Final):
            yield agent_state_to_wire(agent, clock() - turn_start)


async def stream_agent_agui(
    agent: Any,
    prompt: str,
    *,
    adapter: AguiAdapter | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[dict[str, Any]]:
    """Run ``agent.astream`` and yield AG-UI-style events."""
    agui = adapter or AguiAdapter()
    for event in agui.start_run():
        yield event
    turn_start = clock()
    try:
        async for agent_event in agent.astream(prompt):
            wire_event = event_to_wire(agent_event)
            if isinstance(agent_event, Final):
                state_event = agent_state_to_wire(agent, clock() - turn_start)
                for event in agui.convert(state_event):
                    yield event
            for event in agui.convert(wire_event):
                yield event
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for event in agui.error(message):
            yield event


async def stream_agent_wire_sse(
    agent: Any,
    prompt: str,
    *,
    event: str = "aura",
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    """Run ``agent.astream`` and yield Aura wire events as SSE frames."""
    try:
        async for payload in stream_agent_wire(agent, prompt, clock=clock):
            yield encode_json_sse(payload, event=event)
    except Exception as exc:
        yield encode_json_sse(
            {
                "event": "error",
                "message": f"{type(exc).__name__}: {exc}",
            },
            event=event,
        )


async def stream_agent_agui_sse(
    agent: Any,
    prompt: str,
    *,
    event: str = "agui",
    adapter: AguiAdapter | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    """Run ``agent.astream`` and yield AG-UI-style events as SSE frames."""
    async for payload in stream_agent_agui(
        agent,
        prompt,
        adapter=adapter,
        clock=clock,
    ):
        yield encode_json_sse(payload, event=event)
