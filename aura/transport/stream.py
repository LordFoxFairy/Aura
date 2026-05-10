"""Streaming helpers that adapt an Agent run to external event contracts."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from aura.schemas.events import Final
from aura.transport.agui import AguiAdapter
from aura.transport.sse import encode_json_sse
from aura.transport.wire import agent_state_to_wire, event_to_wire


async def stream_agent_wire(
    agent: Any,
    prompt: str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[dict[str, Any]]:
    """Run ``agent.astream`` and yield Aura wire events.

    The final state snapshot is part of the external transport contract, not
    the core loop contract, so it lives here instead of in ``Agent.astream``.
    """
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
