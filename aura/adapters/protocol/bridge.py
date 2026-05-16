"""Protocol-layer bridges that own transport event sequencing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from aura.adapters.protocol.agui import AguiAdapter
from aura.adapters.protocol.wire import agent_state_to_wire, event_to_wire
from aura.domain.protocol.events import WireEvent


class AguiEventBridge:
    """Bridge internal Aura events into ordered AG-UI payloads.

    The bridge owns transport sequencing concerns such as run start, final
    state snapshots, and terminal error conversion so call sites do not need
    to scatter ``AguiAdapter.convert(...)`` decisions. Future subagent/team
    event families can enter through this bridge as additional ``emit_*``
    methods without changing stream orchestration.
    """

    def __init__(
        self,
        *,
        adapter: AguiAdapter | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._adapter = adapter or AguiAdapter()
        self._clock = clock

    @property
    def adapter(self) -> AguiAdapter:
        return self._adapter

    def start(self) -> list[dict[str, Any]]:
        """Emit the transport's run-start envelope."""
        return self._adapter.start_run()

    def emit(
        self,
        event: Any,
        *,
        agent: Any | None = None,
        turn_started_at: float | None = None,
    ) -> list[dict[str, Any]]:
        """Convert one internal Aura event into ordered AG-UI payloads."""
        return self.emit_wire(
            event_to_wire(event),
            agent=agent,
            turn_started_at=turn_started_at,
        )

    def emit_wire(
        self,
        wire_event: WireEvent,
        *,
        agent: Any | None = None,
        turn_started_at: float | None = None,
    ) -> list[dict[str, Any]]:
        """Convert one Aura wire event into ordered AG-UI payloads."""
        out: list[dict[str, Any]] = []
        if wire_event.get("event") == "final" and agent is not None and turn_started_at is not None:
            out.extend(
                self._adapter.convert(
                    agent_state_to_wire(agent, self._clock() - turn_started_at),
                ),
            )
        out.extend(self._adapter.convert(wire_event))
        return out

    def error(self, exc: Exception | str) -> list[dict[str, Any]]:
        """Convert a terminal error into AG-UI payloads."""
        message = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        return self._adapter.error(message)


__all__ = ["AguiEventBridge"]
