"""Canonical protocol adapters for Aura external surfaces."""

from aura.adapters.protocol.agui import AguiAdapter
from aura.adapters.protocol.bridge import AguiEventBridge
from aura.adapters.protocol.stream import (
    stream_agent_agui,
    stream_agent_agui_sse,
    stream_agent_wire,
    stream_agent_wire_sse,
)
from aura.adapters.protocol.wire import (
    agent_state_to_wire,
    compact_event_to_wire,
    event_to_wire,
    permission_request_to_wire,
)

__all__ = [
    "AguiAdapter",
    "AguiEventBridge",
    "agent_state_to_wire",
    "compact_event_to_wire",
    "event_to_wire",
    "permission_request_to_wire",
    "stream_agent_agui",
    "stream_agent_agui_sse",
    "stream_agent_wire",
    "stream_agent_wire_sse",
]
