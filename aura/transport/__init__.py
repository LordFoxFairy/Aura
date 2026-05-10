"""Transport adapters for Aura's external event surfaces.

The core runtime emits :mod:`aura.schemas.events` instances. This package owns
the stable JSON-facing contracts used by desktop, SSE, and AG-UI integrations.
"""

from aura.transport.agui import AguiAdapter
from aura.transport.sse import encode_json_sse, encode_sse
from aura.transport.stream import (
    stream_agent_agui,
    stream_agent_agui_sse,
    stream_agent_wire,
    stream_agent_wire_sse,
)
from aura.transport.wire import agent_state_to_wire, event_to_wire, permission_request_to_wire

__all__ = [
    "AguiAdapter",
    "agent_state_to_wire",
    "encode_json_sse",
    "encode_sse",
    "event_to_wire",
    "permission_request_to_wire",
    "stream_agent_agui",
    "stream_agent_agui_sse",
    "stream_agent_wire",
    "stream_agent_wire_sse",
]
