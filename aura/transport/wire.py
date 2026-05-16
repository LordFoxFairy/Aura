"""Compatibility facade for Aura wire protocol serialization."""

from aura.adapters.protocol.wire import (
    agent_state_to_wire,
    compact_event_to_wire,
    event_to_wire,
    permission_request_to_wire,
)

__all__ = [
    "agent_state_to_wire",
    "compact_event_to_wire",
    "event_to_wire",
    "permission_request_to_wire",
]
