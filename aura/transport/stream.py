"""Compatibility facade for protocol streaming helpers."""

from aura.adapters.protocol.stream import (
    stream_agent_agui,
    stream_agent_agui_sse,
    stream_agent_wire,
    stream_agent_wire_sse,
)

__all__ = [
    "stream_agent_agui",
    "stream_agent_agui_sse",
    "stream_agent_wire",
    "stream_agent_wire_sse",
]
