"""Compatibility facade for SSE framing helpers."""

from aura.adapters.protocol.stream import encode_json_sse, encode_sse

__all__ = ["encode_json_sse", "encode_sse"]
