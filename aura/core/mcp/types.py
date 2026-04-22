"""Shared MCP types — re-export the pydantic config from ``aura.config.schema``.

Kept as a separate module so that code importing from ``aura.core.mcp`` can
ignore whether the model lives here or in the config package; future Aura-
specific MCP errors / protocol types can land here without disturbing
callers.
"""

from __future__ import annotations

from aura.config.schema import MCPServerConfig

__all__ = ["MCPServerConfig"]
