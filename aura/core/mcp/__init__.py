"""MCP integration — library-backed transport + Aura metadata/lifecycle."""

from __future__ import annotations

from aura.core.mcp.adapter import add_aura_metadata, make_mcp_command
from aura.core.mcp.manager import MCPManager
from aura.core.mcp.types import MCPServerConfig

__all__ = [
    "MCPManager",
    "MCPServerConfig",
    "add_aura_metadata",
    "make_mcp_command",
]
