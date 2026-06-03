"""MCP type re-exports for ``aura.infrastructure.mcp`` consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aura.config.schema import MCPServerConfig

__all__ = ["MCPServerConfig", "MCPServerState", "MCPServerStatus"]


MCPServerState = Literal[
    "connected",
    "connecting",
    "disabled",
    "error",
    "needs_auth",
    "never_started",
    "unapproved",
]


@dataclass(frozen=True)
class MCPServerStatus:
    """Snapshot of one MCP server for the ``/mcp`` list view."""

    name: str
    transport: str
    state: MCPServerState
    error_message: str | None
    tool_count: int
    resource_count: int
    prompt_count: int
