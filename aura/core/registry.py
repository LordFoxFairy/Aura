"""Compatibility façade for the capabilities-owned tool registry and catalog."""

from aura.capabilities.tools.catalog import assemble_tool_pool
from aura.capabilities.tools.registry import ToolRegistry, ToolRegistryError

__all__ = ["ToolRegistry", "ToolRegistryError", "assemble_tool_pool"]
