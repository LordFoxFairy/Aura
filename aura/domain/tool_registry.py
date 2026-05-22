"""Capabilities-owned tool registry abstraction."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.tools import BaseTool

from aura.domain.errors import AuraError
from aura.schemas.tool import ToolMetadata


class ToolRegistryError(AuraError):
    """Raised when a tool fails the registry's contract checks."""


def _require_aura_metadata(tool: BaseTool) -> None:
    """Reject tools that don't carry a typed ``aura_metadata: ToolMetadata``."""
    aura_meta = getattr(tool, "aura_metadata", None)
    if not isinstance(aura_meta, ToolMetadata):
        raise ToolRegistryError(
            f"tool {tool.name!r} is missing required aura_metadata: "
            f"ToolMetadata (got {type(aura_meta).__name__}); every "
            f"Aura-registered tool must declare ToolMetadata so the "
            f"loop / permission hook / CLI can read its capability flags"
        )


class ToolRegistry(dict[str, BaseTool]):
    """`dict[tool.name, tool]` with dedup-on-construction."""

    def __init__(self, tools: Iterable[BaseTool] = ()) -> None:
        super().__init__()
        for t in tools:
            _require_aura_metadata(t)
            if t.name in self:
                raise ValueError(f"duplicate tool name: {t.name!r}")
            self[t.name] = t

    def tools(self) -> list[BaseTool]:
        return list(self.values())

    def register(self, tool: BaseTool) -> None:
        _require_aura_metadata(tool)
        if tool.name in self:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self[tool.name] = tool

    def unregister(self, name: str) -> None:
        self.pop(name, None)


__all__ = ["ToolRegistry", "ToolRegistryError"]
