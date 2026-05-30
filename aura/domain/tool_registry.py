"""Name-keyed tool registry with metadata contract."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.tools import BaseTool

from aura.domain.errors import AuraError
from aura.domain.tool import HasAuraMetadata, ToolMetadata


class ToolRegistryError(AuraError):
    pass


def _require_aura_metadata(tool: BaseTool) -> None:
    if not (
        isinstance(tool, HasAuraMetadata)
        and isinstance(tool.aura_metadata, ToolMetadata)
    ):
        found = tool.aura_metadata if isinstance(tool, HasAuraMetadata) else None
        raise ToolRegistryError(
            f"tool {tool.name!r} is missing required aura_metadata: "
            f"ToolMetadata (got {type(found).__name__}); every "
            f"Aura-registered tool must declare ToolMetadata so the "
            f"loop / permission hook / CLI can read its capability flags"
        )


class ToolRegistry(dict[str, BaseTool]):
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
