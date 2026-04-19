"""Name-keyed mapping of LangChain tools used by the agent loop."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.tools import BaseTool


class ToolRegistry(dict[str, BaseTool]):
    """`dict[tool.name, tool]` with dedup-on-construction.

    Subclasses dict to inherit the full mapping protocol (len / getitem /
    contains / iter over keys) and to document the domain type in signatures.
    """

    def __init__(self, tools: Iterable[BaseTool] = ()) -> None:
        super().__init__()
        for t in tools:
            if t.name in self:
                raise ValueError(f"duplicate tool name: {t.name!r}")
            self[t.name] = t

    def tools(self) -> list[BaseTool]:
        return list(self.values())

    def names(self) -> list[str]:
        return list(self.keys())
