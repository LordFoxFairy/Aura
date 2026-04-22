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

    def register(self, tool: BaseTool) -> None:
        """Add ``tool`` to the registry; raise on duplicate name.

        Exists for dynamic registration (MCP tools discovered post-Agent
        construction). Constructor-time dedup via __init__'s loop is the
        normal path; this variant raises a name-oriented error so callers
        get a useful message out of the error path.
        """
        if tool.name in self:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove the tool with the given ``name``; idempotent.

        Absent name is a no-op so MCP teardown / skill reload sites don't
        need to probe membership first.
        """
        self.pop(name, None)
