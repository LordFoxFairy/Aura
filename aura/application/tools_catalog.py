"""Tool catalogue assembly — merges builtin + MCP tools into one pool."""

from collections.abc import Iterable

from langchain_core.tools import BaseTool


def assemble_tool_pool(
    builtins: Iterable[BaseTool],
    mcp_tools: Iterable[BaseTool],
    *,
    mcp_overrides: bool = False,
) -> dict[str, BaseTool]:
    """Merge builtin + MCP tools into a stable, dedup'd, ordered mapping.

    Builtins win on name collision unless ``mcp_overrides=True``. Both
    streams are pre-sorted by name so the result mapping has deterministic
    iteration order across runs.
    """
    from aura.core import journal

    pool: dict[str, BaseTool] = {}
    builtin_names: set[str] = set()
    for tool in sorted(builtins, key=lambda t: t.name):
        if tool.name in pool:
            continue
        pool[tool.name] = tool
        builtin_names.add(tool.name)

    for tool in sorted(mcp_tools, key=lambda t: t.name):
        if tool.name in pool:
            collides_with_builtin = tool.name in builtin_names
            if collides_with_builtin and mcp_overrides:
                pool[tool.name] = tool
                journal.write(
                    "mcp_tool_shadowed",
                    tool=tool.name, winner="mcp", shadowed_by="mcp",
                )
                continue
            winner = "builtin" if collides_with_builtin else "mcp"
            journal.write(
                "mcp_tool_shadowed",
                tool=tool.name, winner=winner, shadowed_by=winner,
            )
            continue
        pool[tool.name] = tool

    return pool
