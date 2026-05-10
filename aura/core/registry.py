"""Name-keyed mapping of LangChain tools used by the agent loop."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.tools import BaseTool

from aura.errors import AuraError
from aura.schemas.tool import ToolMetadata


class ToolRegistryError(AuraError):
    """Raised when a tool fails the registry's contract checks.

    The two contract violations today are:

    - duplicate ``tool.name`` (a tool with that name is already registered)
    - missing or wrong-typed ``aura_metadata`` (every Aura tool must
      expose a :class:`ToolMetadata` instance so loop / hook / CLI
      readers can rely on the typed surface)

    Subclasses :class:`AuraError` so callers that want to uniformly
    handle "Aura knows how to report this" can ``except AuraError``.
    """


def _require_aura_metadata(tool: BaseTool) -> None:
    """Reject tools that don't carry a typed ``aura_metadata: ToolMetadata``.

    Every Aura-registered tool — built-in or MCP-derived — must declare
    its capability flags through :class:`ToolMetadata`. The loop, the
    permission hook, and the CLI renderer read the typed surface via
    :func:`aura.schemas.tool_meta_access.meta_dict`; a tool that skips
    this contract would silently default every flag to ``False`` (see
    that helper's "missing aura_metadata returns ``{}``" branch), which
    is dangerous for ``is_destructive`` in particular.

    Raises :class:`ToolRegistryError` with a name-oriented message so
    the operator can locate the offending tool.
    """
    aura_meta = getattr(tool, "aura_metadata", None)
    if not isinstance(aura_meta, ToolMetadata):
        raise ToolRegistryError(
            f"tool {tool.name!r} is missing required aura_metadata: "
            f"ToolMetadata (got {type(aura_meta).__name__}); every "
            f"Aura-registered tool must declare ToolMetadata so the "
            f"loop / permission hook / CLI can read its capability flags"
        )


class ToolRegistry(dict[str, BaseTool]):
    """`dict[tool.name, tool]` with dedup-on-construction.

    Subclasses dict to inherit the full mapping protocol (len / getitem /
    contains / iter over keys) and to document the domain type in signatures.
    """

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
        """Add ``tool`` to the registry; raise on duplicate name or
        missing ``aura_metadata``.

        Exists for dynamic registration (MCP tools discovered post-Agent
        construction). Constructor-time dedup via __init__'s loop is the
        normal path; this variant raises a name-oriented error so callers
        get a useful message out of the error path.

        ``aura_metadata: ToolMetadata`` is enforced here (Phase 2 Task 4)
        so a tool registered through any path — built-in subclass, MCP
        adapter, or stateful factory — can be trusted by readers.
        """
        _require_aura_metadata(tool)
        if tool.name in self:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove the tool with the given ``name``; idempotent.

        Absent name is a no-op so MCP teardown / skill reload sites don't
        need to probe membership first.
        """
        self.pop(name, None)


def assemble_tool_pool(
    builtins: Iterable[BaseTool],
    mcp_tools: Iterable[BaseTool],
) -> dict[str, BaseTool]:
    """Merge builtin + MCP tools into a stable, dedup'd, ordered mapping.

    Contract:

    - Each input partition is sorted by ``tool.name`` (stable, deterministic
      ordering for prompt-cache friendliness — re-running with the same
      inputs produces the same tool-schema prefix).
    - Builtins come first in the result; MCP tools follow.
    - On a name collision between an MCP tool and a builtin, the builtin
      wins; the MCP tool is dropped and a ``mcp_tool_shadowed`` journal
      event is emitted naming both sides.
    - Within the MCP partition itself, first-seen wins after sort (stable
      against duplicate MCP-side names too — second occurrence is dropped
      silently since the audit-relevant case is builtin-vs-MCP, not the
      degenerate intra-MCP collision).
    - Returns a plain ``dict`` whose insertion order is the final pool
      order; callers that want ``ToolRegistry`` semantics can wrap.
    """
    from aura.core import journal  # noqa: PLC0415  — avoid import cycle at module load

    builtin_sorted = sorted(builtins, key=lambda t: t.name)
    mcp_sorted = sorted(mcp_tools, key=lambda t: t.name)

    pool: dict[str, BaseTool] = {}
    builtin_names: set[str] = set()
    for t in builtin_sorted:
        if t.name in pool:
            continue
        pool[t.name] = t
        builtin_names.add(t.name)

    for t in mcp_sorted:
        if t.name in pool:
            shadowed_by = "builtin" if t.name in builtin_names else "mcp"
            journal.write(
                "mcp_tool_shadowed",
                tool=t.name,
                shadowed_by=shadowed_by,
            )
            continue
        pool[t.name] = t

    return pool
