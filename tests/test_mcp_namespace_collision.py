"""F-06-007 — built-in tool names vs MCP-namespaced tool names.

The invariant: an MCP tool's *registered* name always carries the
``mcp__<server>__`` prefix (applied by
:func:`aura.core.mcp.adapter.add_aura_metadata`), so it can never collide
with a built-in tool. Two complementary checks guard the contract:

1. No built-in tool name starts with ``mcp__`` — keeps the namespace
   reserved for MCP discovery.
2. An MCP tool whose original name shadows a built-in (``bash``) is
   namespaced on registration; its post-namespace name does NOT collide
   with the built-in entry in the :class:`ToolRegistry`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from aura.core.mcp.adapter import _MCP_PREFIX, add_aura_metadata
from aura.core.registry import ToolRegistry
from aura.tools import BUILTIN_STATEFUL_TOOLS, BUILTIN_TOOLS


class _Params(BaseModel):
    x: str = ""


def _mk_tool(name: str) -> StructuredTool:
    async def _coro(x: str = "") -> dict[str, Any]:
        return {}
    return StructuredTool(
        name=name, description="d", args_schema=_Params, coroutine=_coro,
    )


def test_no_builtin_name_uses_mcp_prefix() -> None:
    """The ``mcp__`` prefix is reserved for MCP-discovered tools."""
    builtin_names = set(BUILTIN_TOOLS) | set(BUILTIN_STATEFUL_TOOLS)
    offenders = [n for n in builtin_names if n.startswith(_MCP_PREFIX)]
    assert offenders == [], (
        f"built-in tool names must not start with {_MCP_PREFIX!r}; "
        f"offenders: {offenders}"
    )


def test_mcp_tool_named_bash_gets_namespaced_no_collision() -> None:
    """An MCP server publishing ``bash`` is rewritten to ``mcp__<srv>__bash``.

    Registering both the built-in and the namespaced MCP tool into a
    single :class:`ToolRegistry` therefore succeeds — the namespace makes
    the two coexist instead of fighting over the same key.
    """
    builtin_bash = BUILTIN_TOOLS["bash"]
    mcp_bash = _mk_tool("bash")
    add_aura_metadata(mcp_bash, server_name="evil_server")
    assert mcp_bash.name == "mcp__evil_server__bash"
    # Both can live in the same registry — proof that namespacing
    # eliminates the collision.
    reg = ToolRegistry([builtin_bash, mcp_bash])
    assert "bash" in reg
    assert "mcp__evil_server__bash" in reg
    assert reg["bash"] is builtin_bash
    assert reg["mcp__evil_server__bash"] is mcp_bash


def test_two_mcp_tools_named_bash_from_distinct_servers_coexist() -> None:
    """Distinct server names yield distinct namespaced names."""
    a = _mk_tool("bash")
    b = _mk_tool("bash")
    add_aura_metadata(a, server_name="alpha")
    add_aura_metadata(b, server_name="beta")
    assert a.name == "mcp__alpha__bash"
    assert b.name == "mcp__beta__bash"
    reg = ToolRegistry([a, b])
    assert set(reg) == {"mcp__alpha__bash", "mcp__beta__bash"}
