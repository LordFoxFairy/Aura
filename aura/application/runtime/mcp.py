"""MCP server lifecycle sidecar for one :class:`AgentSession`."""
from __future__ import annotations

import asyncio
from collections.abc import Callable

from langchain_core.tools import BaseTool

from aura.application.commands.types import Command
from aura.application.tools_catalog import assemble_tool_pool
from aura.config.schema import MCPServerConfig
from aura.domain.tool_registry import ToolRegistry
from aura.infrastructure.mcp import MCPManager
from aura.infrastructure.persistence import journal

McpManagerFactory = Callable[[list[MCPServerConfig]], MCPManager]


def _default_manager_factory(configs: list[MCPServerConfig]) -> MCPManager:
    return MCPManager(configs)


class McpRuntime:
    """Sync-constructed; activated via :meth:`connect_all`."""

    def __init__(
        self,
        configs: list[MCPServerConfig],
        *,
        mcp_overrides_builtin: bool = False,
        manager_factory: McpManagerFactory | None = None,
    ) -> None:
        self._configs = list(configs)
        self._mcp_overrides_builtin = mcp_overrides_builtin
        self._manager_factory: McpManagerFactory = (
            manager_factory if manager_factory is not None
            else _default_manager_factory
        )
        self._manager: MCPManager | None = None
        self._commands: list[Command[object]] = []
        self._tools: list[BaseTool] = []

    @property
    def manager(self) -> MCPManager | None:
        return self._manager

    @manager.setter
    def manager(self, value: MCPManager | None) -> None:
        self._manager = value

    @property
    def commands(self) -> list[Command[object]]:
        return self._commands

    @commands.setter
    def commands(self, value: list[Command[object]]) -> None:
        self._commands = list(value)

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    @property
    def has_servers(self) -> bool:
        return bool(self._configs)

    @property
    def is_connected(self) -> bool:
        return self._manager is not None

    @property
    def mcp_overrides_builtin(self) -> bool:
        return self._mcp_overrides_builtin

    async def connect_all(
        self,
        builtin_tools: list[BaseTool],
    ) -> dict[str, BaseTool] | None:
        """Start every configured MCP server and merge tools.

        Returns the merged ``{name: BaseTool}`` on success; ``None`` if no servers
        are configured OR the connect raised (failures journal + swallow — graceful
        degradation is non-negotiable).
        """
        if not self._configs:
            return None
        try:
            manager = self._manager_factory(self._configs)
            tools, commands = await manager.start_all()
        except Exception as exc:  # noqa: BLE001  # log + swallow; logging path must never crash caller
            journal.write(
                "mcp_aconnect_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return None
        self._manager = manager
        self._tools = list(tools)
        self._commands = list(commands)

        # ``assemble_tool_pool`` resolves builtin-vs-MCP collisions per configured policy.
        merged = assemble_tool_pool(
            builtin_tools,
            tools,
            mcp_overrides=self._mcp_overrides_builtin,
        )
        catalogue = manager.resources_catalogue()
        journal.write(
            "mcp_aconnect_done",
            tool_count=len(tools),
            command_count=len(commands),
            resource_count=len(catalogue),
        )
        return merged

    def replace_registry_contents(
        self,
        registry: ToolRegistry,
        merged: dict[str, BaseTool],
    ) -> None:
        """Swap ``registry``'s contents in place; preserves the loop's binding identity."""
        for name in list(registry):
            registry.unregister(name)
        for tool in merged.values():
            registry.register(tool)

    def connected_server_names(self) -> list[str]:
        """Best-effort snapshot of servers in ``connected`` state for the timeout journal."""
        mgr = self._manager
        if mgr is None:
            return []
        try:
            entries = mgr.status()
        except Exception:  # noqa: BLE001  # log + swallow; logging path must never crash caller
            return []
        return [e.name for e in entries if e.state == "connected"]

    async def disconnect_all(
        self,
        *,
        session_id: str,
        timeout_sec: float = 5.0,
    ) -> None:
        """Timeout-bounded teardown; manager dropped on every branch (idempotent)."""
        if self._manager is None:
            return
        mgr = self._manager
        servers_hanging = self.connected_server_names()
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        try:
            await asyncio.wait_for(mgr.stop_all(), timeout=timeout_sec)
        except TimeoutError:
            elapsed = loop.time() - t0
            journal.write(
                "mcp_close_timeout",
                session=session_id,
                elapsed_sec=elapsed,
                timeout_sec=timeout_sec,
                servers_hanging=servers_hanging,
            )
        except Exception as exc:  # noqa: BLE001  # log + swallow; logging path must never crash caller
            journal.write(
                "mcp_close_error",
                session=session_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            journal.write(
                "mcp_stopped",
                session=session_id,
                elapsed_sec=loop.time() - t0,
            )
        finally:
            self._manager = None
            self._tools = []
            self._commands = []
