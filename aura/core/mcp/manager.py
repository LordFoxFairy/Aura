"""MCPManager — thin lifecycle wrapper around :class:`MultiServerMCPClient`.

The heavy lifting (transport, JSON-RPC, schema → pydantic) lives in
``langchain-mcp-adapters``. Aura only adds three concerns on top:

1. **Per-server error isolation.** ``MultiServerMCPClient.get_tools()``
   gathers all servers together with no ``return_exceptions=True``, so a
   single bad server crashes discovery. We iterate per-server instead and
   journal each failure independently — the agent starts with whatever
   servers DID come up.

2. **Aura metadata attachment.** Discovered tools get
   :func:`aura.core.mcp.adapter.add_aura_metadata` applied so the permission
   layer + result-budget hook + concurrency partitioner have the flags they
   need.

3. **Prompt → Command bridging.** MCP prompts are wrapped as Aura commands
   via :func:`aura.core.mcp.adapter.make_mcp_command` so ``/help`` and
   dispatch work uniformly with built-in and skill commands.

The library currently has no ``close()`` — each call spins a fresh stdio
subprocess and tears it down immediately. :meth:`stop_all` is therefore a
defensive no-op plus any aclose-shaped method the library may add later.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from aura.core.mcp.adapter import add_aura_metadata, make_mcp_command
from aura.core.mcp.types import MCPServerConfig

if TYPE_CHECKING:
    from aura.core.commands.types import Command


class MCPManager:
    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._configs = [c for c in configs if c.enabled]
        self._client: MultiServerMCPClient | None = None

    @staticmethod
    async def _list_prompts(
        client: MultiServerMCPClient, server_name: str
    ) -> list[Any]:
        """List an MCP server's prompts via a transient session.

        Broken out as a static method so tests can monkeypatch it without
        having to mock the full ``session()`` async-context-manager surface.
        Returns an empty list if the server exposes no prompts or if
        listing fails — we don't want a missing prompts capability to block
        tool discovery.
        """
        try:
            async with client.session(server_name) as session:
                response = await session.list_prompts()
                return list(response.prompts)
        except Exception:  # noqa: BLE001
            return []

    async def start_all(self) -> tuple[list[BaseTool], list[Command]]:
        """Connect to each enabled server, discover tools + prompts.

        Per-server failures are caught and journalled; successful servers'
        tools and prompt commands are returned. Returns ``([], [])`` if no
        servers are configured.
        """
        if not self._configs:
            return [], []

        from aura.core import journal

        connections = self._build_connections()
        self._client = MultiServerMCPClient(connections)

        all_tools: list[BaseTool] = []
        all_commands: list[Command] = []

        for cfg in self._configs:
            try:
                tools = await self._client.get_tools(server_name=cfg.name)
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "mcp_connect_failed",
                    server=cfg.name,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            for t in tools:
                add_aura_metadata(t, server_name=cfg.name)
                all_tools.append(t)

            prompts = await self._list_prompts(self._client, cfg.name)
            for p in prompts:
                # mcp.types.Prompt has .name and .description (Optional).
                name = getattr(p, "name", None)
                if not isinstance(name, str) or not name:
                    continue
                description = getattr(p, "description", None) or name
                all_commands.append(
                    make_mcp_command(
                        server_name=cfg.name,
                        prompt_name=name,
                        prompt_description=str(description),
                        client=self._client,
                    )
                )

            journal.write(
                "mcp_server_connected",
                server=cfg.name,
                tool_count=len(tools),
                prompt_count=len(prompts),
            )

        return all_tools, all_commands

    async def stop_all(self) -> None:
        """Tear down the client. The current library has no close(); this
        is a defensive shim that tries any aclose-shaped attribute and
        swallows errors so shutdown never raises.
        """
        if self._client is None:
            return
        for attr in ("aclose", "close"):
            fn = getattr(self._client, attr, None)
            if callable(fn):
                with suppress(Exception):
                    result = fn()
                    if hasattr(result, "__await__"):
                        await result
        self._client = None

    def _build_connections(self) -> dict[str, Any]:
        """Assemble the ``connections`` dict the library consumes.

        Only stdio is wired here — other transports would add branches on
        ``cfg.transport`` once the config schema grows.
        """
        connections: dict[str, Any] = {}
        for cfg in self._configs:
            connections[cfg.name] = {
                "transport": "stdio",
                "command": cfg.command,
                "args": list(cfg.args),
                "env": dict(cfg.env) if cfg.env else None,
            }
        return connections
