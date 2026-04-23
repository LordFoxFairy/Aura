"""MCPManager — thin lifecycle wrapper around :class:`MultiServerMCPClient`.

The heavy lifting (transport, JSON-RPC, schema → pydantic) lives in
``langchain-mcp-adapters``. Aura only adds four concerns on top:

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

4. **Resource catalogue + on-demand read.** MCP servers expose URI-identified
   *resources* (files, DB snapshots, DOC pages ...). The manager lists them
   at ``start_all`` time and stores ``(server, uri) -> Resource`` so the
   agent-side ``mcp_read_resource`` tool can (a) list the catalogue into
   its own dynamic description for the LLM, and (b) route ``read_resource``
   calls to the right server's session. Resources are NOT eagerly fetched
   — a library-side call to ``session.read_resource(uri)`` runs only when
   the LLM invokes the tool.

The library currently has no ``close()`` — each call spins a fresh stdio
subprocess and tears it down immediately. :meth:`stop_all` is therefore a
defensive no-op plus any aclose-shaped method the library may add later.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from aura.config.schema import AuraConfigError
from aura.core.mcp.adapter import add_aura_metadata, make_mcp_command, normalize_resource_contents
from aura.core.mcp.types import MCPServerConfig

if TYPE_CHECKING:
    from aura.core.commands.types import Command


def _supported_transports() -> set[str]:
    """Transports the installed ``langchain-mcp-adapters`` understands.

    We probe the sessions module rather than hardcoding the list — if the
    user downgrades the library, we surface an actionable config error at
    startup instead of letting ``create_session`` blow up mid-loop.
    """
    try:
        from langchain_mcp_adapters import sessions  # noqa: PLC0415
    except ImportError:
        return {"stdio"}
    supported = {"stdio"}
    if hasattr(sessions, "SSEConnection"):
        supported.add("sse")
    if hasattr(sessions, "StreamableHttpConnection"):
        supported.add("streamable_http")
    return supported


class MCPManager:
    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._configs = [c for c in configs if c.enabled]
        self._client: MultiServerMCPClient | None = None
        # (server_name, uri_str) -> Resource descriptor. Built during
        # ``start_all`` from each server's ``session.list_resources()``.
        # URI is stored as str (the mcp pydantic AnyUrl is not hashable
        # in a way we care about here, and all downstream callers compare
        # against str URIs anyway). Empty dict = no resources available —
        # Agent wiring uses that to decide whether to register the
        # ``mcp_read_resource`` tool at all.
        self._resources: dict[tuple[str, str], Any] = {}

        supported = _supported_transports()
        for cfg in self._configs:
            if cfg.transport not in supported:
                raise AuraConfigError(
                    source=f"mcp_servers[{cfg.name!r}]",
                    detail=(
                        f"requested transport {cfg.transport!r} but "
                        "langchain-mcp-adapters doesn't support it; "
                        "upgrade the package or pin 'stdio'"
                    ),
                )

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

    @staticmethod
    async def _list_resources(
        client: MultiServerMCPClient, server_name: str
    ) -> list[Any]:
        """List an MCP server's resources via a transient session.

        Sibling of :meth:`_list_prompts` — same rationale for being a
        static method (test monkeypatchability) and same graceful-degrade
        contract (empty list on failure so a server that doesn't implement
        the resources capability doesn't poison tool / prompt discovery).
        """
        try:
            async with client.session(server_name) as session:
                response = await session.list_resources()
                return list(response.resources)
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

            resources = await self._list_resources(self._client, cfg.name)
            for r in resources:
                uri_val = getattr(r, "uri", None)
                if uri_val is None:
                    continue
                self._resources[(cfg.name, str(uri_val))] = r

            journal.write(
                "mcp_server_connected",
                server=cfg.name,
                tool_count=len(tools),
                prompt_count=len(prompts),
                resource_count=len(resources),
            )

        return all_tools, all_commands

    def resources_catalogue(
        self,
    ) -> list[tuple[str, str, str, str, str | None]]:
        """Flattened view of discovered resources for LLM-visible description.

        Returns a list of ``(server, uri, name, description, mime_type)``
        tuples. ``name`` falls back to the last URI segment if the server
        didn't provide one; ``description`` falls back to empty string.
        ``mime_type`` stays ``None`` when the server didn't declare one
        (the LLM doesn't need mime to invoke — only the URI — but
        descriptions may want to show it).

        Order: stable by ``(server, uri)`` so repeated calls produce
        the same catalogue text (important for prompt caching — a
        reordered description would bust the cached prefix).
        """
        entries: list[tuple[str, str, str, str, str | None]] = []
        for (server, uri), resource in self._resources.items():
            name = getattr(resource, "name", None) or uri.rsplit("/", 1)[-1] or uri
            description = getattr(resource, "description", None) or ""
            mime_type = getattr(resource, "mimeType", None)
            entries.append((server, uri, str(name), str(description), mime_type))
        entries.sort(key=lambda e: (e[0], e[1]))
        return entries

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Fetch a resource's contents by URI.

        Looks up the URI across all discovered servers. On a match,
        opens a transient session on that server and calls
        ``session.read_resource(uri)``; the raw
        :class:`ReadResourceResult.contents` list is normalised via
        :func:`aura.core.mcp.adapter.normalize_resource_contents` so the
        returned shape is pure JSON (the tool result has to ship over
        LangChain's message bus, which does not like pydantic models
        with ``AnyUrl``).

        Raises :class:`ValueError` if the URI isn't in the catalogue —
        the caller (``mcp_read_resource`` tool) converts this into a
        ``ToolError`` so the LLM sees an actionable message listing the
        known URIs.
        """
        if self._client is None:
            raise ValueError(
                "MCP client not started; call start_all() before read_resource"
            )
        # Find which server owns this URI. We compare as str so callers can
        # pass either AnyUrl or plain str.
        uri_str = str(uri)
        owning_server: str | None = None
        for (server, known_uri) in self._resources:
            if known_uri == uri_str:
                owning_server = server
                break
        if owning_server is None:
            known = sorted({u for (_, u) in self._resources})
            raise ValueError(
                f"unknown MCP resource uri {uri_str!r}; "
                f"known uris: {known}"
            )

        # ``ClientSession.read_resource`` annotates its URI argument as
        # ``AnyUrl`` (pydantic), but at runtime the MCP JSON-RPC client
        # accepts any serialisable stand-in. Wrap to keep mypy happy
        # without forcing every caller into pydantic-land.
        from pydantic import AnyUrl

        async with self._client.session(owning_server) as session:
            result = await session.read_resource(AnyUrl(uri_str))
        contents = [
            normalize_resource_contents(c) for c in result.contents
        ]
        return {
            "uri": uri_str,
            "server": owning_server,
            "contents": contents,
        }

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

        Dispatches per ``cfg.transport``: stdio passes command/args/env; sse
        and streamable_http pass url/headers. Field presence is already
        enforced by :class:`MCPServerConfig` so we can index freely here.
        """
        connections: dict[str, Any] = {}
        for cfg in self._configs:
            if cfg.transport == "stdio":
                connections[cfg.name] = {
                    "transport": "stdio",
                    "command": cfg.command,
                    "args": list(cfg.args),
                    "env": dict(cfg.env) if cfg.env else None,
                }
            else:  # sse, streamable_http
                conn: dict[str, Any] = {
                    "transport": cfg.transport,
                    "url": cfg.url,
                }
                if cfg.headers:
                    conn["headers"] = dict(cfg.headers)
                connections[cfg.name] = conn
        return connections
