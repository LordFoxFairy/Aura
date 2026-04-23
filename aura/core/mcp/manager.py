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

5. **In-REPL control surface** — ``enable`` / ``disable`` / ``reconnect`` /
   ``status`` power the ``/mcp`` slash command. These never raise on
   unknown-name or mid-session state — they return textual results the
   dispatcher surfaces to the operator. ``status`` is pure-sync + always
   returns a list so the ``/mcp`` list view can render even if every
   server failed.

The library currently has no ``close()`` — each call spins a fresh stdio
subprocess and tears it down immediately. :meth:`stop_all` is therefore a
defensive no-op plus any aclose-shaped method the library may add later.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from aura.config.schema import AuraConfigError
from aura.core.mcp.adapter import add_aura_metadata, make_mcp_command, normalize_resource_contents
from aura.core.mcp.types import MCPServerConfig

if TYPE_CHECKING:
    from aura.core.commands.types import Command


MCPServerState = Literal["connected", "disabled", "error", "never_started"]


@dataclass(frozen=True)
class MCPServerStatus:
    """Snapshot of one MCP server for the ``/mcp`` list view.

    ``state`` is one of ``connected`` / ``disabled`` / ``error`` /
    ``never_started``. ``error_message`` is populated only for the
    ``error`` state; the three count fields are populated only for
    ``connected`` and are zero otherwise (we clear them on disable /
    reconnect to avoid showing stale data from a prior session).
    """

    name: str
    transport: str
    state: MCPServerState
    error_message: str | None
    tool_count: int
    resource_count: int
    prompt_count: int


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
        # Keep ALL configs (enabled + disabled) so ``status()`` /
        # ``/mcp list`` can still show a disabled-by-config server as
        # "disabled" rather than hiding it entirely — matches claude-code's
        # MCPSettings UI which lists every configured server.
        self._configs_all: list[MCPServerConfig] = list(configs)
        self._configs: list[MCPServerConfig] = [c for c in configs if c.enabled]
        self._client: MultiServerMCPClient | None = None
        # (server_name, uri_str) -> Resource descriptor. Built during
        # ``start_all`` from each server's ``session.list_resources()``.
        # URI is stored as str (the mcp pydantic AnyUrl is not hashable
        # in a way we care about here, and all downstream callers compare
        # against str URIs anyway). Empty dict = no resources available —
        # Agent wiring uses that to decide whether to register the
        # ``mcp_read_resource`` tool at all.
        self._resources: dict[tuple[str, str], Any] = {}

        # Per-server state tracking for the ``/mcp`` control surface.
        # ``_state[name]`` is always populated (never_started at boot).
        # ``_errors[name]`` is only set when the latest transition for
        # ``name`` ended in a failed connect. Counts are refreshed on
        # every successful connect and cleared on disable so the list
        # view never shows stale numbers for a server that isn't up.
        self._state: dict[str, MCPServerState] = {}
        self._errors: dict[str, str] = {}
        self._tool_counts: dict[str, int] = {}
        self._prompt_counts: dict[str, int] = {}
        self._resource_counts: dict[str, int] = {}

        for cfg in self._configs_all:
            # enabled=False in config is the "disabled" state from the
            # operator's perspective; enabled=True but not yet connected
            # is "never_started".
            self._state[cfg.name] = (
                "never_started" if cfg.enabled else "disabled"
            )
            self._tool_counts[cfg.name] = 0
            self._prompt_counts[cfg.name] = 0
            self._resource_counts[cfg.name] = 0

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

        connections = self._build_connections()
        self._client = MultiServerMCPClient(connections)

        all_tools: list[BaseTool] = []
        all_commands: list[Command] = []

        for cfg in self._configs:
            tools, commands = await self._connect_one(cfg)
            all_tools.extend(tools)
            all_commands.extend(commands)

        return all_tools, all_commands

    async def _connect_one(
        self, cfg: MCPServerConfig
    ) -> tuple[list[BaseTool], list[Command]]:
        """Connect to a single server and update per-server tracking.

        Extracted from ``start_all`` so ``enable`` / ``reconnect`` can reuse
        the same discovery + bookkeeping path. Always updates ``_state``,
        ``_errors``, and the three count maps before returning. Never
        raises — a failure flips state to ``"error"`` and returns empty
        lists so the caller can proceed with other servers / a fallback
        UI message.
        """
        from aura.core import journal

        # A client must exist before per-server connect (the library's
        # ``session(name)`` looks the name up in ``client.connections``).
        # Callers that haven't run ``start_all`` yet get one spun up here.
        if self._client is None:
            self._client = MultiServerMCPClient(self._build_connections())
        elif cfg.name not in self._client.connections:
            # Server was added after initial ``start_all`` OR was disabled
            # and we removed its entry — re-inject the connection so the
            # library can route ``session()`` calls.
            self._client.connections.update(
                self._build_one_connection(cfg)
            )

        try:
            tools = await self._client.get_tools(server_name=cfg.name)
        except Exception as exc:  # noqa: BLE001
            journal.write(
                "mcp_connect_failed",
                server=cfg.name,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._state[cfg.name] = "error"
            self._errors[cfg.name] = f"{type(exc).__name__}: {exc}"
            self._tool_counts[cfg.name] = 0
            self._prompt_counts[cfg.name] = 0
            self._resource_counts[cfg.name] = 0
            return [], []

        for t in tools:
            add_aura_metadata(t, server_name=cfg.name)

        prompts = await self._list_prompts(self._client, cfg.name)
        commands: list[Command] = []
        for p in prompts:
            # mcp.types.Prompt has .name and .description (Optional).
            name = getattr(p, "name", None)
            if not isinstance(name, str) or not name:
                continue
            description = getattr(p, "description", None) or name
            commands.append(
                make_mcp_command(
                    server_name=cfg.name,
                    prompt_name=name,
                    prompt_description=str(description),
                    client=self._client,
                    # Wire E's prompt-arg forwarding: _MCPPromptCommand reads
                    # ``prompt_arguments`` to build the zip-on-invoke dict so
                    # ``/server__prompt alpha beta`` reaches the server with
                    # ``{arg0: "alpha", arg1: "beta"}`` instead of silently
                    # dropping user args. Defaults to [] for zero-arg prompts.
                    prompt_arguments=getattr(p, "arguments", None) or [],
                )
            )

        resources = await self._list_resources(self._client, cfg.name)
        # Clear any stale resource entries for this server before re-adding
        # — reconnect may legitimately have fewer resources than before.
        self._resources = {
            k: v for k, v in self._resources.items() if k[0] != cfg.name
        }
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

        self._state[cfg.name] = "connected"
        self._errors.pop(cfg.name, None)
        self._tool_counts[cfg.name] = len(tools)
        self._prompt_counts[cfg.name] = len(commands)
        self._resource_counts[cfg.name] = len(resources)
        return list(tools), commands

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
            connections.update(self._build_one_connection(cfg))
        return connections

    @staticmethod
    def _build_one_connection(cfg: MCPServerConfig) -> dict[str, Any]:
        """Single-server version of ``_build_connections``.

        Exposed separately so ``enable`` / ``_connect_one`` can inject one
        server's entry back into an existing client's ``connections`` dict
        without rebuilding the whole map.
        """
        if cfg.transport == "stdio":
            return {
                cfg.name: {
                    "transport": "stdio",
                    "command": cfg.command,
                    "args": list(cfg.args),
                    "env": dict(cfg.env) if cfg.env else None,
                }
            }
        conn: dict[str, Any] = {
            "transport": cfg.transport,
            "url": cfg.url,
        }
        if cfg.headers:
            conn["headers"] = dict(cfg.headers)
        return {cfg.name: conn}

    # ------------------------------------------------------------------
    # In-REPL control surface (``/mcp enable|disable|reconnect``)
    # ------------------------------------------------------------------

    def _config_by_name(self, name: str) -> MCPServerConfig | None:
        for cfg in self._configs_all:
            if cfg.name == name:
                return cfg
        return None

    def known_server_names(self) -> list[str]:
        """Return every known server name in config order.

        Used by the ``/mcp`` command dispatcher to render "known: [...]"
        hints when the operator types an unknown name.
        """
        return [cfg.name for cfg in self._configs_all]

    async def enable(self, name: str) -> str:
        """Bring a server online.

        Contract:
        - Unknown name → ``"no MCP server named <n>; known: [...]"``.
        - Already connected → no-op, returns a "already connected" note.
        - Disabled / never_started / error → attempt connect; return the
          resulting state.

        Never raises on unknown name — ``/mcp enable foo`` should print an
        error, not a traceback.
        """
        cfg = self._config_by_name(name)
        if cfg is None:
            known = self.known_server_names()
            return (
                f"no MCP server named {name!r}; "
                f"known: {known}"
            )
        current = self._state.get(name, "never_started")
        if current == "connected":
            return f"MCP server {name!r} is already connected"
        # Make sure the server's connection entry is in-play even if it
        # was dropped by a prior ``disable`` call.
        if cfg not in self._configs:
            self._configs.append(cfg)
        await self._connect_one(cfg)
        new_state = self._state.get(name, "error")
        if new_state == "connected":
            return f"MCP server {name!r} enabled and connected"
        err = self._errors.get(name, "unknown error")
        return (
            f"MCP server {name!r} failed to connect: {err}"
        )

    async def disable(self, name: str) -> str:
        """Disconnect a server and clear its discovery state.

        Idempotent: disabling an already-disabled server is a friendly
        no-op, not an error. Unknown-name returns the same textual error
        shape as ``enable``.
        """
        cfg = self._config_by_name(name)
        if cfg is None:
            known = self.known_server_names()
            return (
                f"no MCP server named {name!r}; "
                f"known: {known}"
            )
        current = self._state.get(name, "never_started")
        if current == "disabled":
            return f"MCP server {name!r} is already disabled"
        # Drop the connection entry from both our config list AND the
        # live library client so a stray ``session()`` call can't
        # accidentally re-spawn the subprocess.
        self._configs = [c for c in self._configs if c.name != name]
        if self._client is not None:
            self._client.connections.pop(name, None)
        self._resources = {
            k: v for k, v in self._resources.items() if k[0] != name
        }
        self._state[name] = "disabled"
        self._errors.pop(name, None)
        self._tool_counts[name] = 0
        self._prompt_counts[name] = 0
        self._resource_counts[name] = 0
        return f"MCP server {name!r} disabled"

    async def reconnect(self, name: str) -> str:
        """Force a reconnect: disable-then-enable, regardless of state.

        Idempotent — running twice in a row just does the dance twice.
        Unknown-name returns the same textual error shape as ``enable``.
        """
        cfg = self._config_by_name(name)
        if cfg is None:
            known = self.known_server_names()
            return (
                f"no MCP server named {name!r}; "
                f"known: {known}"
            )
        # Tear down current state for this server (best-effort).
        if self._client is not None:
            self._client.connections.pop(name, None)
        self._resources = {
            k: v for k, v in self._resources.items() if k[0] != name
        }
        self._state[name] = "never_started"
        self._errors.pop(name, None)
        self._tool_counts[name] = 0
        self._prompt_counts[name] = 0
        self._resource_counts[name] = 0
        # Re-ensure config is in the active list and reconnect.
        if cfg not in self._configs:
            self._configs.append(cfg)
        await self._connect_one(cfg)
        new_state = self._state.get(name, "error")
        if new_state == "connected":
            return f"MCP server {name!r} reconnected"
        err = self._errors.get(name, "unknown error")
        return (
            f"MCP server {name!r} failed to reconnect: {err}"
        )

    def status(self) -> list[MCPServerStatus]:
        """Return a snapshot for every known server, ordered by config.

        Pure-sync; MUST NOT raise (the caller renders this under the
        ``/mcp`` list view even on a half-torn-down manager). If no
        servers are configured, returns an empty list.
        """
        out: list[MCPServerStatus] = []
        for cfg in self._configs_all:
            name = cfg.name
            state = self._state.get(name, "never_started")
            err = self._errors.get(name) if state == "error" else None
            out.append(
                MCPServerStatus(
                    name=name,
                    transport=cfg.transport,
                    state=state,
                    error_message=err,
                    tool_count=self._tool_counts.get(name, 0),
                    resource_count=self._resource_counts.get(name, 0),
                    prompt_count=self._prompt_counts.get(name, 0),
                )
            )
        return out
