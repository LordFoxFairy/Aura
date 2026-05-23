"""MCP runtime — extracted from :class:`Agent` (Phase 2 §6).

Pre-Phase-2, :class:`aura.core.agent.Agent` interleaved MCP server
lifecycle with everything else: a ``_mcp_manager`` field, a ``_mcp_commands``
list, the connect logic in :meth:`Agent.aconnect`, the disconnect logic in
:meth:`Agent.aclose`, the ``_connected_server_names`` helper for
shutdown-timeout journaling. ~120 lines of MCP-specific code lived on
the god object.

Phase 2 Task 8 lifts that into :class:`McpRuntime`. The runtime owns:

- the live :class:`MCPManager` reference (or ``None`` before connect /
  after teardown)
- the ``mcp_commands`` list (slash-command stubs each connected server
  contributes; consumed by the CLI's command registry)
- the connect / disconnect / status sequencing (with journal events
  for ``mcp_aconnect_failed`` / ``mcp_aconnect_done`` /
  ``mcp_close_timeout`` / ``mcp_close_error`` / ``mcp_stopped``)
- the registry-merge step (``assemble_tool_pool`` invocation honoring
  the configured ``tools.mcp_overrides_builtin`` policy)

It does NOT own:

- the loop ``_rebind_tools`` callback. The runtime returns the merged
  tool set; the caller (``Agent``) re-binds the loop. The loop is
  Agent's collaborator, not the MCP runtime's.
- the slash-command registry. The runtime exposes ``mcp_commands``;
  ``Agent`` / the CLI consume it.
- the choice of ``MCPManager`` factory. Tests monkey-patch the factory
  through ``aura.core.agent.MCPManager`` (the historical seam) — the
  runtime accepts an optional override so the same monkey-patches keep
  working after extraction.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

from aura.application.tools_catalog import assemble_tool_pool
from aura.domain.tool_registry import ToolRegistry
from aura.infrastructure.persistence import journal

if TYPE_CHECKING:
    from aura.config.schema import MCPServerConfig
    from aura.infrastructure.mcp.manager import MCPManager


# Factory type used to lazy-construct the MCPManager. ``Agent.aconnect``
# historically did ``MCPManager(self._config.mcp_servers)`` after a
# top-level ``from aura.infrastructure.mcp import MCPManager``; tests monkey-patch
# that name. We accept a factory so the same hooks keep working without
# the runtime hard-importing the manager module.
McpManagerFactory = Callable[[list["MCPServerConfig"]], "MCPManager"]


def _default_manager_factory(
    configs: list[MCPServerConfig],
) -> MCPManager:
    """Default factory: import + construct :class:`MCPManager`.

    Kept lazy so module load doesn't pull the heavy MCP transport stack
    when no servers are configured.
    """
    from aura.infrastructure.mcp import (
        MCPManager,  # noqa: PLC0415  # deferred import is intentional
    )

    return MCPManager(configs)


class McpRuntime:
    """Lifecycle sidecar for MCP servers attached to one :class:`Agent`.

    One instance per Agent; mirrors Agent's lifetime exactly.

    The runtime is sync-constructed (no servers contacted) and async-
    activated via :meth:`connect_all`. ``Agent.aconnect`` calls into
    :meth:`connect_all`, then re-binds the loop's tool list (loop is
    Agent's collaborator, not ours).
    """

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
        # Live manager — populated by :meth:`connect_all` on success,
        # set back to ``None`` by :meth:`disconnect_all` (success and
        # failure paths alike, so re-entry is idempotent).
        self._manager: MCPManager | None = None
        # Slash-command stubs harvested from each connected server.
        # Consumed by the CLI's command registry; the runtime just
        # holds the list.
        self._commands: list[Any] = []
        # Tool catalog harvested from servers — empty before connect
        # and after disconnect. Kept so callers can introspect what
        # the last connect produced without re-asking the manager.
        self._tools: list[BaseTool] = []

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def manager(self) -> MCPManager | None:
        """Live :class:`MCPManager`, or ``None`` outside connect."""
        return self._manager

    @manager.setter
    def manager(self, value: MCPManager | None) -> None:
        """Direct setter — tests assign a fake manager onto the
        runtime to exercise close-path behaviour without spinning up a
        real transport. Not part of the user-facing surface; the
        forwarding ``Agent._mcp_manager`` shim relies on this setter
        so historical test code keeps working post-extraction.
        """
        self._manager = value

    @property
    def commands(self) -> list[Any]:
        """Slash-command stubs from currently-connected servers."""
        return self._commands

    @commands.setter
    def commands(self, value: list[Any]) -> None:
        self._commands = list(value)

    @property
    def tools(self) -> list[BaseTool]:
        """Tool snapshot from the latest successful connect."""
        return list(self._tools)

    @property
    def has_servers(self) -> bool:
        """True iff any MCP server is configured."""
        return bool(self._configs)

    @property
    def is_connected(self) -> bool:
        """True iff :meth:`connect_all` succeeded and we still hold the
        manager (i.e. :meth:`disconnect_all` has not run)."""
        return self._manager is not None

    @property
    def mcp_overrides_builtin(self) -> bool:
        return self._mcp_overrides_builtin

    # ------------------------------------------------------------------
    # Connect / discovery
    # ------------------------------------------------------------------

    async def connect_all(
        self,
        builtin_tools: list[BaseTool],
    ) -> dict[str, BaseTool] | None:
        """Start every configured MCP server and merge tools.

        Returns the merged ``{name: BaseTool}`` mapping (builtin +
        MCP, with the configured collision policy applied) on success.
        Returns ``None`` if no servers are configured OR if the
        connect sequence raised — callers treat ``None`` as "do not
        re-bind anything; the registry is unchanged".

        Failures journal ``mcp_aconnect_failed`` and are swallowed (the
        agent must keep functioning if a server is down — graceful
        degradation is a v0.3.0 non-negotiable).

        Success journals ``mcp_aconnect_done`` with tool / command /
        resource counts so operators can audit what came online.
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

        # F-02-031 — merge through ``assemble_tool_pool`` so builtin-vs-MCP
        # collisions resolve under the configured policy and emit a
        # ``mcp_tool_shadowed`` journal event with the policy outcome.
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
        """Swap ``registry``'s contents for the merged pool, in place.

        Clear-then-add preserves the existing :class:`ToolRegistry`
        identity (the loop binds the registry instance once at
        construction; mutating contents must not invalidate that
        binding). Idempotent — calling with an empty ``merged`` empties
        the registry, which is the right behaviour for a future
        "reload all" command.
        """
        for name in list(registry):
            registry.unregister(name)
        for tool in merged.values():
            registry.register(tool)

    # ------------------------------------------------------------------
    # Status / diagnostics
    # ------------------------------------------------------------------

    def connected_server_names(self) -> list[str]:
        """Best-effort snapshot of servers still in ``connected`` state.

        Used to populate ``servers_hanging`` on the close-timeout
        journal event. ``manager.status()`` is pure-sync and
        defensively written never to raise — if a half-torn-down
        manager misbehaves we degrade to ``[]`` rather than poisoning
        the shutdown path.
        """
        mgr = self._manager
        if mgr is None:
            return []
        try:
            entries = mgr.status()
        except Exception:  # noqa: BLE001  # log + swallow; logging path must never crash caller
            return []
        return [
            e.name for e in entries
            if getattr(e, "state", None) == "connected"
        ]

    # ------------------------------------------------------------------
    # Disconnect — timeout-bounded teardown (B3)
    # ------------------------------------------------------------------

    async def disconnect_all(
        self,
        *,
        session_id: str,
        timeout_sec: float = 5.0,
    ) -> None:
        """Tear down every connected MCP server.

        Contract (B3 — preserved verbatim from the pre-extraction
        :meth:`Agent.aclose` block):

        - Runs :meth:`MCPManager.stop_all` under :func:`asyncio.wait_for`.
        - On timeout: cancel the coroutine, journal ``mcp_close_timeout``
          with ``{session, elapsed_sec, timeout_sec, servers_hanging}``.
        - On unexpected exception: journal ``mcp_close_error`` with
          the exception message (shutdown is best-effort; a thrown
          exception MUST NOT crash the caller).
        - Normal completion: journal ``mcp_stopped`` with ``elapsed_sec``.
        - The manager reference is dropped on every branch so a
          subsequent ``disconnect_all`` is an idempotent no-op.

        No-op if :attr:`is_connected` is False.
        """
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
