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

6. **Resilience: auto-reconnect + per-op timeout.**
   - Remote transports (``sse`` / ``streamable_http``) schedule exponential-
     backoff reconnect tasks when a connect fails mid-session. Backoff
     schedule: 1s, 2s, 4s, 8s, 16s (capped at 60s), max 5 attempts —
     mirrors ``useManageMCPConnections.ts`` in claude-code v2.1.88.
     Stdio transports skip auto-reconnect: subprocess death is a
     user-visible event that backoff can't heal.
   - All client-facing operations (``get_tools`` / ``list_prompts`` /
     ``list_resources`` / ``read_resource`` / ``get_prompt``) are wrapped
     in :func:`asyncio.wait_for` with a configurable timeout (default
     ``30s``, override via ``AURA_MCP_TIMEOUT_SEC`` env var or the
     ``op_timeout_sec`` ctor kwarg). On timeout we raise a descriptive
     :class:`RuntimeError` so the user gets an actionable message instead
     of a silent hang.

The library currently has no ``close()`` — each call spins a fresh stdio
subprocess and tears it down immediately. :meth:`stop_all` is therefore a
defensive no-op plus any aclose-shaped method the library may add later,
and also cancels any pending reconnect timer tasks.
"""

from __future__ import annotations

import asyncio
import os
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


MCPServerState = Literal[
    "connected",
    "connecting",
    "disabled",
    "error",
    "needs_auth",
    "never_started",
    "unapproved",
]


# ---------------------------------------------------------------------------
# F-06-003 — needs-auth classifier
#
# Routes auth-failure exceptions to the ``needs_auth`` state so the
# manager can:
#   - skip auto-reconnect (retrying a 401 burns rate budget without
#     learning anything new);
#   - journal a distinct ``mcp_connect_needs_auth`` event so operators
#     spot it amid generic connect failures.
# Detection mirrors claude-code: the spec'd JSON-RPC code -32001
# ("Authentication required") plus a substring fallback for SDKs that
# stringify auth errors instead of carrying the code field.
# ---------------------------------------------------------------------------

_NEEDS_AUTH_CODE = -32001
_NEEDS_AUTH_HINTS = ("oauth", "unauthorized", "401", "403")


def _is_needs_auth_error(exc: BaseException) -> bool:
    """Return True iff *exc* indicates an MCP authentication failure.

    Matches:
    - ``exc.code == -32001`` (the MCP-spec'd JSON-RPC auth code).
    - Lowercase ``str(exc)`` containing any of "oauth", "unauthorized",
      "401", "403" — covers SDKs that stringify auth errors without
      preserving the structured code field.
    """
    code = getattr(exc, "code", None)
    if code == _NEEDS_AUTH_CODE:
        return True
    text = str(exc).lower()
    return any(hint in text for hint in _NEEDS_AUTH_HINTS)


# Exponential-backoff reconnect constants. Schedule (seconds): 1, 2, 4, 8, 16
# — capped at 60s (``_MAX_BACKOFF_SEC``), max 5 attempts. Matches
# claude-code's ``useManageMCPConnections.ts`` pattern, only we use stdlib
# ``asyncio.sleep`` instead of ``setTimeout`` + React refs.
_INITIAL_BACKOFF_SEC = 1.0
_MAX_BACKOFF_SEC = 60.0
_MAX_RECONNECT_ATTEMPTS = 5

# Per-operation timeout. Reads ``AURA_MCP_TIMEOUT_SEC`` from env at
# MCPManager construction time; the ctor kwarg wins if supplied. Claude-code
# uses effectively-infinite (~27.8h) — we prefer a hard 30s default so a
# frozen server can't wedge the agent loop indefinitely. An env override
# lets power users match claude-code's behaviour.
_DEFAULT_OP_TIMEOUT_SEC = 30.0
_OP_TIMEOUT_ENV_VAR = "AURA_MCP_TIMEOUT_SEC"


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


def _resolve_op_timeout(explicit: float | None) -> float:
    """Resolve the per-op timeout: explicit kwarg > env > default.

    Returns a positive float. Invalid env values (non-numeric, ≤0) are
    silently ignored and we fall back to the default — we don't want a
    typo in a shell-rc file to block MCP startup. Callers requesting a
    non-positive explicit value get a :class:`ValueError` (programmer
    error, not a config error).
    """
    if explicit is not None:
        if explicit <= 0:
            raise ValueError(
                f"op_timeout_sec must be positive, got {explicit!r}"
            )
        return float(explicit)
    env_val = os.environ.get(_OP_TIMEOUT_ENV_VAR)
    if env_val:
        try:
            parsed = float(env_val)
        except ValueError:
            return _DEFAULT_OP_TIMEOUT_SEC
        if parsed > 0:
            return parsed
    return _DEFAULT_OP_TIMEOUT_SEC


# F-06-004 — MCP server-pushed list-changed notifications. Each ClientSession
# can register a ``message_handler`` callback that receives every inbound
# notification / request. We hand the library a closure per server that
# records ``notifications/{tools,prompts,resources}/list_changed`` events
# to the journal, named by server. Aura's MCP sessions are short-lived
# (one tool call per session) so an auto-rediscovery loop is impractical
# without a persistent-session rearchitecture; the journal trail is the
# observable surface in the meantime.
_LIST_CHANGED_METHODS: frozenset[str] = frozenset({
    "notifications/tools/list_changed",
    "notifications/prompts/list_changed",
    "notifications/resources/list_changed",
})


def _make_list_changed_logger(server_name: str) -> Any:
    """Return a langchain-mcp-adapters ``message_handler`` for *server_name*.

    The handler accepts every inbound message but only acts on the three
    list-changed notification methods. Anything else passes through to
    the library's default handler (a stdlib-only forwarder that suppresses
    unknown messages). Implemented as a closure so the journal event
    carries the server name without a global registry lookup.
    """
    from aura.core.persistence import journal  # noqa: PLC0415

    async def _handler(message: Any) -> None:
        # The library hands us either a request, notification, or exception.
        # We pattern-match on the ``method`` attribute to avoid a heavyweight
        # MCP-types import at module load.
        method = None
        root = getattr(message, "root", None) or message
        method = getattr(root, "method", None)
        if isinstance(method, str) and method in _LIST_CHANGED_METHODS:
            journal.write(
                "mcp_list_changed",
                server=server_name,
                method=method,
            )

    return _handler


class MCPManager:
    def __init__(
        self,
        configs: list[MCPServerConfig],
        *,
        op_timeout_sec: float | None = None,
        project_server_names: set[str] | None = None,
    ) -> None:
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

        # Reconnect task handles, keyed by server name. Stored so
        # ``stop_all`` / ``disable`` / ``reconnect`` / a subsequent
        # ``_schedule_reconnect`` can cancel a pending attempt cleanly
        # (claude-code stores timers in ``reconnectTimersRef``; we store
        # the asyncio Task we spawned). Only remote-transport servers ever
        # get an entry here — stdio never auto-reconnects.
        self._reconnect_tasks: dict[str, asyncio.Task[None]] = {}

        # Per-op timeout (seconds). Resolved once at ctor time so runtime
        # changes to the env var don't cause inconsistent per-call
        # behaviour within one session.
        self._op_timeout_sec: float = _resolve_op_timeout(op_timeout_sec)

        # F-06-001 — project-server approval gate. ``project_server_names``
        # tags which entries came from a project-layer ``mcp_servers.json``
        # (the RCE channel). Auto-detect from :mod:`aura.config.mcp_store`
        # when caller didn't supply the set.
        if project_server_names is None:
            try:
                from aura.config import mcp_store as _store  # noqa: PLC0415
                project_server_names = _store.project_layer_names()
            except Exception:  # noqa: BLE001
                project_server_names = set()
        self._project_server_names: set[str] = set(project_server_names)

        # Pre-compute the unapproved set: any project-layer name whose
        # approval is missing or whose fingerprint doesn't match the live
        # config. Re-checked / mutated by approve / revoke / reload.
        self._unapproved: set[str] = set()
        from aura.config import mcp_approvals as _approvals  # noqa: PLC0415
        from aura.core import journal as _j  # noqa: PLC0415
        for cfg in self._configs_all:
            if cfg.name not in self._project_server_names:
                continue
            if not _approvals.is_approved(cfg):
                self._unapproved.add(cfg.name)
                with suppress(Exception):
                    _j.write(
                        "mcp_server_unapproved",
                        server=cfg.name,
                        project=_approvals.project_key(),
                    )

        for cfg in self._configs_all:
            # enabled=False in config is the "disabled" state from the
            # operator's perspective; enabled=True but not yet connected
            # is "never_started". Unapproved project-layer entries get
            # their own state so ``/mcp list`` surfaces a CTA.
            if cfg.name in self._unapproved:
                self._state[cfg.name] = "unapproved"
            else:
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

    @property
    def op_timeout_sec(self) -> float:
        """Per-operation timeout in seconds (read-only).

        Surfaced for tests + ``/mcp`` status rendering. Resolved once at
        construction time; see :func:`_resolve_op_timeout`.
        """
        return self._op_timeout_sec

    async def _run_with_timeout(
        self, coro: Any, *, op_name: str, server: str,
    ) -> Any:
        """Run *coro* with :func:`asyncio.wait_for` using the manager timeout.

        On :class:`asyncio.TimeoutError`, re-raises as :class:`RuntimeError`
        with a user-actionable message naming the operation + server +
        timeout. Any other exception is passed through unchanged so
        existing error-handling paths keep working.
        """
        try:
            return await asyncio.wait_for(coro, timeout=self._op_timeout_sec)
        except TimeoutError as exc:
            # ``asyncio.wait_for`` raises the stdlib :class:`TimeoutError`
            # (since py3.11 ``asyncio.TimeoutError`` is an alias). We wrap
            # it into a descriptive :class:`RuntimeError` so the surfacing
            # layer doesn't have to decide whether an op-level timeout is
            # "transient network hiccup" or "config is wrong" — the text
            # names the op + server + cap so the operator can act.
            raise RuntimeError(
                f"MCP operation {op_name!r} on server {server!r} "
                f"timed out after {self._op_timeout_sec}s"
            ) from exc

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
        servers are configured. Unapproved project-layer servers are
        skipped — see :meth:`approve` / :meth:`unapproved_server_names`.
        """
        approved_configs = [
            c for c in self._configs if c.name not in self._unapproved
        ]
        if not approved_configs:
            return [], []

        # Build connections only for approved servers — never instantiate
        # a transport for an unapproved entry.
        connections: dict[str, Any] = {}
        for cfg in approved_configs:
            connections.update(self._build_one_connection(cfg))
        self._client = MultiServerMCPClient(connections)

        all_tools: list[BaseTool] = []
        all_commands: list[Command] = []

        for cfg in approved_configs:
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

        Remote-transport (sse / streamable_http) failures additionally
        schedule an exponential-backoff reconnect task. Stdio failures do
        not — a dead subprocess is a user-visible event that retrying
        won't fix.
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
            tools = await self._run_with_timeout(
                self._client.get_tools(server_name=cfg.name),
                op_name="get_tools",
                server=cfg.name,
            )
        except Exception as exc:  # noqa: BLE001
            err_text = f"{type(exc).__name__}: {exc}"
            if _is_needs_auth_error(exc):
                # Auth failure — distinct journal event + state, NO
                # auto-reconnect (a 401 doesn't heal on retry).
                journal.write(
                    "mcp_connect_needs_auth",
                    server=cfg.name,
                    error=err_text,
                )
                self._state[cfg.name] = "needs_auth"
                self._errors[cfg.name] = err_text
            else:
                journal.write(
                    "mcp_connect_failed",
                    server=cfg.name,
                    error=err_text,
                )
                self._state[cfg.name] = "error"
                self._errors[cfg.name] = err_text
                # Remote transports get auto-reconnect; stdio does not
                # (see module docstring for rationale). Guard against
                # double-scheduling: if an attempt is already pending
                # for this server, leave it alone.
                if cfg.transport in ("sse", "streamable_http"):
                    self._schedule_reconnect(cfg)
            self._tool_counts[cfg.name] = 0
            self._prompt_counts[cfg.name] = 0
            self._resource_counts[cfg.name] = 0
            return [], []

        for t in tools:
            add_aura_metadata(t, server_name=cfg.name)

        try:
            prompts = await self._run_with_timeout(
                self._list_prompts(self._client, cfg.name),
                op_name="list_prompts",
                server=cfg.name,
            )
        except RuntimeError:
            # Graceful-degrade on prompt-list timeout: the server is
            # up (``get_tools`` succeeded); we just don't surface any
            # prompts for this session. Journal so the operator sees why.
            journal.write(
                "mcp_list_prompts_timeout",
                server=cfg.name,
                timeout_sec=self._op_timeout_sec,
            )
            prompts = []
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
                    op_timeout_sec=self._op_timeout_sec,
                )
            )

        try:
            resources = await self._run_with_timeout(
                self._list_resources(self._client, cfg.name),
                op_name="list_resources",
                server=cfg.name,
            )
        except RuntimeError:
            journal.write(
                "mcp_list_resources_timeout",
                server=cfg.name,
                timeout_sec=self._op_timeout_sec,
            )
            resources = []
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
        # If we got here, we're connected. Any prior reconnect task for
        # this server is now stale — a subsequent disconnect will spawn a
        # fresh one. Cancel the stale handle so ``stop_all`` has a clean
        # map to iterate.
        self._cancel_reconnect_task(cfg.name)
        return list(tools), commands

    # ------------------------------------------------------------------
    # Auto-reconnect
    # ------------------------------------------------------------------

    def _cancel_reconnect_task(self, name: str) -> None:
        """Cancel any pending reconnect task for *name* (idempotent).

        Safe to call even when no task is scheduled. We drop the handle
        from the map regardless of whether ``.cancel()`` succeeded — the
        task is about to go away one way or another.

        **Self-safety.** If this is invoked from *inside* the reconnect
        task itself (e.g. ``_connect_one`` → successful reconnect →
        clear-the-handle), we only remove the handle from the map but
        do NOT call ``.cancel()`` — cancelling the current task would
        raise :class:`asyncio.CancelledError` at the next ``await`` and
        bubble out of the loop. The task drops out of the map naturally
        when it returns.
        """
        task = self._reconnect_tasks.pop(name, None)
        if task is None or task.done():
            return
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if task is current:
            # Self-cancel suppression: let the task return naturally.
            return
        task.cancel()

    def _schedule_reconnect(self, cfg: MCPServerConfig) -> None:
        """Spawn a background reconnect task for *cfg*.

        Guarded against double-scheduling: if a task is already live for
        this server, we leave it alone so the in-flight backoff curve
        isn't reset by a concurrent failure signal. Only called from
        :meth:`_connect_one` on remote-transport failure, so stdio
        servers never hit this path.
        """
        existing = self._reconnect_tasks.get(cfg.name)
        if existing is not None and not existing.done():
            return
        # Spawn the loop and stash the handle. We intentionally don't
        # await it — reconnect runs concurrently with normal agent work.
        task = asyncio.create_task(
            self._reconnect_loop(cfg),
            name=f"mcp-reconnect:{cfg.name}",
        )
        self._reconnect_tasks[cfg.name] = task

    async def _reconnect_loop(self, cfg: MCPServerConfig) -> None:
        """Exponential-backoff reconnect loop for one remote-transport server.

        Schedule (seconds per attempt sleep): 1, 2, 4, 8, 16 — capped at
        60s (``_MAX_BACKOFF_SEC``). Max 5 attempts
        (``_MAX_RECONNECT_ATTEMPTS``), matching claude-code's
        ``useManageMCPConnections.ts`` constants. If the server is
        disabled or reconnected out-of-band between attempts, we
        short-circuit and return.

        On success: state flips to ``"connected"`` (done inside
        ``_connect_one`` on the successful call) and we return normally.

        On max-attempts-exhausted: ``_state[name]`` stays ``"error"``
        (last ``_connect_one`` call set it) until the operator runs
        ``/mcp reconnect`` or restarts the session.
        """
        from aura.core import journal

        for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
            # Sleep BEFORE each attempt — gives the remote side time to
            # recover. Backoff grows 1s → 2s → 4s → 8s → 16s, capped at
            # 60s. Expressed in terms of attempt number (1-indexed).
            backoff = min(
                _INITIAL_BACKOFF_SEC * (2 ** (attempt - 1)),
                _MAX_BACKOFF_SEC,
            )
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                return

            # Bail if the server got disabled / reconnected out-of-band
            # while we were sleeping. Race-safe: we read the current
            # state, and ``disable`` / ``reconnect`` cancel us explicitly.
            current_state = self._state.get(cfg.name)
            if current_state in ("disabled", "connected"):
                return

            journal.write(
                "mcp_reconnect_attempt",
                server=cfg.name,
                attempt=attempt,
                max_attempts=_MAX_RECONNECT_ATTEMPTS,
                backoff_sec=backoff,
            )

            # ``_connect_one`` fully updates state + counts + journal on
            # both success and failure. It also re-calls
            # ``_schedule_reconnect`` on failure — but the guard at the
            # top of that method prevents us from re-spawning ourselves
            # (we're still live in ``_reconnect_tasks[name]`` until this
            # coroutine returns).
            await self._connect_one(cfg)

            if self._state.get(cfg.name) == "connected":
                journal.write(
                    "mcp_reconnect_succeeded",
                    server=cfg.name,
                    attempt=attempt,
                )
                return

        # Exhausted the attempt budget. ``_connect_one`` already set state
        # back to ``"error"`` on the last failure; we just journal the
        # give-up event so operators see "we tried 5x, you're on your
        # own now".
        journal.write(
            "mcp_reconnect_exhausted",
            server=cfg.name,
            max_attempts=_MAX_RECONNECT_ATTEMPTS,
        )
        # Drop our own handle out of the map so a future /mcp reconnect
        # (or a new failure signal on a re-enabled server) can cleanly
        # spawn a fresh loop.
        self._reconnect_tasks.pop(cfg.name, None)

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

        Raises :class:`RuntimeError` if the underlying MCP read stalls
        past :attr:`op_timeout_sec` — converted from
        :class:`asyncio.TimeoutError` for a readable, user-facing shape.
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

        async def _do_read() -> Any:
            # Wrap the full session+read in a single awaitable so
            # ``wait_for`` cancels both if the read hangs. The session
            # context manager honours the cancellation and tears the
            # underlying transport down.
            assert self._client is not None  # for mypy — guarded above
            async with self._client.session(owning_server) as session:
                return await session.read_resource(AnyUrl(uri_str))

        result = await self._run_with_timeout(
            _do_read(),
            op_name="read_resource",
            server=owning_server,
        )
        contents = [
            normalize_resource_contents(c) for c in result.contents
        ]
        return {
            "uri": uri_str,
            "server": owning_server,
            "contents": contents,
        }

    async def stop_all(self) -> None:
        """Tear down the client + cancel pending reconnect tasks.

        The current library has no close(); this is a defensive shim that
        tries any aclose-shaped attribute and swallows errors so shutdown
        never raises. All scheduled reconnect tasks are cancelled first
        so we don't leak pending work past shutdown.
        """
        # Cancel reconnect timers first — they hold strong refs to ``self``
        # and would otherwise keep firing after teardown. ``.cancel()``
        # is best-effort; the task may be already-finished, in which case
        # cancel is a no-op.
        for task in list(self._reconnect_tasks.values()):
            if not task.done():
                task.cancel()
        # Drain cancelled tasks so tests can assert a clean shutdown.
        # ``gather(..., return_exceptions=True)`` suppresses the
        # CancelledError we just injected.
        if self._reconnect_tasks:
            with suppress(Exception):
                await asyncio.gather(
                    *self._reconnect_tasks.values(),
                    return_exceptions=True,
                )
        self._reconnect_tasks.clear()

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

        F-06-002 — applies ``${VAR}`` / ``${VAR:-default}`` expansion to
        every string leaf (``command`` / each ``args`` / each ``env``
        value / each ``headers`` value / ``url``). This is belt-and-
        suspenders defence for configs that bypass ``mcp_store.load()``
        (programmatic ``MCPServerConfig`` construction in SDK callers /
        tests). An unresolved reference with no default raises a
        :class:`RuntimeError` that names the missing var + server so
        the operator gets actionable text instead of a silent empty
        substitution baked into a transport spawn.

        F-06-004 — every connection carries a ``session_kwargs`` with a
        ``message_handler`` that journals list-changed notifications
        (``notifications/tools/list_changed``,
        ``notifications/prompts/list_changed``,
        ``notifications/resources/list_changed``). This makes the events
        observable to operators even though Aura's MCP sessions are
        currently short-lived (one tool call per session) so an
        auto-rediscovery loop is impractical without a persistent-session
        rearchitecture. The journal trail is enough for an operator to
        see when an upstream server's catalog drifted; subsequent
        ``/mcp reconnect`` rebuilds the cache. The handler is wired here
        rather than in :class:`MCPServerConfig` so SDK callers who
        construct connections by hand still get the audit hook for free.
        """
        from aura.core.mcp.adapter import _expand_env_vars  # noqa: PLC0415

        missing: list[str] = []

        def _expand(text: str | None) -> str | None:
            if text is None:
                return None
            return _expand_env_vars(text, _missing_log=missing)

        session_kwargs = {
            "message_handler": _make_list_changed_logger(cfg.name),
        }

        if cfg.transport == "stdio":
            command = _expand(cfg.command)
            args = [_expand_env_vars(a, _missing_log=missing) for a in cfg.args]
            env_raw = cfg.env or {}
            env = {
                k: _expand_env_vars(v, _missing_log=missing)
                for k, v in env_raw.items()
            }
            if missing:
                raise RuntimeError(
                    f"MCP server {cfg.name!r}: unresolved environment "
                    f"variable(s) in stdio config: "
                    f"{sorted(set(missing))} "
                    "(define them in the parent shell or supply "
                    "${VAR:-default})"
                )
            return {
                cfg.name: {
                    "transport": "stdio",
                    "command": command,
                    "args": args,
                    "env": env if env else None,
                    "session_kwargs": session_kwargs,
                }
            }
        conn: dict[str, Any] = {
            "transport": cfg.transport,
            "url": _expand(cfg.url),
            "session_kwargs": session_kwargs,
        }
        if cfg.headers:
            conn["headers"] = {
                k: _expand_env_vars(v, _missing_log=missing)
                for k, v in cfg.headers.items()
            }
        if missing:
            raise RuntimeError(
                f"MCP server {cfg.name!r}: unresolved environment "
                f"variable(s) in {cfg.transport} config: "
                f"{sorted(set(missing))} "
                "(define them in the parent shell or supply "
                "${VAR:-default})"
            )
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
        # Cancel any in-flight auto-reconnect loop for this server — the
        # operator is explicitly taking over, so a parallel background
        # attempt would just race with us.
        self._cancel_reconnect_task(name)
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
        # Cancel any pending reconnect — the operator has explicitly
        # disabled this server, so backoff retries would un-disable it.
        self._cancel_reconnect_task(name)
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
        # Cancel any pending auto-reconnect — the operator is driving
        # the reconnect themselves, no background retries need to race.
        self._cancel_reconnect_task(name)
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
            err = (
                self._errors.get(name)
                if state in ("error", "needs_auth")
                else None
            )
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

    # ------------------------------------------------------------------
    # F-06-001 — approval gate API
    # ------------------------------------------------------------------

    def unapproved_server_names(self) -> set[str]:
        """Return the set of project-layer server names awaiting approval."""
        return set(self._unapproved)

    def needs_auth_server_names(self) -> set[str]:
        """Return server names whose latest connect failed with a 401/oauth."""
        return {
            name for name, state in self._state.items()
            if state == "needs_auth"
        }

    async def approve(self, name: str) -> str:
        """Approve a project-layer server: persist + (re)connect.

        Unknown-name returns a textual error shape (no exception). A
        user-scope server (not in ``project_server_names``) returns a
        friendly no-op — there's nothing to approve.
        """
        cfg = self._config_by_name(name)
        if cfg is None:
            known = self.known_server_names()
            return (
                f"no MCP server named {name!r}; "
                f"known: {known}"
            )
        if name not in self._project_server_names:
            return (
                f"MCP server {name!r} is user-scope; approval is not required"
            )
        from aura.config import mcp_approvals as _approvals  # noqa: PLC0415
        _approvals.approve(cfg)
        self._unapproved.discard(name)
        # Flip from ``unapproved`` to ``never_started`` so the connect
        # path will pick it up. Then attempt a connect immediately so the
        # user sees the result.
        self._state[name] = "never_started"
        if cfg.enabled and cfg not in self._configs:
            self._configs.append(cfg)
        if cfg.enabled:
            await self._connect_one(cfg)
        new_state = self._state.get(name, "never_started")
        if new_state == "connected":
            return f"MCP server {name!r} approved and connected"
        return f"MCP server {name!r} approved (state: {new_state})"

    async def revoke(self, name: str) -> str:
        """Revoke approval and tear down any live connection."""
        cfg = self._config_by_name(name)
        if cfg is None:
            known = self.known_server_names()
            return (
                f"no MCP server named {name!r}; "
                f"known: {known}"
            )
        from aura.config import mcp_approvals as _approvals  # noqa: PLC0415
        _approvals.revoke(name)
        self._unapproved.add(name)
        # Tear the live connection down.
        self._configs = [c for c in self._configs if c.name != name]
        if self._client is not None:
            self._client.connections.pop(name, None)
        self._resources = {
            k: v for k, v in self._resources.items() if k[0] != name
        }
        self._cancel_reconnect_task(name)
        self._state[name] = "unapproved"
        self._errors.pop(name, None)
        self._tool_counts[name] = 0
        self._prompt_counts[name] = 0
        self._resource_counts[name] = 0
        return f"MCP server {name!r} approval revoked and disconnected"

    async def reload(
        self,
        configs: list[MCPServerConfig],
        *,
        project_server_names: set[str] | None = None,
    ) -> str:
        """Re-seed the manager from a freshly-loaded config list.

        Returns a one-line summary like ``"+1 -0"`` describing what
        changed (added / removed). New entries default to
        ``never_started``; removed entries vanish from status. Existing
        entries keep their current state.
        """
        if project_server_names is None:
            try:
                from aura.config import mcp_store as _store  # noqa: PLC0415
                project_server_names = _store.project_layer_names()
            except Exception:  # noqa: BLE001
                project_server_names = set()

        before_names = {c.name for c in self._configs_all}
        after_names = {c.name for c in configs}
        added = sorted(after_names - before_names)
        removed = sorted(before_names - after_names)

        # Drop removed servers entirely.
        for name in removed:
            self._state.pop(name, None)
            self._errors.pop(name, None)
            self._tool_counts.pop(name, None)
            self._prompt_counts.pop(name, None)
            self._resource_counts.pop(name, None)
            self._unapproved.discard(name)
            self._cancel_reconnect_task(name)
            if self._client is not None:
                self._client.connections.pop(name, None)
            self._resources = {
                k: v for k, v in self._resources.items() if k[0] != name
            }

        # Replace the config list.
        self._configs_all = list(configs)
        self._configs = [c for c in configs if c.enabled]
        self._project_server_names = set(project_server_names)

        # Recompute approval state for new entries.
        from aura.config import mcp_approvals as _approvals  # noqa: PLC0415
        for cfg in self._configs_all:
            if cfg.name in added:
                if (
                    cfg.name in self._project_server_names
                    and not _approvals.is_approved(cfg)
                ):
                    self._unapproved.add(cfg.name)
                    self._state[cfg.name] = "unapproved"
                else:
                    self._state[cfg.name] = (
                        "never_started" if cfg.enabled else "disabled"
                    )
                self._tool_counts[cfg.name] = 0
                self._prompt_counts[cfg.name] = 0
                self._resource_counts[cfg.name] = 0
        return f"+{len(added)} -{len(removed)}"
