"""Lifecycle wrapper around :class:`MultiServerMCPClient`.

Adds per-server error isolation, Aura tool metadata, prompt-to-Command
bridging, a resource catalogue, an in-REPL control surface, and
auto-reconnect + per-op timeout on top of the upstream library.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from langchain_core.tools import BaseTool
from langchain_mcp_adapters import sessions
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import AnyUrl

from aura.application.commands.types import Command
from aura.config import mcp_approvals, mcp_store
from aura.config.env import expand_env_vars
from aura.config.schema import AuraConfigError
from aura.infrastructure.mcp.adapter import (
    add_aura_metadata,
    make_mcp_command,
    normalize_resource_contents,
)
from aura.infrastructure.mcp.types import MCPServerConfig
from aura.infrastructure.persistence import journal


@runtime_checkable
class _HasCode(Protocol):
    code: object


@runtime_checkable
class _HasRoot(Protocol):
    root: object


@runtime_checkable
class _HasMethod(Protocol):
    method: object


@runtime_checkable
class _PromptLike(Protocol):
    name: object
    description: object
    arguments: object


@runtime_checkable
class _ResourceLike(Protocol):
    uri: object
    name: object
    description: object
    mimeType: object


@runtime_checkable
class _AsyncClosable(Protocol):
    def aclose(self) -> Awaitable[object]: ...


@runtime_checkable
class _Closable(Protocol):
    def close(self) -> object: ...


MCPServerState = Literal[
    "connected",
    "connecting",
    "disabled",
    "error",
    "needs_auth",
    "never_started",
    "unapproved",
]


_NEEDS_AUTH_CODE = -32001
_NEEDS_AUTH_HINTS = ("oauth", "unauthorized", "401", "403")


def _is_needs_auth_error(exc: BaseException) -> bool:
    """Return True iff *exc* indicates an MCP authentication failure."""
    if isinstance(exc, _HasCode) and exc.code == _NEEDS_AUTH_CODE:
        return True
    text = str(exc).lower()
    return any(hint in text for hint in _NEEDS_AUTH_HINTS)


# Backoff schedule (seconds): 1, 2, 4, 8, 16 — capped at 60s, max 5 attempts.
_INITIAL_BACKOFF_SEC = 1.0
_MAX_BACKOFF_SEC = 60.0
_MAX_RECONNECT_ATTEMPTS = 5

_DEFAULT_OP_TIMEOUT_SEC = 30.0
_OP_TIMEOUT_ENV_VAR = "AURA_MCP_TIMEOUT_SEC"


@dataclass(frozen=True)
class MCPServerStatus:
    """Snapshot of one MCP server for the ``/mcp`` list view."""

    name: str
    transport: str
    state: MCPServerState
    error_message: str | None
    tool_count: int
    resource_count: int
    prompt_count: int


def _supported_transports() -> set[str]:
    """Transports the installed ``langchain-mcp-adapters`` understands."""
    supported = {"stdio"}
    if hasattr(sessions, "SSEConnection"):
        supported.add("sse")
    if hasattr(sessions, "StreamableHttpConnection"):
        supported.add("streamable_http")
    return supported


def _resolve_op_timeout(explicit: float | None) -> float:
    """Resolve the per-op timeout: explicit kwarg > env > default."""
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


_LIST_CHANGED_METHODS: frozenset[str] = frozenset({
    "notifications/tools/list_changed",
    "notifications/prompts/list_changed",
    "notifications/resources/list_changed",
})


def _make_list_changed_logger(
    server_name: str,
) -> Callable[[Any], Awaitable[None]]:
    """Return a ``message_handler`` closure that journals list-changed events."""
    async def _handler(message: Any) -> None:
        root = message.root if isinstance(message, _HasRoot) and message.root else message
        method = root.method if isinstance(root, _HasMethod) else None
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
        # Keep ALL configs so /mcp list can still show a disabled server.
        self._configs_all: list[MCPServerConfig] = list(configs)
        self._configs: list[MCPServerConfig] = [c for c in configs if c.enabled]
        self._client: MultiServerMCPClient | None = None
        self._resources: dict[tuple[str, str], _ResourceLike] = {}

        self._state: dict[str, MCPServerState] = {}
        self._errors: dict[str, str] = {}
        self._tool_counts: dict[str, int] = {}
        self._prompt_counts: dict[str, int] = {}
        self._resource_counts: dict[str, int] = {}

        self._reconnect_tasks: dict[str, asyncio.Task[None]] = {}

        # Resolved once so runtime env mutations don't cause per-call drift.
        self._op_timeout_sec: float = _resolve_op_timeout(op_timeout_sec)

        if project_server_names is None:
            try:
                project_server_names = mcp_store.project_layer_names()
            except Exception:  # noqa: BLE001  # store failure degrades to empty approval set
                project_server_names = set()
        self._project_server_names: set[str] = set(project_server_names)

        self._unapproved: set[str] = set()
        for cfg in self._configs_all:
            if cfg.name not in self._project_server_names:
                continue
            if not mcp_approvals.is_approved(cfg):
                self._unapproved.add(cfg.name)
                with suppress(Exception):
                    journal.write(
                        "mcp_server_unapproved",
                        server=cfg.name,
                        project=mcp_approvals.project_key(),
                    )

        for cfg in self._configs_all:
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
        return self._op_timeout_sec

    async def _run_with_timeout(
        self, coro: Any, *, op_name: str, server: str,
    ) -> Any:
        """Wrap *coro* in :func:`asyncio.wait_for`; re-raise timeout as RuntimeError."""
        try:
            return await asyncio.wait_for(coro, timeout=self._op_timeout_sec)
        except TimeoutError as exc:
            raise RuntimeError(
                f"MCP operation {op_name!r} on server {server!r} "
                f"timed out after {self._op_timeout_sec}s"
            ) from exc

    @staticmethod
    async def _list_prompts(
        client: MultiServerMCPClient, server_name: str
    ) -> list[Any]:
        """List prompts; empty list on any failure so discovery stays alive."""
        try:
            async with client.session(server_name) as session:
                response = await session.list_prompts()
                return list(response.prompts)
        except Exception:  # noqa: BLE001  # missing prompts capability must not block tool discovery
            return []

    @staticmethod
    async def _list_resources(
        client: MultiServerMCPClient, server_name: str
    ) -> list[Any]:
        """List resources; empty list on any failure so discovery stays alive."""
        try:
            async with client.session(server_name) as session:
                response = await session.list_resources()
                return list(response.resources)
        except Exception:  # noqa: BLE001  # missing resources capability must not block tool discovery
            return []

    async def start_all(self) -> tuple[list[BaseTool], list[Command[object]]]:
        """Connect each enabled+approved server; return discovered tools + prompts."""
        approved_configs = [
            c for c in self._configs if c.name not in self._unapproved
        ]
        if not approved_configs:
            return [], []

        connections: dict[str, Any] = {}
        for cfg in approved_configs:
            connections.update(self._build_one_connection(cfg))
        self._client = MultiServerMCPClient(connections)

        all_tools: list[BaseTool] = []
        all_commands: list[Command[object]] = []

        for cfg in approved_configs:
            tools, commands = await self._connect_one(cfg)
            all_tools.extend(tools)
            all_commands.extend(commands)

        return all_tools, all_commands

    async def _connect_one(
        self, cfg: MCPServerConfig
    ) -> tuple[list[BaseTool], list[Command[object]]]:
        """Connect one server; flip state + counts; never raise.

        Remote-transport failures schedule an auto-reconnect; stdio failures
        do not (subprocess death needs operator intervention).
        """
        if self._client is None:
            self._client = MultiServerMCPClient(self._build_connections())
        elif cfg.name not in self._client.connections:
            self._client.connections.update(
                self._build_one_connection(cfg)
            )

        try:
            tools = await self._run_with_timeout(
                self._client.get_tools(server_name=cfg.name),
                op_name="get_tools",
                server=cfg.name,
            )
        except Exception as exc:  # noqa: BLE001  # per-server connect must not crash discovery
            err_text = f"{type(exc).__name__}: {exc}"
            if _is_needs_auth_error(exc):
                # Auth failure: skip auto-reconnect; a 401 won't heal on retry.
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
            # Server up (get_tools succeeded); just skip prompts this session.
            journal.write(
                "mcp_list_prompts_timeout",
                server=cfg.name,
                timeout_sec=self._op_timeout_sec,
            )
            prompts = []
        commands: list[Command[object]] = []
        for p in prompts:
            if not isinstance(p, _PromptLike):
                continue
            name = p.name
            if not isinstance(name, str) or not name:
                continue
            description = p.description or name
            arguments = p.arguments if isinstance(p.arguments, list) else []
            commands.append(
                make_mcp_command(
                    server_name=cfg.name,
                    prompt_name=name,
                    prompt_description=str(description),
                    client=self._client,
                    prompt_arguments=arguments,
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
        # Reconnect may yield fewer resources; drop stale entries first.
        self._resources = {
            k: v for k, v in self._resources.items() if k[0] != cfg.name
        }
        for r in resources:
            if not isinstance(r, _ResourceLike) or r.uri is None:
                continue
            self._resources[(cfg.name, str(r.uri))] = r

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
        self._cancel_reconnect_task(cfg.name)
        return list(tools), commands

    def _cancel_reconnect_task(self, name: str) -> None:
        """Cancel any pending reconnect task for *name* (idempotent).

        Invariant: self-cancellation is suppressed — calling from inside
        the reconnect task drops the handle but does not invoke
        ``.cancel()`` to avoid raising :class:`asyncio.CancelledError`.
        """
        task = self._reconnect_tasks.pop(name, None)
        if task is None or task.done():
            return
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if task is current:
            return
        task.cancel()

    def _schedule_reconnect(self, cfg: MCPServerConfig) -> None:
        """Spawn a background reconnect task; no-op if one is already live."""
        existing = self._reconnect_tasks.get(cfg.name)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(
            self._reconnect_loop(cfg),
            name=f"mcp-reconnect:{cfg.name}",
        )
        self._reconnect_tasks[cfg.name] = task

    async def _reconnect_loop(self, cfg: MCPServerConfig) -> None:
        """Exponential-backoff reconnect loop for one remote-transport server."""
        for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
            # Sleep BEFORE each attempt to give the remote side time to recover.
            backoff = min(
                _INITIAL_BACKOFF_SEC * (2 ** (attempt - 1)),
                _MAX_BACKOFF_SEC,
            )
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                return

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

            await self._connect_one(cfg)

            if self._state.get(cfg.name) == "connected":
                journal.write(
                    "mcp_reconnect_succeeded",
                    server=cfg.name,
                    attempt=attempt,
                )
                return

        journal.write(
            "mcp_reconnect_exhausted",
            server=cfg.name,
            max_attempts=_MAX_RECONNECT_ATTEMPTS,
        )
        self._reconnect_tasks.pop(cfg.name, None)

    def resources_catalogue(
        self,
    ) -> list[tuple[str, str, str, str, str | None]]:
        """Flattened ``(server, uri, name, description, mime_type)`` tuples.

        Invariant: sorted by ``(server, uri)`` so repeated calls produce
        identical output (prompt-cache stability).
        """
        entries: list[tuple[str, str, str, str, str | None]] = []
        for (server, uri), resource in self._resources.items():
            name = resource.name or uri.rsplit("/", 1)[-1] or uri
            description = resource.description or ""
            mime = resource.mimeType
            mime_type = mime if isinstance(mime, str) else None
            entries.append((server, uri, str(name), str(description), mime_type))
        entries.sort(key=lambda e: (e[0], e[1]))
        return entries

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Fetch a resource's contents by URI.

        Raises :class:`ValueError` for unknown URIs and :class:`RuntimeError`
        when the read stalls past :attr:`op_timeout_sec`.
        """
        if self._client is None:
            raise ValueError(
                "MCP client not started; call start_all() before read_resource"
            )
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

        async def _do_read() -> Any:
            # Single awaitable so wait_for cancels session + read together.
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
        """Cancel reconnect tasks and best-effort close the client."""
        for task in list(self._reconnect_tasks.values()):
            if not task.done():
                task.cancel()
        if self._reconnect_tasks:
            with suppress(Exception):
                await asyncio.gather(
                    *self._reconnect_tasks.values(),
                    return_exceptions=True,
                )
        self._reconnect_tasks.clear()

        if self._client is None:
            return
        # The pinned library exposes no close hook; call one defensively if a
        # future version adds aclose/close so sessions can't leak.
        client: object = self._client
        if isinstance(client, _AsyncClosable):
            with suppress(Exception):
                await client.aclose()
        if isinstance(client, _Closable):
            with suppress(Exception):
                result = client.close()
                if inspect.isawaitable(result):
                    await result
        self._client = None

    def _build_connections(self) -> dict[str, Any]:
        connections: dict[str, Any] = {}
        for cfg in self._configs:
            connections.update(self._build_one_connection(cfg))
        return connections

    @staticmethod
    def _build_one_connection(cfg: MCPServerConfig) -> dict[str, Any]:
        """Build one library ``connections`` entry with env expansion + audit hook.

        Invariant: an unresolved ``${VAR}`` with no default raises
        :class:`RuntimeError` rather than silently substituting empty.
        """
        missing: list[str] = []

        def _expand(text: str | None) -> str | None:
            if text is None:
                return None
            return expand_env_vars(text, _missing_log=missing)

        session_kwargs = {
            "message_handler": _make_list_changed_logger(cfg.name),
        }

        if cfg.transport == "stdio":
            command = _expand(cfg.command)
            args = [expand_env_vars(a, _missing_log=missing) for a in cfg.args]
            env_raw = cfg.env or {}
            env = {
                k: expand_env_vars(v, _missing_log=missing)
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
                k: expand_env_vars(v, _missing_log=missing)
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

    def _config_by_name(self, name: str) -> MCPServerConfig | None:
        for cfg in self._configs_all:
            if cfg.name == name:
                return cfg
        return None

    def known_server_names(self) -> list[str]:
        return [cfg.name for cfg in self._configs_all]

    async def enable(self, name: str) -> str:
        """Bring a server online; never raises on unknown name."""
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
        if cfg not in self._configs:
            self._configs.append(cfg)
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
        """Disconnect + clear discovery state; idempotent."""
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
        self._configs = [c for c in self._configs if c.name != name]
        if self._client is not None:
            self._client.connections.pop(name, None)
        self._resources = {
            k: v for k, v in self._resources.items() if k[0] != name
        }
        self._cancel_reconnect_task(name)
        self._state[name] = "disabled"
        self._errors.pop(name, None)
        self._tool_counts[name] = 0
        self._prompt_counts[name] = 0
        self._resource_counts[name] = 0
        return f"MCP server {name!r} disabled"

    async def reconnect(self, name: str) -> str:
        """Force a disable-then-enable cycle; idempotent."""
        cfg = self._config_by_name(name)
        if cfg is None:
            known = self.known_server_names()
            return (
                f"no MCP server named {name!r}; "
                f"known: {known}"
            )
        if self._client is not None:
            self._client.connections.pop(name, None)
        self._resources = {
            k: v for k, v in self._resources.items() if k[0] != name
        }
        self._cancel_reconnect_task(name)
        self._state[name] = "never_started"
        self._errors.pop(name, None)
        self._tool_counts[name] = 0
        self._prompt_counts[name] = 0
        self._resource_counts[name] = 0
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
        """Pure-sync snapshot of every known server; never raises."""
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

    def unapproved_server_names(self) -> set[str]:
        return set(self._unapproved)

    def needs_auth_server_names(self) -> set[str]:
        return {
            name for name, state in self._state.items()
            if state == "needs_auth"
        }

    async def approve(self, name: str) -> str:
        """Persist approval for a project-layer server, then (re)connect."""
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
        mcp_approvals.approve(cfg)
        self._unapproved.discard(name)
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
        mcp_approvals.revoke(name)
        self._unapproved.add(name)
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
        """Re-seed from a freshly-loaded config list; returns ``"+N -M"`` summary."""
        if project_server_names is None:
            try:
                project_server_names = mcp_store.project_layer_names()
            except Exception:  # noqa: BLE001  # store failure degrades to empty approval set
                project_server_names = set()

        before_names = {c.name for c in self._configs_all}
        after_names = {c.name for c in configs}
        added = sorted(after_names - before_names)
        removed = sorted(before_names - after_names)

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

        self._configs_all = list(configs)
        self._configs = [c for c in configs if c.enabled]
        self._project_server_names = set(project_server_names)

        for cfg in self._configs_all:
            if cfg.name in added:
                if (
                    cfg.name in self._project_server_names
                    and not mcp_approvals.is_approved(cfg)
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
