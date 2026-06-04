"""Stateless ``MultiServerMCPClient`` connection-entry builders with env expansion."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aura.config.env import expand_env_vars
from aura.infrastructure.mcp.types import MCPServerConfig


def build_connections(
    configs: list[MCPServerConfig],
    *,
    message_handler_for: Callable[[str], Callable[[Any], Awaitable[None]]],
) -> dict[str, Any]:
    connections: dict[str, Any] = {}
    for cfg in configs:
        connections.update(
            build_one_connection(cfg, message_handler_for=message_handler_for)
        )
    return connections


def build_one_connection(
    cfg: MCPServerConfig,
    *,
    message_handler_for: Callable[[str], Callable[[Any], Awaitable[None]]],
) -> dict[str, Any]:
    """Build one connections entry; unresolved ${VAR} (no default) raises RuntimeError."""
    missing: list[str] = []

    def _expand(text: str | None) -> str | None:
        if text is None:
            return None
        return expand_env_vars(text, _missing_log=missing)

    session_kwargs = {
        "message_handler": message_handler_for(cfg.name),
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
