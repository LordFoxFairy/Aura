"""Subcommand handlers for ``aura mcp add|list|remove``."""

from __future__ import annotations

import argparse
import sys

from aura.config.mcp_store import (
    Scope,
    find_scope_of,
    global_path,
    load,
    load_layer,
    project_layer_names,
    project_path,
    save,
)
from aura.config.schema import MCPServerConfig


def _parse_env_pairs(raw: list[str]) -> dict[str, str]:
    # Value may contain ``=`` (base64, URLs); key side must be non-empty.
    env: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(
                f"--env expects KEY=VALUE, got {item!r} (no '=' found)"
            )
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--env expects KEY=VALUE, got empty key in {item!r}")
        env[key] = value
    return env


def _resolve_write_scope(raw: str | None) -> Scope:
    # Flag omitted → global (pre-project-layer default).
    if raw in (None, "global"):
        return "global"
    if raw == "project":
        return "project"
    raise ValueError(f"unknown scope: {raw!r}")


def _scope_path(scope: Scope) -> str:
    return str(global_path() if scope == "global" else project_path())


def _cmd_add(args: argparse.Namespace) -> int:
    name: str = args.name
    transport: str = args.transport
    raw_tokens: list[str] = list(args.command_args or [])

    try:
        scope = _resolve_write_scope(getattr(args, "scope", None))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        env = _parse_env_pairs(args.env or [])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if transport == "stdio":
        if not raw_tokens:
            print(
                "error: stdio transport requires a command after '--'. "
                "Usage: aura mcp add <name> [--transport stdio] -- <command> [args...]",
                file=sys.stderr,
            )
            return 1
        command = raw_tokens[0]
        cmd_args = raw_tokens[1:]
        try:
            entry = MCPServerConfig(
                name=name,
                transport="stdio",
                command=command,
                args=cmd_args,
                env=env,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        # sse / streamable_http — the single positional after `--` is the URL.
        if not raw_tokens:
            print(
                f"error: {transport} transport requires a URL after '--'. "
                f"Usage: aura mcp add <name> --transport {transport} -- <url>",
                file=sys.stderr,
            )
            return 1
        if len(raw_tokens) > 1:
            print(
                f"error: {transport} transport takes exactly one URL "
                f"after '--', got {len(raw_tokens)} tokens: {raw_tokens!r}",
                file=sys.stderr,
            )
            return 1
        url = raw_tokens[0]
        try:
            entry = MCPServerConfig(
                name=name,
                transport=transport,  # type: ignore[arg-type]  # deliberately off-type arg to exercise path
                url=url,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    # Per-layer duplicate check — project may legitimately shadow global.
    try:
        existing = load_layer(scope)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if any(s.name == name for s in existing):
        print(
            f"error: MCP server {name!r} already exists in "
            f"{_scope_path(scope)}. Run 'aura mcp remove {name} "
            f"--scope {scope}' first, then re-add.",
            file=sys.stderr,
        )
        return 1

    existing.append(entry)
    save(existing, scope=scope)

    if transport == "stdio":
        joined = " ".join([entry.command or ""] + list(entry.args))
        print(
            f"Added stdio MCP server {name!r} ({scope}) "
            f"with command: {joined.strip()}"
        )
    else:
        print(
            f"Added {transport} MCP server {name!r} ({scope}) "
            f"with url: {entry.url}"
        )
    print(f"File modified: {_scope_path(scope)}")
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    try:
        servers = load()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not servers:
        print("(no MCP servers configured)")
        return 0

    # load() merged layers but discarded origin; re-query per layer to tag each row.
    try:
        global_names = {s.name for s in load_layer("global")}
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        project_names = project_layer_names()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    def _scope_for(name: str) -> str:
        if name in project_names:
            return "project"
        if name in global_names:
            return "global"
        return "?"

    headers = ("NAME", "SCOPE", "TRANSPORT", "COMMAND", "ENABLED")
    rows: list[tuple[str, str, str, str, str]] = []
    for s in servers:
        if s.transport == "stdio":
            command_repr = " ".join([s.command or ""] + list(s.args)).strip()
        else:
            command_repr = s.url or ""
        rows.append(
            (s.name, _scope_for(s.name), s.transport, command_repr,
             "yes" if s.enabled else "no"),
        )

    # Inline table — pulling rich here would force Console init on a pure file-op path.
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def _fmt(row: tuple[str, str, str, str, str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(_fmt(headers))
    for row in rows:
        print(_fmt(row))
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    name: str = args.name
    raw_scope: str | None = getattr(args, "scope", None)

    # auto / omitted → remove from whichever layer currently resolves the name.
    if raw_scope in (None, "auto"):
        owner = find_scope_of(name)
        if owner is None:
            print(
                f"error: MCP server {name!r} not found in "
                f"{_scope_path('global')} or project layers.",
                file=sys.stderr,
            )
            return 1
        scope: Scope = owner
    else:
        try:
            scope = _resolve_write_scope(raw_scope)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    try:
        current = load_layer(scope)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    filtered = [s for s in current if s.name != name]
    if len(filtered) == len(current):
        print(
            f"error: MCP server {name!r} not found in {_scope_path(scope)}.",
            file=sys.stderr,
        )
        return 1
    save(filtered, scope=scope)
    print(f"Removed MCP server {name!r} ({scope})")
    print(f"File modified: {_scope_path(scope)}")
    return 0


def handle_mcp(args: argparse.Namespace) -> int:
    action = getattr(args, "mcp_action", None)
    if action == "add":
        return _cmd_add(args)
    if action == "list":
        return _cmd_list(args)
    if action == "remove":
        return _cmd_remove(args)
    print(
        "error: missing mcp action. Use: aura mcp {add|list|remove}",
        file=sys.stderr,
    )
    return 1


__all__ = ["handle_mcp"]
