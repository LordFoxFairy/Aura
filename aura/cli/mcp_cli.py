"""Subcommand handlers for ``aura mcp add|list|remove``.

These handlers are pure file-ops: no LLM, no REPL, no agent. They edit
``~/.aura/mcp_servers.json`` via :mod:`aura.config.mcp_store` and return
the standard exit codes:

- 0 on success
- 1 on user error (duplicate name on add, unknown name on remove, etc.)
- 2 is reserved for argparse; we never emit it from here.
"""

from __future__ import annotations

import argparse
import sys

from aura.config.mcp_store import get_path, load, save
from aura.config.schema import MCPServerConfig


def _parse_env_pairs(raw: list[str]) -> dict[str, str]:
    """Parse ``--env KEY=VAL`` flags into a dict.

    Accepts values that themselves contain ``=`` (common for base64, URLs).
    The key side must be non-empty; a missing ``=`` is a user error.
    """
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


def _cmd_add(args: argparse.Namespace) -> int:
    name: str = args.name
    transport: str = args.transport
    # ``command_args`` is produced by ``_split_dashdash`` in __main__: the
    # list of tokens after the literal ``--`` separator. Never contains
    # the ``--`` itself.
    raw_tokens: list[str] = list(args.command_args or [])

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
                transport=transport,  # type: ignore[arg-type]
                url=url,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    try:
        servers = load()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if any(s.name == name for s in servers):
        print(
            f"error: MCP server {name!r} already exists in {get_path()}. "
            f"Run 'aura mcp remove {name}' first, then re-add.",
            file=sys.stderr,
        )
        return 1

    servers.append(entry)
    save(servers)

    if transport == "stdio":
        joined = " ".join([entry.command or ""] + list(entry.args))
        print(f"Added stdio MCP server {name!r} with command: {joined.strip()}")
    else:
        print(f"Added {transport} MCP server {name!r} with url: {entry.url}")
    print(f"File modified: {get_path()}")
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

    headers = ("NAME", "TRANSPORT", "COMMAND", "ENABLED")
    rows: list[tuple[str, str, str, str]] = []
    for s in servers:
        if s.transport == "stdio":
            command_repr = " ".join([s.command or ""] + list(s.args)).strip()
        else:
            command_repr = s.url or ""
        rows.append((s.name, s.transport, command_repr, "yes" if s.enabled else "no"))

    # Column widths computed once — header + widest row per column. Simple
    # space-padded table; no external dep since rich can't be used from a
    # pure-file-op path (would trigger Console init and slow the CLI).
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def _fmt(row: tuple[str, str, str, str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(_fmt(headers))
    for row in rows:
        print(_fmt(row))
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    name: str = args.name
    try:
        servers = load()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    filtered = [s for s in servers if s.name != name]
    if len(filtered) == len(servers):
        print(
            f"error: MCP server {name!r} not found in {get_path()}.",
            file=sys.stderr,
        )
        return 1
    save(filtered)
    print(f"Removed MCP server {name!r}")
    print(f"File modified: {get_path()}")
    return 0


def handle_mcp(args: argparse.Namespace) -> int:
    """Dispatch ``aura mcp <action>`` to the matching handler.

    ``mcp_action`` is wired by the argparse subparsers; ``None`` means the
    user ran ``aura mcp`` with no sub-subcommand, which we treat as a user
    error (print help-like hint + exit 1) rather than silently doing
    nothing.
    """
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
