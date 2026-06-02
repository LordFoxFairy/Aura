"""``/mcp`` — in-REPL MCP server management."""

from __future__ import annotations

from aura.application.commands.types import CommandResult, CommandSource
from aura.application.session import AgentSession
from aura.config import mcp_store
from aura.infrastructure.mcp.manager import MCPServerStatus

_VALID_SUBCOMMANDS = (
    "list", "enable", "disable", "reconnect", "approve", "revoke", "reload", "help",
)
_TARGETED_SUBCOMMANDS = frozenset({"enable", "disable", "reconnect", "approve", "revoke"})
_ERR_MAX = 60


class MCPCommand:
    name = "/mcp"
    description = "list / enable / disable / reconnect MCP servers"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[list|enable|disable|reconnect|approve|revoke|reload|help] [name]"

    async def handle(self, arg: str, agent: AgentSession) -> CommandResult:
        tokens = arg.split()
        if not tokens:
            return self._list(agent)
        sub, rest = tokens[0], tokens[1:]
        if sub == "list":
            return self._list(agent)
        if sub == "help":
            return _help()
        if sub == "reload":
            return await self._reload(agent)
        if sub in _TARGETED_SUBCOMMANDS:
            if not rest:
                return CommandResult(
                    handled=True, kind="print",
                    text=f"usage: /mcp {sub} <server-name>",
                )
            return await self._toggle(agent, sub, " ".join(rest).strip())
        return _unknown_subcommand(sub)

    async def _reload(self, agent: AgentSession) -> CommandResult:
        manager = agent.mcp_manager
        if manager is None:
            return CommandResult(
                handled=True, kind="print",
                text="no MCP manager attached (no servers configured)",
            )
        try:
            configs = mcp_store.load()
        except Exception as exc:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
            return CommandResult(handled=True, kind="print", text=f"reload failed: {exc}")
        text = await manager.reload(configs)
        return CommandResult(handled=True, kind="print", text=text)

    def _list(self, agent: AgentSession) -> CommandResult:
        manager = agent.mcp_manager
        statuses: list[MCPServerStatus] = [] if manager is None else manager.status()
        if not statuses:
            return CommandResult(
                handled=True, kind="print", text="(no MCP servers configured)",
            )
        return CommandResult(handled=True, kind="view", text=_render_table(statuses))

    async def _toggle(self, agent: AgentSession, action: str, target: str) -> CommandResult:
        manager = agent.mcp_manager
        if manager is None:
            return CommandResult(
                handled=True, kind="print",
                text="no MCP manager attached (no servers configured)",
            )
        match action:
            case "enable":
                text = await manager.enable(target)
            case "disable":
                text = await manager.disable(target)
            case "reconnect":
                text = await manager.reconnect(target)
            case "approve":
                text = await manager.approve(target)
            case "revoke":
                text = await manager.revoke(target)
            case _:
                return _unknown_subcommand(action)
        return CommandResult(handled=True, kind="print", text=text)


def _render_table(statuses: list[MCPServerStatus]) -> str:
    header = ("NAME", "TRANSPORT", "STATUS", "TOOLS", "RESOURCES", "PROMPTS")
    rows: list[tuple[str, str, str, str, str, str]] = [header]
    for s in statuses:
        rows.append((
            s.name, s.transport, _format_status_cell(s),
            str(s.tool_count), str(s.resource_count), str(s.prompt_count),
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in rows
    ]
    unapproved = [s.name for s in statuses if s.state == "unapproved"]
    if unapproved:
        lines.append("")
        lines.append(
            f"unapproved project-layer servers: {', '.join(unapproved)}. "
            f"Run /mcp approve <name> to load."
        )
    return "\n".join(lines)


def _format_status_cell(s: MCPServerStatus) -> str:
    if s.state == "error":
        msg = s.error_message or "unknown error"
        if len(msg) > _ERR_MAX:
            msg = msg[: _ERR_MAX - 1] + "…"
        return f"error: {msg}"
    return s.state


def _help() -> CommandResult:
    text = "\n".join([
        "Usage: /mcp [subcommand] [args]",
        "",
        "Subcommands:",
        "  list                     show configured MCP servers (default)",
        "  enable <name>            connect a disabled/errored server",
        "  disable <name>           disconnect a currently-connected server",
        "  reconnect <name>         force-restart a server's connection",
        "  approve <name>           grant a project-layer server permission to load",
        "  revoke <name>            revoke a previously approved server",
        "  reload                   re-read mcp_store config and reconcile",
        "  help                     show this message",
        "",
        "Running /mcp with no args is equivalent to /mcp list.",
    ])
    return CommandResult(handled=True, kind="print", text=text)


def _unknown_subcommand(sub: str) -> CommandResult:
    valid = ", ".join(_VALID_SUBCOMMANDS)
    return CommandResult(
        handled=True, kind="print",
        text=f"unknown /mcp subcommand {sub!r}; valid: {valid}",
    )
