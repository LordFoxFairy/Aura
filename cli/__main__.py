"""CLI entry point: parse args, load config, build the agent, and run the REPL."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Protocol, TypeAlias
from uuid import uuid4

from rich.console import Console

from aura import __version__
from aura.application.commands.factory import build_default_registry
from aura.application.commands.registry import dispatch
from aura.application.hooks import HookChain
from aura.application.hooks.file_watcher import FileWatcher, default_watch_paths
from aura.application.hooks.logging import wrap_with_event_logger
from aura.application.hooks.permission import make_permission_hook
from aura.application.permission.asker import PermissionAsker
from aura.application.session import AgentSession, build_agent
from aura.application.teams.runtime import run_teammate_main
from aura.config.loader import load_config
from aura.config.schema import AuraConfig, AuraConfigError, PermissionsConfig
from aura.domain.errors import AuraError
from aura.domain.events import AssistantDelta, Final, ToolCallCompleted
from aura.domain.permission.defaults import DEFAULT_ALLOW_RULES
from aura.domain.permission.mode import Mode
from aura.domain.permission.safety import (
    DEFAULT_PROTECTED_READS,
    DEFAULT_PROTECTED_WRITES,
    SafetyPolicy,
)
from aura.domain.permission.session import RuleSet, SessionRuleSet
from aura.domain.tool import ToolError
from aura.infrastructure import permission_store as store
from aura.infrastructure.persistence import journal
from aura.tools.ask_user import UserAsker
from cli import repl
from cli._non_interactive_askers import (
    PRINT_MODE_PERMISSION_FEEDBACK,
    PRINT_MODE_USER_QUESTION_ERROR,
    make_non_interactive_permission_asker,
    make_non_interactive_user_asker,
)
from cli._permission_asker import make_cli_asker, print_bypass_banner
from cli._user_asker import make_cli_user_asker
from cli.mcp_cli import handle_mcp

AgentRef: TypeAlias = AgentSession | None


class _PrintStreamAgent(Protocol):
    def astream(self, prompt: str) -> AsyncIterator[object]: ...


_MODES: tuple[Mode, ...] = ("default", "bypass", "plan", "accept_edits")


def _force_utf8_streams() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aura",
        description="A general-purpose Python agent.",
    )
    parser.add_argument("--version", action="version", version=f"aura {__version__}")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose output")
    parser.add_argument(
        "--log", action="store_true",
        help="write event log to ~/.aura/logs/events.jsonl",
    )
    parser.add_argument(
        "--bypass-permissions",
        action="store_true",
        help="Disable permission prompts — every tool call allowed without asking. "
             "Dangerous; prefer per-rule allows in .aura/settings.json.",
    )
    parser.add_argument(
        "-p", "--print",
        dest="print_prompt",
        metavar="PROMPT",
        help="Run one prompt in a fresh non-interactive session and print only the final answer.",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    mcp = subparsers.add_parser(
        "mcp",
        help="manage MCP servers (~/.aura/mcp_servers.json + project overrides)",
        description="Manage MCP (Model Context Protocol) server entries.",
    )
    mcp_sub = mcp.add_subparsers(dest="mcp_action")

    mcp_add = mcp_sub.add_parser(
        "add",
        help="add an MCP server",
        description=(
            "Add an MCP server entry.\n\n"
            "Examples:\n"
            "  aura mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem /tmp\n"
            "  aura mcp add -e API_KEY=xxx my-server -- my-mcp-server\n"
            "  aura mcp add --transport sse sentry -- https://mcp.sentry.dev/mcp\n"
            "  aura mcp add --scope project repo-tools -- ./scripts/mcp-server.js\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mcp_add.add_argument("name", help="server name (namespaces tools as mcp__<name>__<tool>)")
    mcp_add.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable_http"],
        default="stdio",
        help="transport type (default: stdio)",
    )
    mcp_add.add_argument(
        "--scope",
        choices=["global", "project"],
        default="global",
        help=(
            "layer to write to: 'global' (~/.aura/mcp_servers.json) or "
            "'project' (<cwd>/.aura/mcp_servers.json)"
        ),
    )
    mcp_add.add_argument(
        "--env", "-e",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment variable for stdio transport (repeatable)",
    )
    # argparse.REMAINDER can't intermix with --transport/--env; populated by _split_dashdash.
    mcp_add.set_defaults(command_args=[])

    mcp_sub.add_parser(
        "list",
        help="list configured MCP servers (merged across scopes)",
    )

    mcp_remove = mcp_sub.add_parser("remove", help="remove an MCP server by name")
    mcp_remove.add_argument("name", help="server name to remove")
    mcp_remove.add_argument(
        "--scope",
        choices=["auto", "global", "project"],
        default="auto",
        help="layer to remove from; 'auto' targets whichever currently owns the name",
    )

    # Teammates talk to the leader via on-disk JSONL mailbox; argv only seeds AgentSession.
    teammate = subparsers.add_parser(
        "teammate",
        help="run an Aura teammate inside a subprocess (pane backend)",
        description="Subprocess entrypoint for pane-backed teammates.",
    )
    teammate.add_argument("--team-id", required=True)
    teammate.add_argument("--member", required=True)
    teammate.add_argument("--storage-root", required=True)
    teammate.add_argument("--agent-type", default="general-purpose")
    teammate.add_argument("--model", default=None)
    teammate.add_argument("--system-prompt", default=None)
    teammate.add_argument("--seed-prompt", default=None)

    return parser


def _resolve_mode(
    args: argparse.Namespace, perm_cfg: PermissionsConfig,
) -> Mode:
    if args.bypass_permissions:
        return "bypass"
    return perm_cfg.mode


def _bypass_refused_message() -> str:
    return (
        "error: --bypass-permissions is disabled by config "
        "(permissions.disable_bypass=true). Remove the flag or update "
        "permissions settings."
    )


def _warn_plaintext_api_keys(
    config: AuraConfig, console: Console, *, verbose: bool = False,
) -> None:
    # Always journal so audit trail survives even when --verbose suppresses the print.
    for provider in config.providers:
        if provider.api_key:
            journal.write("plaintext_api_key_warning", provider=provider.name)
            if verbose:
                console.print(
                    f"[yellow]Warning: provider {provider.name!r} uses a plaintext "
                    f"api_key in config. Prefer api_key_env for security.[/yellow]"
                )


def _fail_startup(console: Console, exc: BaseException) -> int:
    if isinstance(exc, AuraError):
        journal.write("startup_failed", reason=type(exc).__name__, detail=str(exc))
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
    else:
        journal.write("startup_failed", reason="unexpected", detail=str(exc))
        console.print(f"[red]startup error: {exc}[/red]")
    return 2


def _split_dashdash(argv: list[str]) -> tuple[list[str], list[str]]:
    try:
        idx = argv.index("--")
    except ValueError:
        return argv, []
    return argv[:idx], argv[idx + 1 :]


def _stderr_console() -> Console:
    return Console(file=sys.stderr)


def _fresh_print_session_id() -> str:
    return f"print-{uuid4().hex}"


def _make_live_mode_cell(mode: Mode) -> tuple[list[AgentRef], Callable[[], Mode]]:
    _agent_cell: list[AgentRef] = [None]

    def _live_mode() -> Mode:
        a = _agent_cell[0]
        if a is None:
            return mode
        live = a.mode
        for candidate in _MODES:
            if live == candidate:
                return candidate
        return mode

    return _agent_cell, _live_mode


def _build_cli_agent(
    *,
    config: AuraConfig,
    args: argparse.Namespace,
    mode: Mode,
    perm_cfg: PermissionsConfig,
    ruleset: RuleSet,
    deny_rules: RuleSet,
    ask_rules: RuleSet,
    safety_policy: SafetyPolicy,
    project_root: Path,
    permission_asker: PermissionAsker,
    question_asker: UserAsker,
    session_id: str,
) -> AgentSession:
    session = SessionRuleSet()
    agent_cell, live_mode = _make_live_mode_cell(mode)
    hooks = HookChain(
        pre_tool=[
            make_permission_hook(
                asker=permission_asker,
                session=session,
                rules=ruleset,
                deny_rules=deny_rules,
                ask_rules=ask_rules,
                project_root=project_root,
                mode=live_mode,
                safety=safety_policy,
            ),
        ],
    )
    if args.log or config.log.enabled:
        hooks = wrap_with_event_logger(hooks)
    agent = build_agent(
        config,
        hooks=hooks,
        session_id=session_id,
        session_rules=session,
        question_asker=question_asker,
        mode=mode,
        disable_bypass=perm_cfg.disable_bypass,
        ruleset=ruleset,
        deny_ruleset=deny_rules,
        ask_ruleset=ask_rules,
        safety=safety_policy,
    )
    agent_cell[0] = agent
    return agent


async def _run_print_mode(agent: AgentSession | _PrintStreamAgent, prompt: str) -> str:
    if prompt.startswith("/"):
        if not isinstance(agent, AgentSession):
            raise TypeError("slash commands in print mode require a real AgentSession")
        registry = build_default_registry(agent=agent)
        command = await dispatch(prompt, agent, registry)
        if command.handled:
            if command.kind == "view":
                return command.text.strip("\n")
            if command.kind == "print":
                return command.text
            if command.kind == "exit":
                return ""
            return ""

    parts: list[str] = []
    final_message = ""
    saw_error = False
    async for event in agent.astream(prompt):
        if isinstance(event, AssistantDelta):
            parts.append(event.text)
        elif isinstance(event, Final):
            final_message = event.message
        elif isinstance(event, ToolCallCompleted) and event.error:
            saw_error = True
            if PRINT_MODE_PERMISSION_FEEDBACK in event.error:
                raise ToolError(
                    "print mode cannot satisfy this tool's permission prompt; "
                    "rerun in the REPL or add an allow rule"
                )
            if PRINT_MODE_USER_QUESTION_ERROR in event.error:
                raise ToolError(PRINT_MODE_USER_QUESTION_ERROR)
            raise ToolError(event.error)
    text = "".join(parts)
    if text.strip():
        return text
    if final_message and (not saw_error or final_message.strip()):
        return final_message
    return ""


def run_as_teammate(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(
            run_teammate_main(
                team_id=args.team_id,
                member_name=args.member,
                storage_root=args.storage_root,
                agent_type=args.agent_type,
                model_name=args.model,
                system_prompt=args.system_prompt,
                seed_prompt=args.seed_prompt,
            ),
        )
    except KeyboardInterrupt:
        return 130


def main() -> int:
    _force_utf8_streams()
    parser = _make_parser()

    raw_argv = sys.argv[1:]
    pre, post = _split_dashdash(raw_argv)
    args = parser.parse_args(pre)
    if args.print_prompt is not None and args.subcommand is not None:
        print("error: -p/--print cannot be combined with a subcommand", file=sys.stderr)
        return 2
    if args.subcommand == "mcp" and getattr(args, "mcp_action", None) == "add":
        args.command_args = post

    if args.subcommand == "mcp":
        return handle_mcp(args)

    if args.subcommand == "teammate":
        return run_as_teammate(args)

    print_mode = args.print_prompt is not None
    console = _stderr_console() if print_mode else Console()

    try:
        journal.write("config_load_attempt")
        config = load_config()
        journal.write(
            "config_loaded",
            providers=[p.name for p in config.providers],
            default_spec=config.router.get("default", ""),
        )
    except Exception as exc:  # noqa: BLE001  # log + swallow; logging path must never crash caller
        return _fail_startup(console, exc)

    if args.log or config.log.enabled:
        log_path = Path(config.log.path).expanduser()
        journal.configure(log_path)
        console.print(f"[dim]event log: {log_path}[/dim]")

    journal.write(
        "startup",
        version=__version__,
        verbose=args.verbose,
        log_enabled=bool(args.log or config.log.enabled),
    )

    try:
        _warn_plaintext_api_keys(config, console, verbose=args.verbose)
        project_root = Path.cwd()
        local_path, created = store.ensure_local_settings(project_root)
        if created and not print_mode:
            console.print(
                f"[dim]created {local_path} — machine-local permission "
                "overrides go here (gitignored)[/dim]"
            )
        try:
            perm_cfg = store.load(project_root)
            known_tools = list(config.tools.enabled) + ["mcp__*"]
            disk_rules = store.load_ruleset(
                project_root, known_tool_names=known_tools,
            )
            deny_rules = store.load_deny_ruleset(project_root)
            ask_rules = store.load_ask_ruleset(project_root)
        except AuraConfigError as exc:
            return _fail_startup(console, exc)
        # User rules first so audit credits them, not the default backstop.
        ruleset = RuleSet(rules=disk_rules.rules + DEFAULT_ALLOW_RULES)
        safety_policy = SafetyPolicy(
            protected_writes=DEFAULT_PROTECTED_WRITES,
            protected_reads=DEFAULT_PROTECTED_READS,
            exempt=tuple(perm_cfg.safety_exempt),
        )
        mode = _resolve_mode(args, perm_cfg)
        if args.bypass_permissions and perm_cfg.disable_bypass:
            print(_bypass_refused_message(), file=sys.stderr)
            journal.write(
                "bypass_refused",
                reason="disable_bypass",
                source="cli_flag",
            )
            return 2
        if mode == "bypass":
            print_bypass_banner(console)
            journal.write("permission_bypass_active")
        if print_mode:
            permission_asker = make_non_interactive_permission_asker()
            question_asker = make_non_interactive_user_asker()
            session_id = _fresh_print_session_id()
        else:
            permission_asker = make_cli_asker(timeout=perm_cfg.prompt_timeout_sec)
            question_asker = make_cli_user_asker(
                timeout=perm_cfg.prompt_timeout_sec,
            )
            session_id = "default"
        agent = _build_cli_agent(
            config=config,
            args=args,
            mode=mode,
            perm_cfg=perm_cfg,
            ruleset=ruleset,
            deny_rules=deny_rules,
            ask_rules=ask_rules,
            safety_policy=safety_policy,
            project_root=project_root,
            permission_asker=permission_asker,
            question_asker=question_asker,
            session_id=session_id,
        )
        journal.write("agent_built")
    except Exception as exc:  # noqa: BLE001  # log + swallow; logging path must never crash caller
        return _fail_startup(console, exc)

    async def _entry() -> int:
        try:
            await agent.aconnect()
        except Exception as exc:  # noqa: BLE001  # log + swallow; logging path must never crash caller
            console.print(f"[yellow]mcp connect error (continuing): {exc}[/yellow]")
            journal.write("mcp_connect_cli_error", error=str(exc))
        watcher = None
        if not print_mode:
            watcher = FileWatcher(
                paths=default_watch_paths(Path.cwd()),
                chain=agent.hooks,
                state=agent.state,
            )
            try:
                await watcher.start()
            except Exception as exc:  # noqa: BLE001  # log + swallow; logging path must never crash caller
                journal.write(
                    "file_watcher_start_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
        try:
            if print_mode:
                try:
                    text = await _run_print_mode(agent, args.print_prompt or "")
                except ToolError as exc:
                    print(str(exc), file=sys.stderr)
                    return 2
                if text:
                    print(text)
                return 0
            await repl.run_repl_async(
                agent, console=console, verbose=args.verbose,
                bypass=(mode == "bypass"),
            )
            return 0
        finally:
            if watcher is not None:
                await watcher.stop()
            if print_mode:
                agent.storage.clear(agent.session_id)
            await agent.aclose()

    try:
        return asyncio.run(_entry())
    except KeyboardInterrupt:
        console.print()
        journal.write("shutdown_sigint")
        return 130
    finally:
        agent.close()
        journal.write("shutdown")


if __name__ == "__main__":
    sys.exit(main())
