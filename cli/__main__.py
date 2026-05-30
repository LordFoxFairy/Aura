"""CLI entry point: parse args, load config, build the agent, and run the REPL."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

from aura import __version__

if TYPE_CHECKING:
    from rich.console import Console

    from aura.config.schema import AuraConfig, PermissionsConfig
    from aura.core.agent import Agent
    from aura.domain.permission.mode import Mode


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

    # Teammates talk to the leader via on-disk JSONL mailbox; argv only seeds Agent.
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
    from aura.core import journal

    for provider in config.providers:
        if provider.api_key:
            journal.write("plaintext_api_key_warning", provider=provider.name)
            if verbose:
                console.print(
                    f"[yellow]Warning: provider {provider.name!r} uses a plaintext "
                    f"api_key in config. Prefer api_key_env for security.[/yellow]"
                )


def _fail_startup(console: Console, exc: BaseException) -> int:
    from aura.core import journal
    from aura.domain.errors import AuraError

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


def run_as_teammate(args: argparse.Namespace) -> int:
    # Lazy import keeps the parent ``aura`` invocation light.
    from aura.application.teams.runtime import run_teammate_main

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
    if args.subcommand == "mcp" and getattr(args, "mcp_action", None) == "add":
        args.command_args = post

    if args.subcommand == "mcp":
        from cli.mcp_cli import handle_mcp

        return handle_mcp(args)

    if args.subcommand == "teammate":
        return run_as_teammate(args)

    from pathlib import Path

    from rich.console import Console

    from aura.application.hooks import HookChain
    from aura.application.hooks.logging import wrap_with_event_logger
    from aura.application.hooks.permission import make_permission_hook
    from aura.config.loader import load_config
    from aura.config.schema import AuraConfigError
    from aura.core import journal
    from aura.core.agent import build_agent
    from aura.domain.permission.defaults import DEFAULT_ALLOW_RULES
    from aura.domain.permission.safety import (
        DEFAULT_PROTECTED_READS,
        DEFAULT_PROTECTED_WRITES,
        SafetyPolicy,
    )
    from aura.domain.permission.session import RuleSet, SessionRuleSet
    from aura.infrastructure import permission_store as store
    from cli._permission_asker import make_cli_asker, print_bypass_banner
    from cli._user_asker import make_cli_user_asker
    from cli.repl import run_repl_async

    console = Console()

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
        if created:
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
        session = SessionRuleSet()
        asker = make_cli_asker(timeout=perm_cfg.prompt_timeout_sec)
        # Forward-ref cell — hook reads Agent.mode live so shift+tab toggles propagate.
        _agent_cell: list[Agent | None] = [None]

        def _live_mode() -> Mode:
            from typing import cast
            a = _agent_cell[0]
            if a is None:
                return mode
            return cast("Mode", a.mode)

        hooks = HookChain(
            pre_tool=[
                make_permission_hook(
                    asker=asker,
                    session=session,
                    rules=ruleset,
                    deny_rules=deny_rules,
                    ask_rules=ask_rules,
                    project_root=project_root,
                    mode=_live_mode,
                    safety=safety_policy,
                ),
            ],
        )
        if args.log or config.log.enabled:
            hooks = wrap_with_event_logger(hooks)
        agent = build_agent(
            config,
            hooks=hooks,
            session_rules=session,
            question_asker=make_cli_user_asker(
                timeout=perm_cfg.prompt_timeout_sec,
            ),
            mode=mode,
            disable_bypass=perm_cfg.disable_bypass,
            ruleset=ruleset,
            deny_ruleset=deny_rules,
            ask_ruleset=ask_rules,
            safety=safety_policy,
        )
        _agent_cell[0] = agent
        journal.write("agent_built")
    except Exception as exc:  # noqa: BLE001  # log + swallow; logging path must never crash caller
        return _fail_startup(console, exc)

    async def _entry() -> None:
        try:
            await agent.aconnect()
        except Exception as exc:  # noqa: BLE001  # log + swallow; logging path must never crash caller
            console.print(f"[yellow]mcp connect error (continuing): {exc}[/yellow]")
            journal.write("mcp_connect_cli_error", error=str(exc))
        from aura.application.hooks.file_watcher import FileWatcher, default_watch_paths
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
            await run_repl_async(
                agent, console=console, verbose=args.verbose,
                bypass=(mode == "bypass"),
            )
        finally:
            await watcher.stop()
            await agent.aclose()

    try:
        asyncio.run(_entry())
    except KeyboardInterrupt:
        console.print()
        journal.write("shutdown_sigint")
        return 130
    finally:
        agent.close()
        journal.write("shutdown")

    return 0


if __name__ == "__main__":
    sys.exit(main())
