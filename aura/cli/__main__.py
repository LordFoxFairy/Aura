"""CLI entry point: parse args, load config, build the agent, and run the REPL."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

from aura import __version__

if TYPE_CHECKING:
    from rich.console import Console

    from aura.config.schema import AuraConfig
    from aura.core.agent import Agent
    from aura.core.permissions.mode import Mode
    from aura.schemas.permissions import PermissionsConfig


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
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="verbose output",
    )
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

    # Subcommands. ``subcommand=None`` (the default) keeps the existing
    # "run REPL" behaviour — every global flag above still works when no
    # subcommand is given. A subcommand short-circuits to its own handler
    # before main() ever builds the agent.
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
            "which layer to write to: 'global' (~/.aura/mcp_servers.json, "
            "the default — available in every project) or 'project' "
            "(<cwd>/.aura/mcp_servers.json, travels with the repo and "
            "overrides global on name collisions)"
        ),
    )
    mcp_add.add_argument(
        "--env", "-e",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment variable for stdio transport (repeatable)",
    )
    # NOTE: ``command_args`` is populated by ``_split_dashdash`` in main()
    # BEFORE argparse parses the rest. We can't use ``argparse.REMAINDER``
    # here because it gobbles greedily once the first positional (``name``)
    # is consumed — meaning ``aura mcp add foo --transport stdio -- cmd``
    # would stuff ``--transport stdio -- cmd`` into REMAINDER instead of
    # letting ``--transport`` parse normally. We register the attribute
    # with a default so downstream code can always read ``args.command_args``.
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
        help=(
            "which layer to remove from. 'auto' (default) targets "
            "whichever layer currently owns the name — project wins on "
            "collision, so 'auto' removes the entry the user actually "
            "sees. Pass an explicit scope to remove a shadowed entry."
        ),
    )

    return parser


def _resolve_mode(
    args: argparse.Namespace, perm_cfg: PermissionsConfig,
) -> Mode:
    """Pick the active permission mode.

    Precedence (highest first):
    1. ``--bypass-permissions`` CLI flag
    2. ``permissions.mode`` in ``.aura/settings.json`` (or local override)
    3. Built-in default ``"default"`` (via PermissionsConfig.mode default)

    No double-track: the mode comes SOLELY from the permission config
    loaded by ``store.load`` (which already merges project + local), never
    from ``AuraConfig`` — that file is provider/router/storage only.
    """
    if args.bypass_permissions:
        return "bypass"
    return perm_cfg.mode


def _bypass_refused_message() -> str:
    """Error message shown when ``--bypass-permissions`` hits the
    ``disable_bypass`` kill switch.

    Factored out so the CLI check and the programmatic (Agent-level)
    guard share the exact same user-facing wording — an operator grepping
    logs for either surface finds the same string.
    """
    return (
        "error: --bypass-permissions is disabled by config "
        "(permissions.disable_bypass=true). Remove the flag or update "
        "permissions settings."
    )


def _warn_plaintext_api_keys(
    config: AuraConfig, console: Console, *, verbose: bool = False,
) -> None:
    """Audit plaintext ``api_key`` usage in config.

    Journal entry fires ALWAYS so a security audit has full visibility.
    The console warning only prints under ``--verbose`` — spamming it on
    every startup taught operators to tune it out, which defeats the
    purpose. Operators who care read the journal; casual users aren't
    nagged every session.
    """
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
    # AuraError carries a user-facing source/detail message; anything else is
    # reported as "unexpected" so we never surface a raw stacktrace to the user.
    from aura.core import journal
    from aura.errors import AuraError

    if isinstance(exc, AuraError):
        journal.write("startup_failed", reason=type(exc).__name__, detail=str(exc))
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
    else:
        journal.write("startup_failed", reason="unexpected", detail=str(exc))
        console.print(f"[red]startup error: {exc}[/red]")
    return 2


def _split_dashdash(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``argv`` on the first literal ``--`` token.

    ``claude mcp add <name> -- <command> [args...]`` uses ``--`` as a
    shell-safe separator between flags and the pass-through command. We
    honour that convention by peeling the trailing ``--`` + remainder off
    before argparse sees the list — argparse's own ``--`` handling is
    positional-only and doesn't play well with ``REMAINDER`` once there
    are optional flags (``--transport``, ``--env``) mixed in.
    """
    try:
        idx = argv.index("--")
    except ValueError:
        return argv, []
    return argv[:idx], argv[idx + 1 :]


def main() -> int:
    _force_utf8_streams()
    parser = _make_parser()

    raw_argv = sys.argv[1:]
    pre, post = _split_dashdash(raw_argv)
    args = parser.parse_args(pre)
    # Attach the post-`--` tokens to the namespace so the mcp add handler
    # sees them regardless of whether argparse registered ``command_args``.
    if args.subcommand == "mcp" and getattr(args, "mcp_action", None) == "add":
        args.command_args = post

    # Subcommand fast-path: pure file ops, no agent / LLM / REPL. Claude-code
    # parity — ``claude mcp add`` doesn't spin up the chat UI, and neither
    # does ``aura mcp add``. Dispatch here returns an exit code directly so
    # we never touch ``load_config`` or ``build_agent``.
    if args.subcommand == "mcp":
        from aura.cli.mcp_cli import handle_mcp

        return handle_mcp(args)

    from pathlib import Path

    from rich.console import Console

    from aura.cli.permission import make_cli_asker, print_bypass_banner
    from aura.cli.repl import run_repl_async
    from aura.cli.user_question import make_cli_user_asker
    from aura.config.loader import load_config
    from aura.config.schema import AuraConfigError
    from aura.core import journal
    from aura.core.agent import build_agent
    from aura.core.hooks import HookChain
    from aura.core.hooks.logging import wrap_with_event_logger
    from aura.core.hooks.permission import make_permission_hook
    from aura.core.permissions import store
    from aura.core.permissions.defaults import DEFAULT_ALLOW_RULES
    from aura.core.permissions.safety import (
        DEFAULT_PROTECTED_READS,
        DEFAULT_PROTECTED_WRITES,
        SafetyPolicy,
    )
    from aura.core.permissions.session import RuleSet, SessionRuleSet

    console = Console()

    try:
        journal.write("config_load_attempt")
        config = load_config()
        journal.write(
            "config_loaded",
            providers=[p.name for p in config.providers],
            default_spec=config.router.get("default", ""),
        )
    except Exception as exc:  # noqa: BLE001
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
            # Load the full PermissionsConfig (mode + allow + safety_exempt),
            # not just the RuleSet — we need ``safety_exempt`` to compose a
            # user-specific SafetyPolicy. Previously the CLI only called
            # ``load_ruleset`` and dropped ``safety_exempt`` on the floor.
            perm_cfg = store.load(project_root)
            disk_rules = store.load_ruleset(project_root)
        except AuraConfigError as exc:
            return _fail_startup(console, exc)
        # User rules first, built-in defaults last. Both are allow-only so
        # the DECISION is the same regardless of order — what changes is
        # which rule gets reported in the ``permission_decision`` journal
        # event. A user who wrote ``"read_file(/tmp/specific)"`` in their
        # settings.json deserves to see THEIR rule in the audit, not a
        # generic default tool-wide. Defaults act as the backstop when no
        # user rule matches.
        ruleset = RuleSet(rules=disk_rules.rules + DEFAULT_ALLOW_RULES)
        # SafetyPolicy composed from defaults + user exempt list. Without
        # this, ``permissions.safety_exempt`` in settings.json would be
        # parsed and ignored.
        safety_policy = SafetyPolicy(
            protected_writes=DEFAULT_PROTECTED_WRITES,
            protected_reads=DEFAULT_PROTECTED_READS,
            exempt=tuple(perm_cfg.safety_exempt),
        )
        mode = _resolve_mode(args, perm_cfg)
        # Kill switch — enforced at the SAME layer as the CLI flag
        # decision, before any agent-level state spins up, so that an
        # operator can never enter bypass mode via any path when the
        # org config forbids it. Exits with code 2 (config error) to
        # distinguish from code 1 (runtime error) and 130 (SIGINT).
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
        # Prompt timeout: plumb ``PermissionsConfig.prompt_timeout_sec``
        # (default 300s = 5 min, ``None`` = wait forever) through to both
        # the permission asker and the ask_user_question asker. Fail-safe
        # on unattended / stale sessions — a hung prompt would otherwise
        # block the whole turn forever.
        asker = make_cli_asker(timeout=perm_cfg.prompt_timeout_sec)
        # Forward-ref cell: the permission hook needs to read Agent's LIVE
        # mode (Agent.set_mode is called mid-session by shift+tab /
        # enter_plan_mode / exit_plan_mode). Closing over the startup
        # ``mode`` value kept the hook enforcing the stale mode — integration
        # tests surfaced this. The hook now takes a Callable; we fill the
        # cell right after ``build_agent`` returns so every hook invocation
        # reads ``agent.mode`` fresh.
        _agent_cell: list[Agent | None] = [None]

        def _live_mode() -> Mode:
            # ``Agent.mode`` is typed ``str`` (enum would create a circular
            # dep between Agent and permissions.mode); narrow here since
            # every writer (set_mode validator + CLI resolve) guarantees a
            # valid Mode literal. Cast is cheaper than threading a Literal
            # through Agent's public surface.
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
            # Hand the resolved mode in so the bottom bar + any future
            # agent-level consumer can read it without re-consulting the
            # permission store.
            mode=mode,
            # Propagate disable_bypass to the Agent layer so a
            # programmatic ``set_mode("bypass")`` can't sneak past the
            # CLI-level guard (same enforcement layer, different entry
            # point).
            disable_bypass=perm_cfg.disable_bypass,
        )
        _agent_cell[0] = agent
        journal.write("agent_built")
    except Exception as exc:  # noqa: BLE001
        return _fail_startup(console, exc)

    async def _entry() -> None:
        # aconnect must run inside the same event loop as the REPL so the
        # underlying MultiServerMCPClient's stdio sessions stay bound to a
        # single loop. Never re-raise — graceful degradation is the whole
        # point of the wrapping in Agent.aconnect itself; this extra try
        # is belt-and-braces for future refactors.
        try:
            await agent.aconnect()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]mcp connect error (continuing): {exc}[/yellow]")
            journal.write("mcp_connect_cli_error", error=str(exc))
        await run_repl_async(
            agent, console=console, verbose=args.verbose,
            bypass=(mode == "bypass"),
            statusline=perm_cfg.statusline,
        )

    try:
        asyncio.run(_entry())
    except KeyboardInterrupt:
        # asyncio.run swallows the first SIGINT (to cancel the running task);
        # only a second Ctrl+C propagates here. Convert it to a clean exit so
        # the user never sees a Python traceback.
        console.print()
        journal.write("shutdown_sigint")
        return 130
    finally:
        agent.close()
        journal.write("shutdown")

    return 0


if __name__ == "__main__":
    sys.exit(main())
