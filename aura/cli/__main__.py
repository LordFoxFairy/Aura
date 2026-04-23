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


def main() -> int:
    _force_utf8_streams()
    parser = _make_parser()
    args = parser.parse_args()

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
        if mode == "bypass":
            print_bypass_banner(console)
            journal.write("permission_bypass_active")
        session = SessionRuleSet()
        asker = make_cli_asker()
        hooks = HookChain(
            pre_tool=[
                make_permission_hook(
                    asker=asker,
                    session=session,
                    rules=ruleset,
                    project_root=project_root,
                    mode=mode,
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
            question_asker=make_cli_user_asker(),
            # Hand the resolved mode in so the bottom bar + any future
            # agent-level consumer can read it without re-consulting the
            # permission store.
            mode=mode,
        )
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
