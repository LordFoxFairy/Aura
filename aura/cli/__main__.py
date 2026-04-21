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


def _resolve_mode(args: argparse.Namespace, config: AuraConfig) -> Mode:
    # CLI flag wins over config; config.permissions.mode wins over built-in default.
    if args.bypass_permissions:
        return "bypass"
    if config.permissions is not None and config.permissions.mode:
        return config.permissions.mode
    return "default"


def _warn_plaintext_api_keys(config: AuraConfig, console: Console) -> None:
    for provider in config.providers:
        if provider.api_key:
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
    from aura.config.loader import load_config
    from aura.config.schema import AuraConfigError
    from aura.core import journal
    from aura.core.agent import build_agent
    from aura.core.hooks import HookChain
    from aura.core.hooks.logging import wrap_with_event_logger
    from aura.core.hooks.permission import make_permission_hook
    from aura.core.permissions import store
    from aura.core.permissions.defaults import DEFAULT_ALLOW_RULES
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
        _warn_plaintext_api_keys(config, console)
        project_root = Path.cwd()
        local_path, created = store.ensure_local_settings(project_root)
        if created:
            console.print(
                f"[dim]created {local_path} — machine-local permission "
                "overrides go here (gitignored)[/dim]"
            )
        try:
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
        mode = _resolve_mode(args, config)
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
                ),
            ],
        )
        if args.log or config.log.enabled:
            hooks = wrap_with_event_logger(hooks)
        agent = build_agent(config, hooks=hooks, session_rules=session)
        journal.write("agent_built")
    except Exception as exc:  # noqa: BLE001
        return _fail_startup(console, exc)

    try:
        asyncio.run(run_repl_async(agent, console=console, verbose=args.verbose))
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
