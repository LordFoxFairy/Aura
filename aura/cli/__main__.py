"""aura CLI entry — argparse + build_agent + run_repl."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

from aura import __version__

if TYPE_CHECKING:
    from rich.console import Console

    from aura.config.schema import AuraConfig


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
    return parser


def _warn_plaintext_api_keys(config: AuraConfig, console: Console) -> None:
    for provider in config.providers:
        if provider.api_key:
            console.print(
                f"[yellow]Warning: provider {provider.name!r} uses a plaintext "
                f"api_key in config. Prefer api_key_env for security.[/yellow]"
            )


def main() -> int:
    _force_utf8_streams()
    parser = _make_parser()
    args = parser.parse_args()

    from pathlib import Path

    from rich.console import Console

    from aura.cli.permission import make_cli_asker
    from aura.cli.repl import run_repl_async
    from aura.config.loader import load_config
    from aura.config.schema import AuraConfigError
    from aura.core.agent import build_agent
    from aura.core.hooks import HookChain
    from aura.core.journal import journal, setup_file_journal
    from aura.core.logging import wrap_with_event_logger
    from aura.core.permission import PermissionSession, make_permission_hook

    console = Console()

    if args.log:
        log_path = Path.home() / ".aura" / "logs" / "events.jsonl"
        setup_file_journal(log_path)
        console.print(f"[dim]event log: {log_path}[/dim]")

    journal().write(
        "startup",
        version=__version__,
        verbose=args.verbose,
        log_enabled=bool(args.log),
    )

    try:
        journal().write("config_load_attempt")
        config = load_config()
        journal().write(
            "config_loaded",
            providers=[p.name for p in config.providers],
            default_spec=config.router.get("default", ""),
        )
        _warn_plaintext_api_keys(config, console)
        session = PermissionSession()
        asker = make_cli_asker(session)
        hooks = HookChain(
            pre_tool=[make_permission_hook(asker=asker, session=session)],
        )
        if args.log:
            hooks = wrap_with_event_logger(hooks)
        agent = build_agent(config, hooks=hooks)
        journal().write("agent_built")
    except AuraConfigError as exc:
        journal().write("startup_failed", reason="config", detail=str(exc))
        console.print(f"[red]config error: {exc}[/red]")
        return 2
    except Exception as exc:  # noqa: BLE001
        journal().write("startup_failed", reason="unexpected", detail=str(exc))
        console.print(f"[red]startup error: {exc}[/red]")
        return 2

    try:
        asyncio.run(run_repl_async(agent, console=console, verbose=args.verbose))
    finally:
        agent.close()
        journal().write("shutdown")

    return 0


if __name__ == "__main__":
    sys.exit(main())
