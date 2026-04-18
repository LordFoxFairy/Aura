"""aura CLI entry — argparse + build_agent + run_repl."""

from __future__ import annotations

import argparse
import asyncio
import sys

from aura import __version__


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aura",
        description="A general-purpose Python agent.",
    )
    parser.add_argument("--version", action="version", version=f"aura {__version__}")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="verbose output",
    )
    return parser


def main() -> int:
    parser = _make_parser()
    parser.parse_args()

    from rich.console import Console

    from aura.cli.permission import make_cli_asker
    from aura.cli.repl import run_repl_async
    from aura.config.loader import load_config
    from aura.config.schema import AuraConfigError
    from aura.core.agent import build_agent
    from aura.core.hooks import HookChain
    from aura.core.permission import PermissionSession, make_permission_hook

    console = Console()

    try:
        config = load_config()
        session = PermissionSession()
        asker = make_cli_asker(session)
        hooks = HookChain(pre_tool=[make_permission_hook(asker=asker, session=session)])
        agent = build_agent(config, hooks=hooks)
    except AuraConfigError as exc:
        console.print(f"[red]config error: {exc}[/red]")
        return 2
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]startup error: {exc}[/red]")
        return 2

    try:
        asyncio.run(run_repl_async(agent, console=console))
    finally:
        agent.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
