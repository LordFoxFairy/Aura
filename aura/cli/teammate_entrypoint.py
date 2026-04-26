"""Subprocess entrypoint for pane-backed teammates.

Invoked by :class:`~aura.core.teams.backends.pane.PaneBackend` via
``python -m aura.cli.teammate_entrypoint`` inside a freshly-split tmux
pane. Builds a minimal Aura runtime and drives
:func:`~aura.core.teams.runtime.run_teammate_main`, then exits when the
leader posts ``shutdown_request`` to this teammate's mailbox.

CLI shape (kept tiny so ``shlex.join`` from the parent stays readable):

.. code-block:: text

    python -m aura.cli.teammate_entrypoint \\
        --team-id <id> \\
        --member <name> \\
        --storage-root <path> \\
        --agent-type general-purpose \\
        [--model openai:gpt-4o] \\
        [--system-prompt "..."] \\
        [--seed-prompt "first task..."]

Exit codes:

- ``0`` — clean shutdown after consuming a ``shutdown_request``.
- ``2`` — argparse / startup failure (config load, agent build, etc.).
- ``130`` — KeyboardInterrupt; same convention as the main CLI.

Communication with the leader is exclusively via the on-disk JSONL
mailbox, so the subprocess needs no IPC channel back to the parent.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aura.cli.teammate_entrypoint",
        description="Run an Aura teammate inside a tmux pane subprocess.",
    )
    parser.add_argument(
        "--team-id",
        required=True,
        help="team_id slug the teammate belongs to (matches "
             "<storage_root>/teams/<team_id>/)",
    )
    parser.add_argument(
        "--member",
        required=True,
        help="this teammate's member name (must match the manager's "
             "TeammateMember.name)",
    )
    parser.add_argument(
        "--storage-root",
        required=True,
        help="directory containing the leader's index.sqlite + teams/ "
             "subtree (typically ~/.aura)",
    )
    parser.add_argument(
        "--agent-type",
        default="general-purpose",
        help="agent flavor passed through to build_agent (default: "
             "general-purpose)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model spec override (e.g. 'openai:gpt-4o'); inherits the "
             "leader's router default when omitted",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="optional per-teammate system prompt override",
    )
    parser.add_argument(
        "--seed-prompt",
        default=None,
        help="first message the teammate consumes (skips the mailbox); "
             "matches add_member(seed_prompt=...)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry — parse args, drive ``run_teammate_main``, return exit code.

    Wraps the async runtime in :func:`asyncio.run` so the subprocess
    can be invoked the same way the leader's pane backend launches it
    (``python -m aura.cli.teammate_entrypoint ...``).
    """
    parser = _make_parser()
    args = parser.parse_args(argv)
    # Lazy import: the entrypoint module is imported by the test suite
    # for ``--help`` smoke testing; defer the heavy chain (config loader,
    # agent builder, persistence) to actual execution.
    from aura.core.teams.runtime import run_teammate_main
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


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess
    sys.exit(main())
