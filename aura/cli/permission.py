"""Interactive y/N/a permission asker for the CLI.

Returns ``AskerResponse`` — the hook decides what to do with the choice;
this module only owns the user I/O.

Task 8 replaces this with a ``prompt_toolkit`` list-select UI + the
§8.2 smart rule-hint derivation. For today, keep the 3-choice text prompt.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool

from aura.cli.render import compact_args
from aura.core.hooks.permission import AskerResponse, PermissionAsker
from aura.core.permissions.rule import Rule
from aura.core.persistence import journal


def make_cli_asker() -> PermissionAsker:
    async def _ask(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,
    ) -> AskerResponse:
        args_preview = compact_args(args)
        journal.write(
            "permission_asked",
            tool=tool.name,
            is_destructive=(tool.metadata or {}).get("is_destructive", False),
            args_preview=args_preview,
        )
        prompt = (
            f"\n  tool: {tool.name}({args_preview})\n"
            "  allow this call?  [y]es / [N]o / [a]lways: "
        )
        try:
            answer = await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            journal.write(
                "permission_answered", tool=tool.name, answer="eof", allowed=False,
            )
            return AskerResponse(choice="deny")

        answer = answer.strip().lower()
        if answer == "a":
            journal.write(
                "permission_answered", tool=tool.name, answer="a", allowed=True,
            )
            return AskerResponse(choice="always", scope="session", rule=rule_hint)
        if answer == "y":
            journal.write(
                "permission_answered", tool=tool.name, answer="y", allowed=True,
            )
            return AskerResponse(choice="accept")
        journal.write(
            "permission_answered",
            tool=tool.name,
            answer=answer or "(empty)",
            allowed=False,
        )
        return AskerResponse(choice="deny")

    return _ask
