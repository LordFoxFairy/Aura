"""Interactive y/N/a permission asker for the CLI."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel

from aura.core.hooks.permission import PermissionAsker, PermissionSession
from aura.core.persistence import journal
from aura.tools.base import AuraTool


def _short(args: dict[str, Any], *, max_len: int = 80) -> str:
    rendered = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) <= max_len:
        return rendered
    return rendered[:max_len] + "\u2026"


def make_cli_asker(session: PermissionSession) -> PermissionAsker:
    async def _ask(tool: AuraTool, params: BaseModel) -> bool:
        args_preview = _short(params.model_dump())
        journal.write(
            "permission_asked",
            tool=tool.name,
            is_destructive=tool.is_destructive,
            params_preview=args_preview,
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
            return False
        answer = answer.strip().lower()
        if answer == "a":
            session.allowlist.add(tool.name)
            journal.write(
                "permission_answered", tool=tool.name, answer="a", allowed=True,
            )
            return True
        allowed = answer == "y"
        journal.write(
            "permission_answered",
            tool=tool.name,
            answer=answer or "(empty)",
            allowed=allowed,
        )
        return allowed

    return _ask
