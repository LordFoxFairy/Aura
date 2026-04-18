"""Interactive y/N/a permission asker for the CLI."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel

from aura.core.permission import PermissionAsker, PermissionSession
from aura.tools.base import AuraTool


def _short(args: dict[str, Any], *, max_len: int = 80) -> str:
    rendered = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) <= max_len:
        return rendered
    return rendered[:max_len] + "…"


def make_cli_asker(session: PermissionSession) -> PermissionAsker:
    async def _ask(tool: AuraTool, params: BaseModel) -> bool:
        args_preview = _short(params.model_dump())
        prompt = (
            f"\n  tool: {tool.name}({args_preview})\n"
            "  allow this call?  [y]es / [N]o / [a]lways: "
        )
        try:
            answer = await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            return False
        answer = answer.strip().lower()
        if answer == "a":
            session.allowlist.add(tool.name)
            return True
        return answer == "y"

    return _ask
