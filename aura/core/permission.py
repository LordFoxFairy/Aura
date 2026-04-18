"""Permission layer — PreToolHook factory that gates destructive tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from aura.core.hooks import PreToolHook
from aura.core.state import LoopState
from aura.tools.base import AuraTool, ToolResult


@dataclass
class PermissionSession:
    allowlist: set[str] = field(default_factory=set)


class PermissionAsker(Protocol):
    async def __call__(
        self, tool: AuraTool, params: BaseModel,
    ) -> bool: ...


def make_permission_hook(
    *,
    asker: PermissionAsker,
    session: PermissionSession,
) -> PreToolHook:
    async def _hook(
        *,
        tool: AuraTool,
        params: BaseModel,
        state: LoopState,
        **_: Any,
    ) -> ToolResult | None:
        if tool.is_read_only:
            return None
        if tool.name in session.allowlist:
            return None
        if await asker(tool, params):
            return None
        return ToolResult(ok=False, error="permission denied by user")

    return _hook
