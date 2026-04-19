"""Permission layer — PreToolHook factory that gates destructive tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.tools import BaseTool

from aura.core.hooks import PreToolHook
from aura.core.state import LoopState
from aura.tools.base import ToolResult


@dataclass
class PermissionSession:
    allowlist: set[str] = field(default_factory=set)


class PermissionAsker(Protocol):
    async def __call__(
        self, tool: BaseTool, args: dict[str, Any],
    ) -> bool: ...


def make_permission_hook(
    *,
    asker: PermissionAsker,
    session: PermissionSession,
) -> PreToolHook:
    async def _hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **_: Any,
    ) -> ToolResult | None:
        if (tool.metadata or {}).get("is_read_only", False):
            return None
        if tool.name in session.allowlist:
            return None
        if await asker(tool, args):
            return None
        return ToolResult(ok=False, error="permission denied by user")

    return _hook
