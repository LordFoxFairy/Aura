"""Loop-level permission interceptor.

Policy is applied by resolve(), invoked by the loop before each tool.acall().
Tools carry capability flags only (see aura.tools.base.AuraTool); they do NOT
implement any permission method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

from aura.tools.base import AuraTool


@dataclass(frozen=True)
class PermissionDecision:
    allow: bool
    reason: str | None = None


@dataclass
class PermissionSession:
    """Mutable session state — tools user has granted 'always allow' for this run."""
    allowlist: set[str] = field(default_factory=set)


class PermissionAsker(Protocol):
    """Interactive gate — called when the policy cannot decide unilaterally."""
    async def __call__(self, tool: AuraTool, params: BaseModel) -> bool: ...


async def resolve(
    tool: AuraTool,
    params: BaseModel,
    *,
    asker: PermissionAsker,
    session: PermissionSession,
) -> PermissionDecision:
    """Decide whether tool(params) should execute.

    Policy (MVP):
      1. tool.is_read_only → allow (asker NOT invoked)
      2. tool.name in session.allowlist → allow (asker NOT invoked)
      3. else → await asker(tool, params); map bool to Decision
    """
    if tool.is_read_only:
        return PermissionDecision(allow=True)
    if tool.name in session.allowlist:
        return PermissionDecision(allow=True)
    allowed = await asker(tool, params)
    if allowed:
        return PermissionDecision(allow=True)
    return PermissionDecision(allow=False, reason="denied by user")
