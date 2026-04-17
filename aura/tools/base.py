"""AuraTool Protocol + ToolResult — framework-agnostic, no permission logic.

Permission decisions are made by the loop via aura.core.permission.resolve();
tools only declare capability flags (is_read_only / is_destructive /
is_concurrency_safe) so the policy layer can short-circuit on read-only calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    display: str | None = None


@runtime_checkable
class AuraTool(Protocol):
    name: str
    description: str
    input_model: type[BaseModel]
    is_read_only: bool
    is_destructive: bool
    is_concurrency_safe: bool

    async def acall(self, params: BaseModel) -> ToolResult: ...
