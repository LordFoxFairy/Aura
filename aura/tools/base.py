"""AuraTool protocol and supporting types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, runtime_checkable

from pydantic import BaseModel


class PermissionResult(NamedTuple):
    allow: bool
    reason: str | None = None


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

    def check_permissions(self, params: BaseModel) -> PermissionResult: ...

    async def acall(self, params: BaseModel) -> ToolResult: ...
