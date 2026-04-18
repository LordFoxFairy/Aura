"""Tool contract for the Aura agent loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

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


@dataclass(frozen=True)
class _Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    is_read_only: bool
    is_destructive: bool
    is_concurrency_safe: bool
    _call: Callable[[BaseModel], Awaitable[ToolResult]]

    async def acall(self, params: BaseModel) -> ToolResult:
        return await self._call(params)


def build_tool(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
    call: Callable[[BaseModel], Awaitable[ToolResult]],
    is_read_only: bool = False,
    is_destructive: bool = False,
    is_concurrency_safe: bool = False,
) -> AuraTool:
    return cast(
        AuraTool,
        _Tool(
            name=name,
            description=description,
            input_model=input_model,
            is_read_only=is_read_only,
            is_destructive=is_destructive,
            is_concurrency_safe=is_concurrency_safe,
            _call=call,
        ),
    )
