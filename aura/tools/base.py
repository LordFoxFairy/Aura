"""Tool contract for the Aura agent loop."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast, overload, runtime_checkable

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
    max_result_size_chars: int | None

    async def acall(self, params: BaseModel) -> ToolResult:
        return await self._call(params)


@overload
def build_tool(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
    call: Callable[[BaseModel], ToolResult],
    is_read_only: bool = False,
    is_destructive: bool = False,
    is_concurrency_safe: bool = False,
    max_result_size_chars: int | None = None,
) -> AuraTool: ...


@overload
def build_tool(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
    call: Callable[[BaseModel], Awaitable[ToolResult]],
    is_read_only: bool = False,
    is_destructive: bool = False,
    is_concurrency_safe: bool = False,
    max_result_size_chars: int | None = None,
) -> AuraTool: ...


def build_tool(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
    call: Callable[[BaseModel], ToolResult] | Callable[[BaseModel], Awaitable[ToolResult]],
    is_read_only: bool = False,
    is_destructive: bool = False,
    is_concurrency_safe: bool = False,
    max_result_size_chars: int | None = None,
) -> AuraTool:
    if inspect.iscoroutinefunction(call):
        async_call: Callable[[BaseModel], Awaitable[ToolResult]] = call
    else:
        sync_call = cast(Callable[[BaseModel], ToolResult], call)

        async def async_call(params: BaseModel) -> ToolResult:
            return await asyncio.to_thread(sync_call, params)

    return cast(
        AuraTool,
        _Tool(
            name=name,
            description=description,
            input_model=input_model,
            is_read_only=is_read_only,
            is_destructive=is_destructive,
            is_concurrency_safe=is_concurrency_safe,
            _call=async_call,
            max_result_size_chars=max_result_size_chars,
        ),
    )
