"""Tool contract for the Aura agent loop."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, overload

from pydantic import BaseModel


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    display: str | None = None


@dataclass(frozen=True)
class AuraTool:
    name: str
    description: str
    input_model: type[BaseModel]
    is_read_only: bool
    is_destructive: bool
    is_concurrency_safe: bool
    max_result_size_chars: int | None
    _call: Callable[[BaseModel], Awaitable[ToolResult]]

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
    call: Callable[..., Any],
    is_read_only: bool = False,
    is_destructive: bool = False,
    is_concurrency_safe: bool = False,
    max_result_size_chars: int | None = None,
) -> AuraTool:
    if inspect.iscoroutinefunction(call):
        async_call: Callable[[BaseModel], Awaitable[ToolResult]] = call
    else:
        def async_call(params: BaseModel) -> Awaitable[ToolResult]:
            return asyncio.to_thread(call, params)

    return AuraTool(
        name=name,
        description=description,
        input_model=input_model,
        is_read_only=is_read_only,
        is_destructive=is_destructive,
        is_concurrency_safe=is_concurrency_safe,
        max_result_size_chars=max_result_size_chars,
        _call=async_call,
    )
