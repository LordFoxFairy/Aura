"""Ad-hoc tool factory for tests — ``build_tool``.

Production tools subclass ``BaseTool`` directly and set
``metadata=tool_metadata(...)``; this factory is a terser way to spin up
a one-off ``StructuredTool`` when writing tests. ``ToolResult``/``ToolError``/
``tool_metadata`` live in ``aura.schemas.tool``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from aura.schemas.tool import tool_metadata

_TParams = TypeVar("_TParams", bound=BaseModel)


def build_tool(
    *,
    name: str,
    description: str,
    args_schema: type[_TParams],
    func: Callable[..., Any] | None = None,
    coroutine: Callable[..., Awaitable[Any]] | None = None,
    is_read_only: bool = False,
    is_destructive: bool = False,
    is_concurrency_safe: bool = False,
    max_result_size_chars: int | None = None,
) -> BaseTool:
    return StructuredTool.from_function(
        func=func,
        coroutine=coroutine,
        name=name,
        description=description,
        args_schema=args_schema,
        metadata=tool_metadata(
            is_read_only=is_read_only,
            is_destructive=is_destructive,
            is_concurrency_safe=is_concurrency_safe,
            max_result_size_chars=max_result_size_chars,
        ),
    )
