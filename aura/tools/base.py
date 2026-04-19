"""Tool contract — ``ToolResult`` / ``ToolError`` and the ``build_tool`` factory.

Project-specific flags live in ``tool.metadata`` so ``bind_tools`` accepts
the tool natively; read via ``(tool.metadata or {}).get(...)``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    display: str | None = None


class ToolError(Exception):
    """Raise from a tool function to surface a user-facing error message."""


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
    metadata: dict[str, Any] = {
        "is_read_only": is_read_only,
        "is_destructive": is_destructive,
        "is_concurrency_safe": is_concurrency_safe,
        "max_result_size_chars": max_result_size_chars,
    }
    return StructuredTool.from_function(
        func=func,
        coroutine=coroutine,
        name=name,
        description=description,
        args_schema=args_schema,
        metadata=metadata,
    )
