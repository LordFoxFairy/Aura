"""Tool contract for the Aura agent loop.

Tools are LangChain ``StructuredTool`` instances built via :func:`build_tool`.
Project-specific flags (``is_read_only`` / ``is_destructive`` /
``is_concurrency_safe`` / ``max_result_size_chars``) live in ``tool.metadata``
so that ``bind_tools([t1, t2])`` accepts them natively; read them inline with
``(tool.metadata or {}).get("is_destructive", False)``.
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
