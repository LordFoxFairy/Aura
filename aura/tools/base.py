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

from aura.schemas.tool import ToolArgsPreview, ToolRuleMatcher, tool_metadata

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
    rule_matcher: ToolRuleMatcher | None = None,
    args_preview: ToolArgsPreview | None = None,
) -> BaseTool:
    """TEST-ONLY factory for one-off ``StructuredTool`` instances.

    Production tools MUST subclass ``BaseTool`` directly and set
    ``metadata=tool_metadata(...)``. This factory exists purely as syntactic
    sugar for tests that need to spin up a minimal tool with specific
    capability flags — e.g. hook-level tests that care about the
    ``is_destructive`` bit but not about a real implementation body.

    Using ``build_tool`` outside ``tests/`` is a code smell: production tools
    benefit from subclass-scoped type checking, clearer stack traces, and
    the ability to override lifecycle hooks. If you find yourself reaching
    for ``build_tool`` in ``aura/``, promote the tool to its own subclass.
    """
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
            rule_matcher=rule_matcher,
            args_preview=args_preview,
        ),
    )
