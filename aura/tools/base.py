"""Tool contract — ``ToolResult`` / ``ToolError`` and the ``tool_metadata`` helper.

Every Aura tool is a ``BaseTool`` subclass that declares four project-specific
flags in its ``metadata`` dict:

- ``is_read_only``      — safe to auto-approve
- ``is_destructive``    — permission gate + safety rail
- ``is_concurrency_safe`` — loop may run in parallel with siblings
- ``max_result_size_chars`` — post-tool budget hook truncation threshold

``tool_metadata(...)`` returns the four-key dict with the right defaults so
each tool class only has to spell out the flags that deviate from False/None.

Flags live in ``metadata`` (not as dedicated fields) because LangChain's
``bind_tools`` only sends ``name`` / ``description`` / ``args_schema`` to the
model; anything extra we stash in ``metadata`` stays local to the loop and
hook layer — read via ``(tool.metadata or {}).get(...)``.

``ToolError`` is the *user-facing* error signal raised from tool bodies; the
loop converts it to a failed ``ToolResult``. Any other exception surfaces as
``<ExceptionType>: <msg>`` in the ``ToolResult.error`` string.

``build_tool`` remains as a terse factory for ad-hoc test tools — prefer a
``BaseTool`` subclass for production tools.
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
    """Raise from a tool ``_run`` / ``_arun`` to surface a user-facing error."""


def tool_metadata(
    *,
    is_read_only: bool = False,
    is_destructive: bool = False,
    is_concurrency_safe: bool = False,
    max_result_size_chars: int | None = None,
) -> dict[str, Any]:
    """Build the four-key metadata dict every Aura tool carries on ``metadata``."""
    return {
        "is_read_only": is_read_only,
        "is_destructive": is_destructive,
        "is_concurrency_safe": is_concurrency_safe,
        "max_result_size_chars": max_result_size_chars,
    }


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
    """Terse factory for ad-hoc tools (primarily tests). Production tools
    should subclass ``BaseTool`` directly and set ``metadata=tool_metadata(...)``.
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
        ),
    )
