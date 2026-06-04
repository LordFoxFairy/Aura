"""Tool base class and ad-hoc test factory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from aura.domain.tool import (
    ToolArgsPreview,
    ToolMetadata,
    ToolRuleMatcher,
    ValidationResult,
)

_TParams = TypeVar("_TParams", bound=BaseModel)


class Tool(BaseTool):
    def validate_input(self, args: dict[str, Any]) -> ValidationResult:
        del args
        return ValidationResult(invalid=False)


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
    timeout_sec: float | None = None,
) -> BaseTool:
    # StructuredTool is extra="ignore"; bypass validation to attach aura_metadata.
    tool = StructuredTool.from_function(
        func=func,
        coroutine=coroutine,
        name=name,
        description=description,
        args_schema=args_schema,
    )
    aura_meta = ToolMetadata(
        is_read_only=is_read_only,
        is_destructive=is_destructive,
        is_concurrency_safe=is_concurrency_safe,
        rule_matcher=rule_matcher,
        args_preview=args_preview,
        timeout_sec=timeout_sec,
        max_result_size_chars=max_result_size_chars,
    )
    object.__setattr__(tool, "aura_metadata", aura_meta)
    return tool
