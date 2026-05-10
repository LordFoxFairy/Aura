"""Ad-hoc tool factory for tests — ``build_tool``.

Production tools subclass ``BaseTool`` directly and set
``aura_metadata: ToolMetadata``; this factory is a terser way to spin up
a one-off ``StructuredTool`` when writing tests. ``ToolResult``/``ToolError``/
``ToolMetadata`` live in ``aura.schemas.tool``.

Phase 5 Task 1 also introduces :class:`Tool`, a thin Aura mixin over
LangChain's ``BaseTool`` that supplies a default ``validate_input``
(accept-everything). Production tools migrate to subclass ``Tool`` in
later tasks (per-tool override of ``validate_input``); the loop's
permission gate (Task 8) calls ``validate_input`` on every tool, so
tools that don't yet inherit from :class:`Tool` are handled by a
``getattr`` fallback at the call site.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from aura.schemas.tool import (
    ToolArgsPreview,
    ToolMetadata,
    ToolRuleMatcher,
    ValidationResult,
)

_TParams = TypeVar("_TParams", bound=BaseModel)


class Tool(BaseTool):
    """Aura tool base — adds ``validate_input`` over LangChain's ``BaseTool``.

    Phase 5 Task 1 contract: every Aura tool exposes
    ``validate_input(args) -> ValidationResult``. The default returns
    ``ValidationResult(invalid=False)`` (accept) so subclasses only
    override when they have non-permission-related input constraints
    (e.g., ``read_file`` rejects relative paths above cwd, ``web_fetch``
    rejects unsupported URL schemes).

    The split mirrors claude-code's ``validateInput`` vs.
    ``checkPermissions``: the tool author owns "are these args
    structurally usable", the operator (permission gate) owns "should
    the user approve this". Surfacing the former as a structured
    ``ValidationResult`` lets the loop ``Block`` cleanly without
    routing through the ``ToolError`` exception path.

    Validation is intentionally synchronous — it should be cheap and
    not require I/O. Tools whose validity depends on async work raise
    ``ToolError`` from ``_arun`` instead.
    """

    def validate_input(self, args: dict[str, Any]) -> ValidationResult:
        """Default: accept any args. Subclasses override to constrain."""
        del args  # default impl is shape-agnostic; subclasses inspect
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
    """TEST-ONLY factory for one-off ``StructuredTool`` instances.

    Production tools MUST subclass ``BaseTool`` directly and set
    ``aura_metadata: ToolMetadata``. This factory exists purely as syntactic
    sugar for tests that need to spin up a minimal tool with specific
    capability flags — e.g. hook-level tests that care about the
    ``is_destructive`` bit but not about a real implementation body.

    Using ``build_tool`` outside ``tests/`` is a code smell: production tools
    benefit from subclass-scoped type checking, clearer stack traces, and
    the ability to override lifecycle hooks. If you find yourself reaching
    for ``build_tool`` in ``aura/``, promote the tool to its own subclass.
    """
    # Aura's metadata contract is exclusively ``aura_metadata: ToolMetadata``;
    # LangChain's ``BaseTool.metadata`` is left empty. ``StructuredTool`` is a
    # pydantic model with ``extra="ignore"`` so we use ``object.__setattr__``
    # to attach the field without tripping pydantic validation.
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
