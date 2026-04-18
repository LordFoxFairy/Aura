"""AuraTool Protocol + ToolResult — framework-agnostic, no permission logic.

Permission decisions are made by the loop via aura.core.permission.resolve();
tools only declare capability flags (is_read_only / is_destructive /
is_concurrency_safe) so the policy layer can short-circuit on read-only calls.
"""

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
    """Concrete tool produced by build_tool(). Satisfies the AuraTool Protocol structurally."""

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
    """Build an AuraTool from a minimal declarative spec.

    Fail-closed defaults: destructive=False but concurrency_safe=False too — caller must
    explicitly opt in to parallel-safe dispatch. This is the claude-code buildTool pattern
    adapted for Python (spec §3.2).

    The returned object satisfies the AuraTool Protocol structurally (runtime_checkable).
    """
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
