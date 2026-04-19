"""Tool execution protocol — ``ToolResult`` / ``ToolError`` / ``tool_metadata``.

Every Aura tool returns a ``ToolResult`` to the loop (either directly from a
successful call or wrapped by the loop after catching an exception), and may
raise ``ToolError`` from its body to report a user-facing failure.

``tool_metadata(...)`` returns the four-key dict every Aura tool carries on
``BaseTool.metadata``. Flags live in ``metadata`` (not as dedicated fields)
because LangChain's ``bind_tools`` only sends ``name``/``description``/
``args_schema`` to the model; anything extra we stash in ``metadata`` stays
local to the loop and hook layer — read via ``(tool.metadata or {}).get(...)``.

- ``is_read_only``       — safe to auto-approve in permission gate
- ``is_destructive``     — permission gate + safety rail must fire
- ``is_concurrency_safe`` — loop may batch with siblings under ``asyncio.gather``
- ``max_result_size_chars`` — post-tool budget hook truncation threshold
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
