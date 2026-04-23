"""Tool execution protocol — ``ToolResult`` / ``ToolError`` / ``tool_metadata``.

Every Aura tool returns a ``ToolResult`` to the loop (either directly from a
successful call or wrapped by the loop after catching an exception), and may
raise ``ToolError`` from its body to report a user-facing failure.

``tool_metadata(...)`` returns the dict every Aura tool carries on
``BaseTool.metadata``. Flags live in ``metadata`` (not as dedicated fields)
because LangChain's ``bind_tools`` only sends ``name``/``description``/
``args_schema`` to the model; anything extra we stash in ``metadata`` stays
local to the loop and hook layer — read via ``(tool.metadata or {}).get(...)``.

Keys:

- ``is_read_only``        — purely informational as of 2026-04-21 (Plan B
                             refactor). Used by the CLI prompt tag
                             ("read-only"). Permission decisions flow
                             through rules (see
                             ``aura.core.permissions.defaults.DEFAULT_ALLOW_RULES``)
                             and the safety policy, NOT this flag.
- ``is_destructive``      — permission gate + safety rail must fire
- ``is_concurrency_safe`` — loop may batch with siblings under ``asyncio.gather``
- ``max_result_size_chars`` — post-tool budget hook truncation threshold
- ``rule_matcher``        — ``(args, content) -> bool`` — how this tool decides
                             whether a pattern rule like ``"bash(npm test)"``
                             covers a given invocation. ``None`` means the tool
                             supports only tool-wide rules.
- ``args_preview``        — ``(args) -> str`` — one-line preview of this call's
                             arguments, rendered by the CLI permission prompt.
                             ``None`` falls back to the tool name.
- ``timeout_sec``         — ``float | None`` — hard deadline (seconds) the loop
                             wraps around ``tool.ainvoke``. ``None`` = no
                             deadline (tools with their own internal timeout
                             ladder, e.g. bash, set this to ``None`` so the
                             outer wrapper doesn't stack).
- ``is_search_command``   — ``bool`` — renderer folds long output for
                             search/read tools so the REPL isn't flooded by a
                             1000-line grep. False = render as usual.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ToolRuleMatcher = Callable[[dict[str, Any], str], bool]
ToolArgsPreview = Callable[[dict[str, Any]], str]


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
    rule_matcher: ToolRuleMatcher | None = None,
    args_preview: ToolArgsPreview | None = None,
    timeout_sec: float | None = None,
    is_search_command: bool = False,
) -> dict[str, Any]:
    """Build the metadata dict every Aura tool carries on ``metadata``."""
    return {
        "is_read_only": is_read_only,
        "is_destructive": is_destructive,
        "is_concurrency_safe": is_concurrency_safe,
        "max_result_size_chars": max_result_size_chars,
        "rule_matcher": rule_matcher,
        "args_preview": args_preview,
        "timeout_sec": timeout_sec,
        "is_search_command": is_search_command,
    }
