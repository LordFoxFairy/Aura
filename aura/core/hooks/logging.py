"""HookChain that writes lifecycle events to the global journal."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from aura.core.hooks import HookChain
from aura.core.persistence import journal
from aura.core.state import LoopState
from aura.tools.base import ToolResult


def make_event_logger_hooks() -> HookChain:
    async def _pre_model(
        *, history: list[BaseMessage], state: LoopState, **_: Any,
    ) -> None:
        journal.write(
            "pre_model",
            turn=state.turn_count,
            history_len=len(history),
        )

    async def _post_model(
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],
        state: LoopState,
        **_: Any,
    ) -> None:
        usage = getattr(ai_message, "usage_metadata", None) or {}
        content = str(ai_message.content) if ai_message.content else ""
        journal.write(
            "post_model",
            turn=state.turn_count,
            content_chars=len(content),
            content_preview=_trim(content, 500),
            tool_calls=len(ai_message.tool_calls or []),
            usage=dict(usage),
            total_tokens=state.total_tokens_used,
        )

    async def _pre_tool(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: Any,
    ) -> ToolResult | None:
        try:
            args_preview = _trim(json.dumps(args, ensure_ascii=False, default=str), 200)
        except Exception:  # noqa: BLE001
            args_preview = "<unserializable>"
        journal.write(
            "pre_tool",
            turn=state.turn_count,
            tool=tool.name,
            is_destructive=(tool.metadata or {}).get("is_destructive", False),
            args_preview=args_preview,
        )
        return None

    async def _post_tool(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        result: ToolResult,
        state: LoopState,
        **_: Any,
    ) -> ToolResult:
        output_chars = (
            len(json.dumps(result.output, default=str, ensure_ascii=False))
            if result.output is not None
            else 0
        )
        journal.write(
            "post_tool",
            turn=state.turn_count,
            tool=tool.name,
            ok=result.ok,
            error=result.error,
            output_chars=output_chars,
        )
        return result

    return HookChain(
        pre_model=[_pre_model],
        post_model=[_post_model],
        pre_tool=[_pre_tool],
        post_tool=[_post_tool],
    )


def wrap_with_event_logger(inner: HookChain) -> HookChain:
    """Splice the journal logger around *inner* so the audit trail brackets
    the business hooks: log runs FIRST on `pre_*` (capture what was asked,
    before any hook can deny) and LAST on `post_*` (capture the final
    result, after any hook has mutated it)."""
    log = make_event_logger_hooks()
    return HookChain(
        pre_model=[*log.pre_model, *inner.pre_model],
        post_model=[*inner.post_model, *log.post_model],
        pre_tool=[*log.pre_tool, *inner.pre_tool],
        post_tool=[*inner.post_tool, *log.post_tool],
    )


def _trim(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[:max_len] + "\u2026"
