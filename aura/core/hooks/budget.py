"""Size-budget post-tool hook and usage-tracking post-model hook."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from aura.core.hooks import HookChain, PostModelHook, PostToolHook, PreModelHook
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult


def make_size_budget_hook(
    *,
    max_chars: int = 10_000,
    spill_dir: Path | None = None,
) -> PostToolHook:
    async def _hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        result: ToolResult,
        state: LoopState,
        **_: Any,
    ) -> ToolResult:
        if not result.ok or result.output is None:
            return result

        effective_max = (tool.metadata or {}).get("max_result_size_chars") or max_chars

        serialized = json.dumps(result.output, default=str, ensure_ascii=False)
        if len(serialized) <= effective_max:
            return result

        truncation: dict[str, Any] = {
            "truncated": True,
            "total_chars": len(serialized),
            "preview": serialized[:effective_max],
        }
        if spill_dir is not None:
            spill_dir.mkdir(parents=True, exist_ok=True)
            spill_path = spill_dir / f"{uuid.uuid4().hex}.json"
            spill_path.write_text(serialized, encoding="utf-8")
            truncation["spill_path"] = str(spill_path)

        return ToolResult(
            ok=True,
            output=truncation,
            display=result.display,
        )

    return _hook


def _extract_token_stats(ai_message: AIMessage) -> dict[str, int]:
    """Pull per-turn input / output / cache-read token counts from *ai_message*.

    LangChain normalizes most providers into ``usage_metadata`` with
    ``input_tokens`` / ``output_tokens`` / ``total_tokens``. Anthropic's
    prompt-cache fields (``cache_read_input_tokens``) aren't reflected in
    the normalized shape yet, so we also peek at ``response_metadata['usage']``.
    Any missing / wrong-typed field degrades to 0 rather than raising —
    the status bar tolerates zeroes, it doesn't tolerate a crashed hook.
    """
    out = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}

    usage = getattr(ai_message, "usage_metadata", None) or {}
    if isinstance(usage, dict):
        val = usage.get("input_tokens")
        if isinstance(val, int):
            out["input_tokens"] = val
        val = usage.get("output_tokens")
        if isinstance(val, int):
            out["output_tokens"] = val

    # Anthropic: response_metadata.usage.cache_read_input_tokens
    meta = getattr(ai_message, "response_metadata", None) or {}
    if isinstance(meta, dict):
        anthropic_usage = meta.get("usage")
        if isinstance(anthropic_usage, dict):
            val = anthropic_usage.get("cache_read_input_tokens")
            if isinstance(val, int):
                out["cache_read_tokens"] = val

    return out


def make_usage_tracking_hook() -> PostModelHook:
    async def _hook(
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],
        state: LoopState,
        **_: Any,
    ) -> None:
        # Legacy field: cumulative total-tokens counter. Kept so the
        # /verbose summary and older log lines still work.
        usage = getattr(ai_message, "usage_metadata", None)
        if usage:
            total = usage.get("total_tokens")
            if isinstance(total, int):
                state.total_tokens_used += total

        # New structured stats for the status bar.
        per_turn = _extract_token_stats(ai_message)
        stats = state.custom.setdefault(
            "_token_stats",
            {
                "last_input_tokens": 0,
                "last_cache_read_tokens": 0,
                "last_output_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cache_read_tokens": 0,
                "turn_count": 0,
            },
        )
        stats["last_input_tokens"] = per_turn["input_tokens"]
        stats["last_cache_read_tokens"] = per_turn["cache_read_tokens"]
        stats["last_output_tokens"] = per_turn["output_tokens"]
        stats["total_input_tokens"] += per_turn["input_tokens"]
        stats["total_output_tokens"] += per_turn["output_tokens"]
        stats["total_cache_read_tokens"] = (
            stats.get("total_cache_read_tokens", 0) + per_turn["cache_read_tokens"]
        )
        stats["turn_count"] = stats.get("turn_count", 0) + 1

        # Always-on ``turn_usage`` journal event. Unlike the optional
        # event-logger's ``post_model`` (attached via ``--log``), this fires
        # every turn by default so ``/stats`` and future v0.14 historical
        # aggregation have real data without the user opting in. Model name
        # extraction is best-effort — some providers put it in
        # ``response_metadata['model_name']``, others in ``['model']``; we
        # accept either and fall back to empty string if neither exists.
        from aura.core.persistence import journal

        model_name = ""
        meta = getattr(ai_message, "response_metadata", None) or {}
        if isinstance(meta, dict):
            for key in ("model_name", "model", "model_id"):
                val = meta.get(key)
                if isinstance(val, str) and val:
                    model_name = val
                    break
        journal.write(
            "turn_usage",
            turn=state.turn_count,
            model=model_name,
            input_tokens=per_turn["input_tokens"],
            output_tokens=per_turn["output_tokens"],
            cache_read_tokens=per_turn["cache_read_tokens"],
        )

        # Update the pet-buddy mood now that the turn has landed. The
        # buddy observer is deliberately the LAST thing in this hook —
        # it's pure display state, never load-bearing, and mustn't
        # interfere with the journal write above if it somehow raises.
        from aura.cli import buddy as _buddy

        await _buddy.observe_post_model(state=state)

    return _hook


def make_buddy_observer_hook() -> PostToolHook:
    """post_tool observer wired alongside the size-budget hook.

    Updates :mod:`aura.cli.buddy` mood based on tool success/failure
    (failure → worried, success → clear worry flag) and passes the
    result through unmodified. Pure observation: never mutates
    ``result`` so ordering against ``make_size_budget_hook`` is
    irrelevant.
    """

    async def _hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        result: ToolResult,
        state: LoopState,
        **_: Any,
    ) -> ToolResult:
        from aura.cli import buddy as _buddy

        await _buddy.observe_post_tool(state=state, result=result)
        return result

    return _hook


def make_buddy_thinking_hook() -> PreModelHook:
    """pre_model observer that flips the buddy to ``thinking`` mood.

    Fires AFTER the user's HumanMessage is appended (G1 contract) but
    BEFORE ``ainvoke`` blocks. Status-bar render between this hook and
    the model reply will show the buddy in its "thinking" mood, giving
    the operator instant feedback that the prompt was received and is
    in flight. ``observe_post_model`` flips it out again on reply.

    Worry-preserving: if a recent tool error left ``had_recent_error``
    set, the buddy stays worried — see ``observe_pre_model`` for the
    rationale (continuity over micro-state-change visibility).
    """

    async def _hook(
        *,
        history: list[BaseMessage],
        state: LoopState,
        **_: Any,
    ) -> None:
        from aura.cli import buddy as _buddy

        await _buddy.observe_pre_model(state=state)

    return _hook


def default_hooks(
    *,
    max_result_size_chars: int = 50_000,
    spill_dir: Path | None = None,
) -> HookChain:
    return HookChain(
        pre_model=[make_buddy_thinking_hook()],
        post_model=[make_usage_tracking_hook()],
        post_tool=[
            make_size_budget_hook(
                max_chars=max_result_size_chars, spill_dir=spill_dir,
            ),
            make_buddy_observer_hook(),
        ],
    )
