"""Size-budget (post-tool) + token-usage (post-model) hooks."""

from __future__ import annotations

import dataclasses
import json
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from aura.application.hooks import HookChain, PostModelHook, PostToolHook
from aura.application.loop_state import LoopState
from aura.domain.state_values import TokenStats
from aura.domain.tokens import estimate_message_tokens, estimate_text_tokens
from aura.domain.tool import ToolResult
from aura.domain.tool_meta_access import meta_dict
from aura.infrastructure.persistence import journal


def make_size_budget_hook(
    *,
    max_chars: int = 10_000,
    spill_dir: Path | None = None,
) -> PostToolHook:
    async def _hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],  # noqa: ARG001  # Protocol kw arg; unused
        result: ToolResult,
        state: LoopState,  # noqa: ARG001  # Protocol kw arg; unused
        **_: Any,
    ) -> ToolResult:
        if not result.ok or result.output is None:
            return result

        effective_max = meta_dict(tool).get("max_result_size_chars") or max_chars

        serialized = json.dumps(result.output, default=str, ensure_ascii=False)
        if len(serialized) <= effective_max:
            return result

        truncation: dict[str, bool | int | str] = {
            "truncated": True,
            "total_chars": len(serialized),
            "preview": serialized[:effective_max],
        }
        if spill_dir is not None:
            spill_dir.mkdir(parents=True, exist_ok=True)
            spill_path = spill_dir / f"{uuid.uuid4().hex}.json"
            spill_path.write_text(serialized, encoding="utf-8")
            truncation["spill_path"] = str(spill_path)

        return ToolResult(ok=True, output=truncation, display=result.display)

    return _hook


def _extract_token_usage(ai_message: AIMessage) -> dict[str, int]:
    """Per-turn input/output/cache-read counts; missing fields degrade to 0."""
    out = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}

    # isinstance guards: a non-conforming provider can hand back malformed values.
    usage = ai_message.usage_metadata
    if usage is not None:
        input_tokens = usage.get("input_tokens")
        if isinstance(input_tokens, int):
            out["input_tokens"] = input_tokens
        output_tokens = usage.get("output_tokens")
        if isinstance(output_tokens, int):
            out["output_tokens"] = output_tokens

    anthropic_usage = ai_message.response_metadata.get("usage")
    if isinstance(anthropic_usage, dict):
        cache_read = anthropic_usage.get("cache_read_input_tokens")
        if isinstance(cache_read, int):
            out["cache_read_tokens"] = cache_read

    return out


def make_usage_tracking_hook() -> PostModelHook:
    async def _hook(
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],
        state: LoopState,
        **_: Any,
    ) -> None:
        usage = ai_message.usage_metadata
        if usage:
            total = usage.get("total_tokens")
            if isinstance(total, int):
                state.total_tokens_used += total

        per_turn = _extract_token_usage(ai_message)
        # Char-estimator fallback for providers without usage_metadata (DashScope, some Ollama).
        if per_turn["input_tokens"] == 0 and per_turn["output_tokens"] == 0:
            per_turn["input_tokens"] = sum(estimate_message_tokens(msg) for msg in history)
            ai_content = ai_message.content
            per_turn["output_tokens"] = (
                estimate_text_tokens(ai_content)
                if isinstance(ai_content, str)
                else estimate_text_tokens(str(ai_content))
            )
            state.total_tokens_used += per_turn["input_tokens"] + per_turn["output_tokens"]

        prev = state.slots.token_stats
        new_stats = TokenStats(
            last_input_tokens=per_turn["input_tokens"],
            last_output_tokens=per_turn["output_tokens"],
            last_cache_read_tokens=per_turn["cache_read_tokens"],
            total_input_tokens=prev.total_input_tokens + per_turn["input_tokens"],
            total_output_tokens=prev.total_output_tokens + per_turn["output_tokens"],
            total_cache_read_tokens=(prev.total_cache_read_tokens + per_turn["cache_read_tokens"]),
            turn_count=prev.turn_count + 1,
        )
        state.slots = dataclasses.replace(state.slots, token_stats=new_stats)

        model_name = ""
        for key in ("model_name", "model", "model_id"):
            val = ai_message.response_metadata.get(key)
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

    return _hook


def default_hooks(
    *,
    max_result_size_chars: int = 50_000,
    spill_dir: Path | None = None,
) -> HookChain:
    return HookChain(
        pre_model=[],
        post_model=[make_usage_tracking_hook()],
        post_tool=[
            make_size_budget_hook(
                max_chars=max_result_size_chars,
                spill_dir=spill_dir,
            ),
        ],
    )
