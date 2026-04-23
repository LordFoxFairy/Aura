"""Tests for token-usage extraction in make_usage_tracking_hook.

The hook must extract per-turn input / output / cached-prompt tokens from
whichever shape the provider hands back (LangChain normalizes most of this
into ``usage_metadata``, but cache-read info lives in ``response_metadata``
for Anthropic). Results land in ``state.custom['_token_stats']`` for the
status bar to read, and ``state.total_tokens_used`` stays a cumulative
total so legacy callers keep working.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from aura.core.hooks.budget import make_usage_tracking_hook
from aura.schemas.state import LoopState


async def test_usage_hook_extracts_anthropic_style_token_metadata() -> None:
    hook = make_usage_tracking_hook()
    state = LoopState()

    # LangChain-anthropic populates usage_metadata with input/output/total
    # and forwards Anthropic's cache_read fields inside response_metadata.
    ai = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 5400,
            "output_tokens": 120,
            "total_tokens": 5520,
        },
        response_metadata={
            "usage": {
                "input_tokens": 5400,
                "cache_read_input_tokens": 34000,
                "cache_creation_input_tokens": 0,
                "output_tokens": 120,
            },
        },
    )
    await hook(ai_message=ai, history=[], state=state)

    stats = state.custom["_token_stats"]
    assert stats["last_input_tokens"] == 5400
    assert stats["last_cache_read_tokens"] == 34000
    assert stats["last_output_tokens"] == 120
    assert stats["total_input_tokens"] == 5400
    assert stats["total_output_tokens"] == 120


async def test_usage_hook_extracts_openai_style_token_metadata() -> None:
    hook = make_usage_tracking_hook()
    state = LoopState()

    # langchain-openai maps prompt/completion into usage_metadata's
    # input/output keys; no cache_read exposure in the public shape.
    ai = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 321,
            "output_tokens": 88,
            "total_tokens": 409,
        },
        response_metadata={
            "token_usage": {
                "prompt_tokens": 321,
                "completion_tokens": 88,
                "total_tokens": 409,
            },
        },
    )
    await hook(ai_message=ai, history=[], state=state)

    stats = state.custom["_token_stats"]
    assert stats["last_input_tokens"] == 321
    assert stats["last_output_tokens"] == 88
    assert stats["last_cache_read_tokens"] == 0


async def test_usage_hook_tolerates_missing_token_metadata() -> None:
    hook = make_usage_tracking_hook()
    state = LoopState()

    # Ollama / fake models may omit usage_metadata entirely. No crash,
    # stats stay at zero, total stays at zero.
    ai = AIMessage(content="no usage here")
    await hook(ai_message=ai, history=[], state=state)

    stats = state.custom.get("_token_stats", {})
    assert stats.get("last_input_tokens", 0) == 0
    assert stats.get("last_cache_read_tokens", 0) == 0
    assert stats.get("last_output_tokens", 0) == 0
    assert state.total_tokens_used == 0


async def test_usage_hook_accumulates_totals_across_turns() -> None:
    hook = make_usage_tracking_hook()
    state = LoopState()

    ai1 = AIMessage(
        content="a",
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )
    ai2 = AIMessage(
        content="b",
        usage_metadata={"input_tokens": 200, "output_tokens": 20, "total_tokens": 220},
    )
    await hook(ai_message=ai1, history=[], state=state)
    await hook(ai_message=ai2, history=[], state=state)

    stats = state.custom["_token_stats"]
    # Last-turn values reflect turn 2 only.
    assert stats["last_input_tokens"] == 200
    assert stats["last_output_tokens"] == 20
    # Totals accumulate.
    assert stats["total_input_tokens"] == 300
    assert stats["total_output_tokens"] == 30


async def test_usage_hook_stores_stats_on_state_custom() -> None:
    hook = make_usage_tracking_hook()
    state = LoopState()

    ai = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
    await hook(ai_message=ai, history=[], state=state)

    # Key lives on state.custom so it doesn't pollute the LoopState schema.
    assert "_token_stats" in state.custom
    assert isinstance(state.custom["_token_stats"], dict)


async def test_usage_hook_backward_compat_total_tokens_used_still_tracked() -> None:
    hook = make_usage_tracking_hook()
    state = LoopState()

    ai = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    await hook(ai_message=ai, history=[], state=state)
    # Legacy /verbose summary reads this field; keep it working.
    assert state.total_tokens_used == 15
