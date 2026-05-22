"""F-0910-003 — PromptTooLong resilience path in compact.

The summary turn detects context-overflow signatures and recursively splits
the input in half until each chunk fits, then summarizes the partial summaries.
When even a single message is rejected (or the recursion cap is hit) it falls
back to a deterministic excerpt rather than crashing the session.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from aura.application.compact.compact import (
    _is_prompt_too_long,
    _run_summary_turn_with_retry,
)


class _FailingModel(BaseChatModel):
    """Model that raises PTL on the first ``fail_count`` calls then succeeds."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, fail_count: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.__dict__["fail_count"] = fail_count
        self.__dict__["calls"] = 0
        self.__dict__["seen_lengths"] = []

    @property
    def calls(self) -> int:
        return self.__dict__["calls"]  # type: ignore[no-any-return]

    @property
    def seen_lengths(self) -> list[int]:
        return self.__dict__["seen_lengths"]  # type: ignore[no-any-return]

    @property
    def _llm_type(self) -> str:
        return "failing"

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["calls"] += 1
        # Track HOW MUCH content the summary call saw — proves the drop step ran.
        self.__dict__["seen_lengths"].append(
            sum(len(str(m.content)) for m in messages),
        )
        if self.__dict__["calls"] <= self.__dict__["fail_count"]:
            raise RuntimeError("the prompt is too long")
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="OK"))],
        )

    def _generate(self, *a: Any, **k: Any) -> ChatResult:
        raise NotImplementedError


def test_is_prompt_too_long_phrases() -> None:
    assert _is_prompt_too_long(RuntimeError("the prompt is too long"))
    assert _is_prompt_too_long(RuntimeError("context length exceeded"))
    assert _is_prompt_too_long(RuntimeError("MAXIMUM context"))
    assert not _is_prompt_too_long(RuntimeError("totally unrelated"))


@pytest.mark.asyncio
async def test_resilient_split_succeeds_after_initial_failure() -> None:
    """Splitting the input in half produces chunks the provider accepts."""
    model = _FailingModel(fail_count=2)
    msgs: list[BaseMessage] = [
        HumanMessage(content="x" * 1000) for _ in range(10)
    ]
    out = await _run_summary_turn_with_retry(model, msgs)
    # Final result is a non-empty summary; we don't pin the exact string
    # because the recursive merge concatenates partial summaries.
    assert out
    # Retry must actually retry — first attempt fails, recursion follows.
    assert model.calls > 1
    # First attempt sees the full input; subsequent halves see strictly less.
    assert model.seen_lengths[1] < model.seen_lengths[0]


@pytest.mark.asyncio
async def test_resilient_falls_back_when_provider_always_rejects() -> None:
    """When even a 1-message chunk is rejected, return a deterministic excerpt."""
    model = _FailingModel(fail_count=10**9)  # always fails
    msgs: list[BaseMessage] = [
        HumanMessage(content="x" * 1000) for _ in range(10)
    ]
    out = await _run_summary_turn_with_retry(model, msgs)
    # Never crashes — always produces something the loop can carry forward.
    assert "<goal>" in out
    assert "<history-excerpt>" in out
    # Model was attempted before the fallback fired.
    assert model.calls > 0


@pytest.mark.asyncio
async def test_non_ptl_error_re_raised_immediately() -> None:
    """A non-PTL exception is propagated on the first attempt without retry."""

    class _Boom(BaseChatModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.__dict__["calls"] = 0

        @property
        def _llm_type(self) -> str:
            return "boom"

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **_: Any,
        ) -> ChatResult:
            self.__dict__["calls"] += 1
            raise RuntimeError("api error 500")

        def _generate(self, *a: Any, **k: Any) -> ChatResult:
            raise NotImplementedError

    model = _Boom()
    with pytest.raises(RuntimeError, match="api error"):
        await _run_summary_turn_with_retry(model, [HumanMessage(content="x")])
    assert model.__dict__["calls"] == 1
