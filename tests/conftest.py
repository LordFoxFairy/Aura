"""Shared pytest fixtures for Aura tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import ConfigDict


@pytest.fixture(autouse=True)
def clear_aura_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove AURA_CONFIG from the environment for every test.

    Without this, a CI runner or local shell with AURA_CONFIG set would leak
    into any test that calls load_config (directly or via Agent.__init__).
    """
    monkeypatch.delenv("AURA_CONFIG", raising=False)


# ---------------------------------------------------------------------------
# FakeChatModel — scripts AIMessageChunk sequences per turn
# ---------------------------------------------------------------------------


@dataclass
class FakeTurn:
    """One LLM turn's worth of scripted chunks (for _astream) or a final AIMessage."""

    chunks: list[AIMessageChunk]

    def as_ai_message(self) -> AIMessage:
        """Accumulate chunks into a final AIMessage for _agenerate."""
        acc: AIMessageChunk | None = None
        for c in self.chunks:
            acc = c if acc is None else acc + c
        if acc is None:
            return AIMessage(content="")
        return AIMessage(content=acc.content, tool_calls=acc.tool_calls or [])


class FakeChatModel(BaseChatModel):
    """Records bind_tools calls and plays back scripted turns for _astream / _agenerate."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, turns: list[FakeTurn] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Bypass pydantic __setattr__ — BaseChatModel is strict about unknown fields
        self.__dict__["_turns"] = list(turns or [])
        self.__dict__["seen_bound_tools"] = []
        self.__dict__["astream_calls"] = 0
        self.__dict__["agenerate_calls"] = 0

    # Typed accessors so tests can read state without mypy complaints
    @property
    def seen_bound_tools(self) -> list[list[Any]]:
        return self.__dict__["seen_bound_tools"]  # type: ignore[no-any-return]

    @property
    def astream_calls(self) -> int:
        return self.__dict__["astream_calls"]  # type: ignore[no-any-return]

    @property
    def agenerate_calls(self) -> int:
        return self.__dict__["agenerate_calls"]  # type: ignore[no-any-return]

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        self.__dict__["seen_bound_tools"].append(list(tools))
        return self

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.__dict__["astream_calls"] += 1
        turn = self._pop_turn()
        for chunk in turn.chunks:
            yield ChatGenerationChunk(message=chunk)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["agenerate_calls"] += 1
        turn = self._pop_turn()
        ai = turn.as_ai_message()
        return ChatResult(generations=[ChatGeneration(message=ai)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **_: Any,
    ) -> ChatResult:
        """Synchronous fallback — delegates to _pop_turn (test-only, not used in async paths)."""
        turn = self._pop_turn()
        ai = turn.as_ai_message()
        return ChatResult(generations=[ChatGeneration(message=ai)])

    def _pop_turn(self) -> FakeTurn:
        turns: list[FakeTurn] = self.__dict__["_turns"]
        if not turns:
            raise RuntimeError("FakeChatModel: no scripted turns left")
        return turns.pop(0)


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------


def text_chunk(text: str, *, final: bool = False) -> AIMessageChunk:
    """Produce an AIMessageChunk carrying text content. final=True sets chunk_position='last'."""
    kwargs: dict[str, Any] = {"content": text}
    if final:
        kwargs["response_metadata"] = {"finish_reason": "stop"}
        return AIMessageChunk(**kwargs, chunk_position="last")
    return AIMessageChunk(**kwargs)


def _make_tcc(name: str, args: str, id: str, index: int) -> ToolCallChunk:  # noqa: A002
    return ToolCallChunk(name=name, args=args, id=id, index=index, type="tool_call_chunk")


def tool_call_chunk(
    name: str,
    args_json: str,
    id: str,  # noqa: A002
    *,
    final: bool = True,
) -> AIMessageChunk:
    """Produce an AIMessageChunk carrying ONE tool call.

    Defaults final=True — AIMessage.tool_calls population requires chunk_position='last'
    somewhere in the accumulated chain.
    """
    tcc = [_make_tcc(name, args_json, id, 0)]
    if final:
        return AIMessageChunk(
            content="",
            tool_call_chunks=tcc,
            chunk_position="last",
            response_metadata={"finish_reason": "tool_calls"},
        )
    return AIMessageChunk(content="", tool_call_chunks=tcc)


def parallel_tool_calls_chunk(calls: list[dict[str, str]]) -> AIMessageChunk:
    """Produce ONE AIMessageChunk carrying MULTIPLE tool calls at once.

    Each entry in `calls` must have keys: `name`, `args_json`, `id`.
    """
    tcc = [
        _make_tcc(c["name"], c["args_json"], c["id"], i)
        for i, c in enumerate(calls)
    ]
    return AIMessageChunk(
        content="",
        tool_call_chunks=tcc,
        chunk_position="last",
        response_metadata={"finish_reason": "tool_calls"},
    )
