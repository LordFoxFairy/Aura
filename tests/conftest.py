"""Shared pytest fixtures for Aura tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import ConfigDict


@pytest.fixture(autouse=True)
def clear_aura_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURA_CONFIG", raising=False)


@dataclass
class FakeTurn:
    """A scripted turn. Either message (for _agenerate/legacy tests) OR chunks (for _astream)."""
    message: AIMessage | None = None
    chunks: list[AIMessageChunk] | None = None

    def resolved_message(self) -> AIMessage:
        if self.message is not None:
            return self.message
        if self.chunks:
            acc: AIMessageChunk | None = None
            for c in self.chunks:
                acc = c if acc is None else acc + c
            assert acc is not None
            return AIMessage(content=acc.content, tool_calls=list(acc.tool_calls or []))
        return AIMessage(content="")


class FakeChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, turns: list[FakeTurn] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__dict__["_turns"] = list(turns or [])
        self.__dict__["seen_bound_tools"] = []
        self.__dict__["ainvoke_calls"] = 0
        self.__dict__["astream_calls"] = 0

    @property
    def seen_bound_tools(self) -> list[list[Any]]:
        return self.__dict__["seen_bound_tools"]  # type: ignore[no-any-return]

    @property
    def ainvoke_calls(self) -> int:
        return self.__dict__["ainvoke_calls"]  # type: ignore[no-any-return]

    @property
    def astream_calls(self) -> int:
        return self.__dict__["astream_calls"]  # type: ignore[no-any-return]

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **_: Any,
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
        if turn.chunks:
            for chunk in turn.chunks:
                yield ChatGenerationChunk(message=chunk)
            return
        msg = turn.message or AIMessage(content="")
        synthetic = AIMessageChunk(
            content=msg.content,
            tool_call_chunks=[
                ToolCallChunk(
                    name=tc["name"],
                    args=json.dumps(tc.get("args", {})),
                    id=tc["id"],
                    index=i,
                )
                for i, tc in enumerate(msg.tool_calls or [])
            ],
        )
        yield ChatGenerationChunk(message=synthetic)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["ainvoke_calls"] += 1
        turn = self._pop_turn()
        return ChatResult(generations=[ChatGeneration(message=turn.resolved_message())])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        raise NotImplementedError("sync path not used")

    def _pop_turn(self) -> FakeTurn:
        turns: list[FakeTurn] = self.__dict__["_turns"]
        if not turns:
            raise RuntimeError("FakeChatModel: no scripted turns left")
        return turns.pop(0)


def text_chunk(text: str, *, final: bool = False) -> AIMessageChunk:
    if final:
        return AIMessageChunk(content=text, chunk_position="last")
    return AIMessageChunk(content=text)


def tool_call_chunk(
    name: str, args_json: str, id: str, *, final: bool = True,
) -> AIMessageChunk:
    tcc: list[ToolCallChunk] = [ToolCallChunk(name=name, args=args_json, id=id, index=0)]
    if final:
        return AIMessageChunk(
            content="", tool_call_chunks=tcc, chunk_position="last",
        )
    return AIMessageChunk(content="", tool_call_chunks=tcc)


def parallel_tool_calls_chunk(calls: list[dict[str, str]]) -> AIMessageChunk:
    tcc: list[ToolCallChunk] = [
        ToolCallChunk(name=c["name"], args=c["args_json"], id=c["id"], index=i)
        for i, c in enumerate(calls)
    ]
    return AIMessageChunk(
        content="", tool_call_chunks=tcc, chunk_position="last",
    )
