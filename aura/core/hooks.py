"""Lifecycle hooks invoked by the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from aura.core.state import LoopState
from aura.tools.base import AuraTool, ToolResult


class PreModelHook(Protocol):
    async def __call__(
        self,
        *,
        history: list[BaseMessage],
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class PostModelHook(Protocol):
    async def __call__(
        self,
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class PreToolHook(Protocol):
    async def __call__(
        self,
        *,
        tool: AuraTool,
        params: BaseModel,
        state: LoopState,
        **kwargs: Any,
    ) -> ToolResult | None: ...


class PostToolHook(Protocol):
    async def __call__(
        self,
        *,
        tool: AuraTool,
        params: BaseModel,
        result: ToolResult,
        state: LoopState,
        **kwargs: Any,
    ) -> ToolResult: ...


@dataclass
class HookChain:
    pre_model: list[PreModelHook] = field(default_factory=list)
    post_model: list[PostModelHook] = field(default_factory=list)
    pre_tool: list[PreToolHook] = field(default_factory=list)
    post_tool: list[PostToolHook] = field(default_factory=list)

    async def run_pre_model(
        self, *, history: list[BaseMessage], state: LoopState,
    ) -> None:
        for hook in self.pre_model:
            await hook(history=history, state=state)

    async def run_post_model(
        self,
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],
        state: LoopState,
    ) -> None:
        for hook in self.post_model:
            await hook(ai_message=ai_message, history=history, state=state)

    async def run_pre_tool(
        self, *, tool: AuraTool, params: BaseModel, state: LoopState,
    ) -> ToolResult | None:
        for hook in self.pre_tool:
            decision = await hook(tool=tool, params=params, state=state)
            if decision is not None:
                return decision
        return None

    async def run_post_tool(
        self,
        *,
        tool: AuraTool,
        params: BaseModel,
        result: ToolResult,
        state: LoopState,
    ) -> ToolResult:
        for hook in self.post_tool:
            result = await hook(
                tool=tool, params=params, result=result, state=state,
            )
        return result

    def merge(self, other: HookChain) -> HookChain:
        return HookChain(
            pre_model=[*self.pre_model, *other.pre_model],
            post_model=[*self.post_model, *other.post_model],
            pre_tool=[*self.pre_tool, *other.pre_tool],
            post_tool=[*self.post_tool, *other.post_tool],
        )
