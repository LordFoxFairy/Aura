"""Lifecycle hooks invoked by the agent loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from aura.tools.base import AuraTool, ToolResult

PreModelHook = Callable[..., Awaitable[None]]
PostModelHook = Callable[..., Awaitable[None]]
PreToolHook = Callable[..., Awaitable[ToolResult | None]]
PostToolHook = Callable[..., Awaitable[ToolResult]]


@dataclass
class HookChain:
    pre_model: list[PreModelHook] = field(default_factory=list)
    post_model: list[PostModelHook] = field(default_factory=list)
    pre_tool: list[PreToolHook] = field(default_factory=list)
    post_tool: list[PostToolHook] = field(default_factory=list)

    async def run_pre_model(self, *, history: list[BaseMessage]) -> None:
        for hook in self.pre_model:
            await hook(history=history)

    async def run_post_model(self, *, ai_message: AIMessage, history: list[BaseMessage]) -> None:
        for hook in self.post_model:
            await hook(ai_message=ai_message, history=history)

    async def run_pre_tool(
        self, *, tool: AuraTool, params: BaseModel
    ) -> ToolResult | None:
        for hook in self.pre_tool:
            decision = await hook(tool=tool, params=params)
            if decision is not None:
                return decision
        return None

    async def run_post_tool(
        self, *, tool: AuraTool, params: BaseModel, result: ToolResult
    ) -> ToolResult:
        for hook in self.post_tool:
            result = await hook(tool=tool, params=params, result=result)
        return result
