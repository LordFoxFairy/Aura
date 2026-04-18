"""4 个生命周期 hook Protocol + HookChain — **kwargs: Any 保证向前兼容。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from aura.core.state import LoopState
from aura.tools.base import AuraTool, ToolResult


class PreModelHook(Protocol):
    # 可原地 mutate history（compact / inject system message 等场景）；无返回值。
    async def __call__(
        self,
        *,
        history: list[BaseMessage],
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class PostModelHook(Protocol):
    # 只读观察（usage 累计 / audit log）；不得修改 history 或 ai_message。
    async def __call__(
        self,
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class PreToolHook(Protocol):
    # 返回 ToolResult = 短路，记入 history 但不调 acall；返回 None = 放行。
    async def __call__(
        self,
        *,
        tool: AuraTool,
        params: BaseModel,
        state: LoopState,
        **kwargs: Any,
    ) -> ToolResult | None: ...


class PostToolHook(Protocol):
    # 链式调用：上一个 hook 的输出作为下一个 hook 的 result 输入（变换而非观察）。
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
        # 非破坏性拼接：self 优先 other 后；不修改任何一方的原始列表。
        return HookChain(
            pre_model=[*self.pre_model, *other.pre_model],
            post_model=[*self.post_model, *other.post_model],
            pre_tool=[*self.pre_tool, *other.pre_tool],
            post_tool=[*self.post_tool, *other.post_tool],
        )
