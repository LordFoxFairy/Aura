"""Aura agent loop."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from pydantic import BaseModel

from aura.core.events import AgentEvent, AssistantDelta, Final, ToolCallCompleted, ToolCallStarted
from aura.core.hooks import HookChain
from aura.core.registry import ToolRegistry
from aura.core.state import LoopState
from aura.tools.base import AuraTool, ToolResult


def _serialize(result: ToolResult) -> str:
    return json.dumps(result.output) if result.ok else (result.error or "tool failed")


@dataclass(frozen=True)
class ToolStep:
    tool_call: ToolCall
    tool: AuraTool
    params: BaseModel
    decision: ToolResult | None


class AgentLoop:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        registry: ToolRegistry,
        hooks: HookChain | None = None,
        state: LoopState | None = None,
    ) -> None:
        self._registry = registry
        self._hooks = hooks or HookChain()
        self._state = state or LoopState()
        self._bound = model.bind_tools(registry.schemas()) if registry else model

    @property
    def state(self) -> LoopState:
        return self._state

    async def run_turn(
        self, *, user_prompt: str, history: list[BaseMessage]
    ) -> AsyncIterator[AgentEvent]:
        history.append(HumanMessage(content=user_prompt))

        while True:
            ai = await self._invoke_model(history)

            if ai.content:
                yield AssistantDelta(text=str(ai.content))
            if not ai.tool_calls:
                yield Final(message=str(ai.content))
                return

            async for event in self._dispatch_tool_calls(ai.tool_calls, history):
                yield event

    async def _invoke_model(self, history: list[BaseMessage]) -> AIMessage:
        self._state.turn_count += 1
        await self._hooks.run_pre_model(history=history, state=self._state)
        ai = await self._bound.ainvoke(history)
        history.append(ai)
        await self._hooks.run_post_model(ai_message=ai, history=history, state=self._state)
        return ai

    async def _dispatch_tool_calls(
        self, tool_calls: list[ToolCall], history: list[BaseMessage]
    ) -> AsyncIterator[AgentEvent]:
        steps = await self._plan_tool_calls(tool_calls)
        for step in steps:
            tc = step.tool_call
            yield ToolCallStarted(name=tc["name"], input=dict(tc["args"]))

            result = await self._execute_step(step)

            status: Literal["success", "error"] = "success" if result.ok else "error"
            history.append(
                ToolMessage(
                    content=_serialize(result),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                    status=status,
                )
            )
            yield ToolCallCompleted(name=tc["name"], output=result.output, error=result.error)

    async def _plan_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolStep]:
        steps: list[ToolStep] = []
        for tc in tool_calls:
            tool = self._registry[tc["name"]]
            params = tool.input_model.model_validate(tc["args"])
            decision = await self._hooks.run_pre_tool(
                tool=tool, params=params, state=self._state
            )
            steps.append(ToolStep(tool_call=tc, tool=tool, params=params, decision=decision))
        return steps

    async def _execute_step(self, step: ToolStep) -> ToolResult:
        if step.decision is not None:
            return step.decision
        result = await step.tool.acall(step.params)
        return await self._hooks.run_post_tool(
            tool=step.tool, params=step.params, result=result, state=self._state
        )
