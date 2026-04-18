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
from pydantic import BaseModel, ValidationError

from aura.core.events import AgentEvent, AssistantDelta, Final, ToolCallCompleted, ToolCallStarted
from aura.core.hooks import HookChain
from aura.core.journal import journal
from aura.core.registry import ToolRegistry
from aura.core.state import LoopState
from aura.tools.base import AuraTool, ToolResult


def _serialize(result: ToolResult) -> str:
    return json.dumps(result.output) if result.ok else (result.error or "tool failed")


@dataclass(frozen=True)
class ToolStep:
    tool_call: ToolCall
    tool: AuraTool | None
    params: BaseModel | None
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
            journal().write("turn_begin", turn=self._state.turn_count + 1)
            ai = await self._invoke_model(history)

            if ai.content:
                yield AssistantDelta(text=str(ai.content))
            if not ai.tool_calls:
                journal().write("turn_end", turn=self._state.turn_count, ended_with="final")
                yield Final(message=str(ai.content))
                return

            async for event in self._dispatch_tool_calls(ai.tool_calls, history):
                yield event
            journal().write(
                "turn_end",
                turn=self._state.turn_count,
                ended_with="tool_loop",
                tool_count=len(ai.tool_calls),
            )

    async def _invoke_model(self, history: list[BaseMessage]) -> AIMessage:
        self._state.turn_count += 1
        await self._hooks.run_pre_model(history=history, state=self._state)
        ai = await self._bound.ainvoke(history)
        history.append(ai)
        await self._hooks.run_post_model(
            ai_message=ai, history=history, state=self._state,
        )
        return ai

    async def _dispatch_tool_calls(
        self, tool_calls: list[ToolCall], history: list[BaseMessage]
    ) -> AsyncIterator[AgentEvent]:
        steps = await self._plan_tool_calls(tool_calls)
        journal().write(
            "tool_plan_built",
            turn=self._state.turn_count,
            steps=[
                {
                    "tool": s.tool_call.get("name"),
                    "short_circuited": s.decision is not None,
                }
                for s in steps
            ],
        )
        for step in steps:
            tc = step.tool_call
            yield ToolCallStarted(name=tc["name"], input=dict(tc["args"]))

            journal().write(
                "tool_execute_begin",
                tool=tc["name"],
                tool_call_id=tc["id"],
            )
            result = await self._execute_step(step)
            journal().write(
                "tool_execute_end",
                tool=tc["name"],
                tool_call_id=tc["id"],
                ok=result.ok,
                error=result.error,
            )

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
            tool = self._registry.get(tc["name"])
            if tool is None:
                steps.append(ToolStep(
                    tool_call=tc, tool=None, params=None,
                    decision=ToolResult(ok=False, error=f"unknown tool: {tc['name']!r}"),
                ))
                continue

            try:
                params = tool.input_model.model_validate(tc["args"])
            except ValidationError as exc:
                steps.append(ToolStep(
                    tool_call=tc, tool=tool, params=None,
                    decision=ToolResult(ok=False, error=f"invalid args: {exc}"),
                ))
                continue

            decision = await self._hooks.run_pre_tool(
                tool=tool, params=params, state=self._state
            )
            steps.append(ToolStep(tool_call=tc, tool=tool, params=params, decision=decision))
        return steps

    async def _execute_step(self, step: ToolStep) -> ToolResult:
        if step.decision is not None:
            return step.decision
        assert step.tool is not None
        assert step.params is not None
        try:
            result = await step.tool.acall(step.params)
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        return await self._hooks.run_post_tool(
            tool=step.tool, params=step.params, result=result, state=self._state
        )
