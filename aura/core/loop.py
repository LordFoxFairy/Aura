"""AgentLoop — 协调一次对话的 turn 循环，驱动 model → tool → model 的迭代。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from pydantic import BaseModel, ValidationError

from aura.core.events import AgentEvent, AssistantDelta, Final, ToolCallCompleted, ToolCallStarted
from aura.core.hooks import HookChain
from aura.core.persistence import journal
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
        system_prompt: str | None = None,
    ) -> None:
        self._registry = registry
        self._hooks = hooks or HookChain()
        self._state = state or LoopState()
        self._system_prompt = system_prompt
        # bind_tools 只在构造时调一次：schema 在 registry 固定后不变，多 turn 共享同一 bound model。
        self._bound = model.bind_tools(registry.schemas()) if registry else model

    @property
    def state(self) -> LoopState:
        return self._state

    async def run_turn(
        self, *, user_prompt: str, history: list[BaseMessage]
    ) -> AsyncIterator[AgentEvent]:
        history.append(HumanMessage(content=user_prompt))

        while True:
            journal.write("turn_begin", turn=self._state.turn_count + 1)
            ai = await self._invoke_model(history)

            if ai.content:
                yield AssistantDelta(text=str(ai.content))
            if not ai.tool_calls:
                journal.write("turn_end", turn=self._state.turn_count, ended_with="final")
                yield Final(message=str(ai.content))
                return

            async for event in self._dispatch_tool_calls(ai.tool_calls, history):
                yield event
            journal.write(
                "turn_end",
                turn=self._state.turn_count,
                ended_with="tool_loop",
                tool_count=len(ai.tool_calls),
            )

    async def _invoke_model(self, history: list[BaseMessage]) -> AIMessage:
        # turn_count 先于 pre_model hook 递增，hook 看到的是"即将开始的第 N 轮"。
        self._state.turn_count += 1
        await self._hooks.run_pre_model(history=history, state=self._state)
        # SystemMessage 不持久化到 history —— 每次调 model 前临时插入，
        # 保证总是拿到最新环境（日期/cwd/tools）。
        messages = (
            [SystemMessage(content=self._system_prompt), *history]
            if self._system_prompt
            else history
        )
        ai = await self._bound.ainvoke(messages)
        history.append(ai)
        await self._hooks.run_post_model(
            ai_message=ai, history=history, state=self._state,
        )
        return ai

    async def _dispatch_tool_calls(
        self, tool_calls: list[ToolCall], history: list[BaseMessage],
    ) -> AsyncIterator[AgentEvent]:
        # plan（解析 + pre_tool hook）→ partition batches → execute：三段式分离关注点。
        steps = await self._plan_tool_calls(tool_calls)
        journal.write(
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

        batches = ToolRegistry.partition_batches(steps)
        for batch in batches:
            if len(batch) == 1:
                async for event in self._run_batch_serial(batch[0], history):
                    yield event
            else:
                async for event in self._run_batch_parallel(batch, history):
                    yield event

    async def _run_batch_serial(
        self, step: ToolStep, history: list[BaseMessage],
    ) -> AsyncIterator[AgentEvent]:
        tc = step.tool_call
        yield ToolCallStarted(name=tc["name"], input=dict(tc["args"]))
        journal.write(
            "tool_execute_begin", tool=tc["name"], tool_call_id=tc["id"],
        )
        result = await self._execute_step(step)
        journal.write(
            "tool_execute_end",
            tool=tc["name"], tool_call_id=tc["id"],
            ok=result.ok, error=result.error,
        )
        self._append_tool_message(history, tc, result)
        yield ToolCallCompleted(
            name=tc["name"], output=result.output, error=result.error,
        )

    async def _run_batch_parallel(
        self, batch: list[ToolStep], history: list[BaseMessage],
    ) -> AsyncIterator[AgentEvent]:
        journal.write(
            "tool_batch_begin",
            turn=self._state.turn_count,
            size=len(batch),
            tools=[s.tool_call.get("name") for s in batch],
        )
        # 所有 ToolCallStarted 在 gather 之前 yield：用户侧看到并行 tool call 同时宣告。
        for step in batch:
            tc = step.tool_call
            yield ToolCallStarted(name=tc["name"], input=dict(tc["args"]))
            journal.write(
                "tool_execute_begin", tool=tc["name"], tool_call_id=tc["id"],
            )

        results = await asyncio.gather(
            *(self._execute_step(s) for s in batch),
        )

        # invariant 1：zip strict=True 保证 ToolMessage.append 顺序严格等于 tool_call 顺序。
        for step, result in zip(batch, results, strict=True):
            tc = step.tool_call
            journal.write(
                "tool_execute_end",
                tool=tc["name"], tool_call_id=tc["id"],
                ok=result.ok, error=result.error,
            )
            self._append_tool_message(history, tc, result)
            yield ToolCallCompleted(
                name=tc["name"], output=result.output, error=result.error,
            )

        journal.write("tool_batch_end", turn=self._state.turn_count, size=len(batch))

    def _append_tool_message(
        self, history: list[BaseMessage], tc: ToolCall, result: ToolResult,
    ) -> None:
        status: Literal["success", "error"] = "success" if result.ok else "error"
        history.append(
            ToolMessage(
                content=_serialize(result),
                tool_call_id=tc["id"],
                name=tc["name"],
                status=status,
            )
        )

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
            # tool 作者的 acall 不要求永不抛异常；loop 在此兜底，保证结果总能写回 history。
            result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        return await self._hooks.run_post_tool(
            tool=step.tool, params=step.params, result=result, state=self._state
        )
