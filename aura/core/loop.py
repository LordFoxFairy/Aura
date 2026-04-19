"""AgentLoop — 协调一次对话的 turn 循环，驱动 model → tool → model 的迭代。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from aura.core.hooks import HookChain
from aura.core.memory.context import Context
from aura.core.persistence import journal
from aura.core.registry import ToolRegistry
from aura.schemas.events import (
    AgentEvent,
    AssistantDelta,
    Final,
    ToolCallCompleted,
    ToolCallStarted,
)
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolError, ToolResult

# 成功调用后需把路径反馈给 Context progressive 状态的工具 → 其 path 参数名。
# bash（shell 语义不固定）和 web_fetch（URL 而非文件系统）刻意排除。
PATH_TRIGGER_TOOLS: dict[str, str] = {
    "read_file": "path",
    "write_file": "path",
    "edit_file": "path",
    "grep": "path",
    "glob": "path",
}


def _serialize(result: ToolResult) -> str:
    # `default=str` + `ensure_ascii=False`：遇到非 JSON-native 值
    # （datetime / Path / bytes）降级为字符串而非抛异常。
    # 这里抛出会导致那一条 ToolMessage 漏 append —— 破坏 tool_call.id 与
    # ToolMessage 的严格对齐。
    if result.ok:
        return json.dumps(result.output, default=str, ensure_ascii=False)
    return result.error or "tool failed"


@dataclass(frozen=True)
class ToolStep:
    tool_call: ToolCall
    tool: BaseTool | None
    args: dict[str, object] | None
    decision: ToolResult | None


def partition_batches(steps: list[ToolStep]) -> list[list[ToolStep]]:
    """将 steps 按并发安全性分批（保序，不重排）。

    1. 连续的 is_concurrency_safe 且 decision=None 的 step 合并成一个并行 batch，
       批内用 gather 一次并发执行并保序拿回结果。
    2. 非 safe 或已被 pre_tool 短路（decision 非 None）的 step 单独成 batch。
    3. 维持原 tool_call 顺序 —— 并发只发生在 batch 内，不跨 batch。
    """
    batches: list[list[ToolStep]] = []
    current: list[ToolStep] = []
    for step in steps:
        tool = step.tool
        safe = (
            tool is not None
            and (tool.metadata or {}).get("is_concurrency_safe", False)
            and step.decision is None
        )
        if safe:
            current.append(step)
            continue
        if current:
            batches.append(current)
            current = []
        batches.append([step])
    if current:
        batches.append(current)
    return batches


class AgentLoop:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        registry: ToolRegistry,
        context: Context,
        hooks: HookChain | None = None,
        state: LoopState | None = None,
    ) -> None:
        self._registry = registry
        self._hooks = hooks or HookChain()
        self._state = state or LoopState()
        self._context = context
        # bind_tools 只在构造时调一次：schema 在 registry 固定后不变，多 turn 共享同一 bound model。
        # 空 registry 跳过 bind_tools —— 某些 provider 对 tools=[] 行为不一致，直接不绑更稳。
        self._bound = model.bind_tools(registry.tools()) if len(registry) > 0 else model

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
        # Context.build 是整个代码库中唯一组装 messages 的构造点 ——
        # 不要在别处拼 SystemMessage/HumanMessage 传给模型。
        messages = self._context.build(history)
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

        for batch in partition_batches(steps):
            async for event in self._run_batch(batch, history):
                yield event

    async def _run_batch(
        self, batch: list[ToolStep], history: list[BaseMessage],
    ) -> AsyncIterator[AgentEvent]:
        # size=1 也走统一 gather 路径 —— 单工具开销只多几行 journal，换来
        # 批量分发逻辑的单一实现。
        # 保序：所有 Started 在 gather 之前 yield（并发 tool call 同时宣告开始），
        # Completed 按 tool_call 顺序依次 yield —— 保证 tool_call.id 与
        # ToolMessage 的严格一一对齐，provider API 才能正确串联。
        journal.write(
            "tool_batch_begin",
            turn=self._state.turn_count,
            size=len(batch),
            tools=[s.tool_call.get("name") for s in batch],
        )
        for step in batch:
            tc = step.tool_call
            yield ToolCallStarted(name=tc["name"], input=dict(tc["args"]))
            journal.write(
                "tool_execute_begin", tool=tc["name"], tool_call_id=tc["id"],
            )

        results = await asyncio.gather(*(self._execute_step(s) for s in batch))

        for step, result in zip(batch, results, strict=True):
            tc = step.tool_call
            journal.write(
                "tool_execute_end",
                tool=tc["name"], tool_call_id=tc["id"],
                ok=result.ok, error=result.error,
            )
            # tool_call.id 与 ToolMessage 必须严格一一对齐 —— 这里直接 append，
            # 不抽函数；`_serialize` 已保证绝不抛出（default=str 兜底）。
            status: Literal["success", "error"] = "success" if result.ok else "error"
            history.append(
                ToolMessage(
                    content=_serialize(result),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                    status=status,
                )
            )
            yield ToolCallCompleted(
                name=tc["name"], output=result.output, error=result.error,
            )

        journal.write("tool_batch_end", turn=self._state.turn_count, size=len(batch))

    async def _plan_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolStep]:
        steps: list[ToolStep] = []
        for tc in tool_calls:
            tool = self._registry.get(tc["name"])
            if tool is None:
                steps.append(ToolStep(
                    tool_call=tc, tool=None, args=None,
                    decision=ToolResult(ok=False, error=f"unknown tool: {tc['name']!r}"),
                ))
                continue

            # 早校验：在派发前把 pydantic 错误转成 decision 短路 —— pre_tool hook
            # 不该看到 invalid args（否则 permission/budget 基于错误假设做决策）。
            raw_args = dict(tc["args"])
            schema = tool.args_schema
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                try:
                    schema.model_validate(raw_args)
                except ValidationError as exc:
                    steps.append(ToolStep(
                        tool_call=tc, tool=tool, args=None,
                        decision=ToolResult(ok=False, error=f"invalid args: {exc}"),
                    ))
                    continue

            decision = await self._hooks.run_pre_tool(
                tool=tool, args=raw_args, state=self._state
            )
            steps.append(ToolStep(tool_call=tc, tool=tool, args=raw_args, decision=decision))
        return steps

    async def _execute_step(self, step: ToolStep) -> ToolResult:
        if step.decision is not None:
            return step.decision
        assert step.tool is not None
        assert step.args is not None
        try:
            output = await step.tool.ainvoke(step.args)
            result = ToolResult(ok=True, output=output)
            self._maybe_trigger_path(step)
        except ToolError as exc:
            # 工具作者主动抛的用户态错误，消息原样给模型看。
            result = ToolResult(ok=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            # 任意异常兜底：保证结果总能写回 history，避免 tool_call id 漏匹配。
            # CancelledError 继承 BaseException（非 Exception），不会被这里吞掉。
            result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        return await self._hooks.run_post_tool(
            tool=step.tool, args=step.args, result=result, state=self._state
        )

    def _maybe_trigger_path(self, step: ToolStep) -> None:
        """成功的 path-aware tool 调用后，把路径反馈给 Context 的 progressive 状态。

        仅在 `_execute_step` 的成功分支（decision 为 None + ainvoke 未抛）被调用，
        因此 `step.tool` 与 `step.args` 必非 None —— 由 `_plan_tool_calls` 保证。
        """
        arg_name = PATH_TRIGGER_TOOLS.get(step.tool.name)  # type: ignore[union-attr]
        if arg_name is None:
            return
        raw = step.args.get(arg_name)  # type: ignore[union-attr]
        if not isinstance(raw, str) or not raw:
            return
        try:
            resolved = Path(raw).resolve()
        except OSError:
            return
        self._context.on_tool_touched_path(resolved)
