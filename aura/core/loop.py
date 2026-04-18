"""Aura agent loop."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from aura.core.events import AgentEvent, AssistantDelta, Final, ToolCallCompleted, ToolCallStarted
from aura.core.hooks import HookChain
from aura.core.registry import ToolRegistry
from aura.tools.base import ToolResult


def _serialize_tool_output(result: ToolResult) -> str:
    return json.dumps(result.output) if result.ok else (result.error or "tool failed")


async def run_turn(
    *,
    user_prompt: str,
    history: list[BaseMessage],
    model: BaseChatModel,
    registry: ToolRegistry,
    hooks: HookChain,
) -> AsyncIterator[AgentEvent]:
    history.append(HumanMessage(content=user_prompt))
    bound = model.bind_tools(registry.schemas()) if registry else model

    while True:
        await hooks.run_pre_model(history=history)

        raw = await bound.ainvoke(history)
        assert isinstance(raw, AIMessage)
        ai_msg = raw
        history.append(ai_msg)

        await hooks.run_post_model(ai_message=ai_msg, history=history)

        if ai_msg.content:
            yield AssistantDelta(text=str(ai_msg.content))

        if not ai_msg.tool_calls:
            yield Final(message=str(ai_msg.content))
            return

        for tc in ai_msg.tool_calls:
            args = tc["args"] if isinstance(tc["args"], dict) else {}
            yield ToolCallStarted(name=tc["name"], input=args)

            tool = registry[tc["name"]]
            params = tool.input_model.model_validate(tc["args"])

            short = await hooks.run_pre_tool(tool=tool, params=params)
            if short is not None:
                result = short
            else:
                result = await tool.acall(params)
                result = await hooks.run_post_tool(tool=tool, params=params, result=result)

            status: Literal["success", "error"] = "success" if result.ok else "error"
            history.append(
                ToolMessage(
                    content=_serialize_tool_output(result),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                    status=status,
                )
            )
            yield ToolCallCompleted(name=tc["name"], output=result.output, error=result.error)
