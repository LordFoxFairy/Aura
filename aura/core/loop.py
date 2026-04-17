"""The Aura agent loop — run_turn() drives one user prompt to completion.

Non-negotiable invariants (spec §4.2, enforced across all loop tests):
1. Every AIMessage.tool_calls[i].id gets a matching ToolMessage before next model call.
2. Cancellation discards partial AIMessage — no half-written turn in history.
3. bind_tools() receives the canonical LangChain tool schema dict per registered tool.
4. AssistantDelta events yield as chunks arrive (not buffered).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from aura.core.events import AgentEvent, AssistantDelta, Final, ToolCallCompleted, ToolCallStarted
from aura.core.history import tool_schema_for
from aura.tools.base import AuraTool, ToolResult


async def run_turn(
    *,
    user_prompt: str,
    history: list[BaseMessage],
    model: BaseChatModel,
    registry: dict[str, AuraTool],
    provider: str,
) -> AsyncIterator[AgentEvent]:
    """Run one user turn to completion, yielding events as the stream progresses.

    Mutates `history` in place: appends the HumanMessage, then the AIMessage once
    the stream completes. Does NOT append anything if the stream is cancelled —
    see Task 23 for cancellation handling.

    `provider` is the protocol string ("openai" / "anthropic" / "ollama"); the
    Ollama-with-tools branch (Task 22) will read it to choose ainvoke over astream.

    Invariant 1: Every AIMessage.tool_calls[i].id gets a matching ToolMessage
    appended to history before the next model call (outer while loop iteration).
    """
    history.append(HumanMessage(content=user_prompt))

    bound = model.bind_tools([tool_schema_for(t) for t in registry.values()]) if registry else model

    while True:
        acc: AIMessageChunk | None = None
        async for raw_chunk in bound.astream(history):
            # raw_chunk is always AIMessageChunk in streaming paths
            chunk = cast(AIMessageChunk, raw_chunk)
            if chunk.content:
                yield AssistantDelta(text=str(chunk.content))
            acc = chunk if acc is None else acc + chunk

        if acc is None:
            # Model produced zero chunks — edge case. Still append an empty AIMessage
            # to keep the loop invariants coherent.
            ai_msg = AIMessage(content="")
        else:
            ai_msg = AIMessage(content=acc.content, tool_calls=list(acc.tool_calls or []))

        history.append(ai_msg)

        if not ai_msg.tool_calls:
            yield Final(message=str(ai_msg.content))
            return

        # Dispatch each tool call serially (invariant 1: one ToolMessage per tool_call.id).
        for tc in ai_msg.tool_calls:
            name = tc["name"]
            args = tc["args"]
            tool_call_id = tc["id"]

            yield ToolCallStarted(name=name, input=dict(args) if isinstance(args, dict) else {})

            tool = registry[name]  # Task 20 adds the unknown-tool guard
            params = tool.input_model.model_validate(args)
            result = await tool.acall(params)

            content = _serialize_tool_output(result)
            status: Literal["success", "error"] = "success" if result.ok else "error"
            history.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name=name,
                    status=status,
                )
            )
            yield ToolCallCompleted(name=name, output=result.output, error=result.error)

        # Loop back to the next model call — history now has the AIMessage
        # plus one ToolMessage per tool_call.


def _serialize_tool_output(result: ToolResult) -> str:
    """Serialize a ToolResult to a JSON string for ToolMessage.content.

    ToolMessage.content is str. We JSON-encode the output so the model sees
    structured data. On failure, we send the error message (not the output).
    """
    if result.ok:
        return json.dumps(result.output)
    return result.error or "tool failed"
