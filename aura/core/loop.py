"""The Aura agent loop — run_turn() drives one user prompt to completion.

Non-negotiable invariants (spec §4.2, enforced across all loop tests):
1. Every AIMessage.tool_calls[i].id gets a matching ToolMessage before next model call.
2. Cancellation discards partial AIMessage — no half-written turn in history.
3. bind_tools() receives the canonical LangChain tool schema dict per registered tool.
4. AssistantDelta events yield as chunks arrive (not buffered).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage

from aura.core.events import AgentEvent, AssistantDelta, Final
from aura.core.history import tool_schema_for
from aura.tools.base import AuraTool


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
    For now (Task 16) we only handle the no-tool-calls path.
    """
    history.append(HumanMessage(content=user_prompt))

    bound = model.bind_tools([tool_schema_for(t) for t in registry.values()]) if registry else model

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

    # Task 16 scope: no tool-call dispatch yet. If acc has tool_calls, future tasks handle it.
    # For now just emit Final for the text-only path.
    if not ai_msg.tool_calls:
        yield Final(message=str(ai_msg.content))
