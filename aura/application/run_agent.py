"""run_agent — the free-function entry for one conversation turn.

The stateful host builds an ``AgentContext`` whose loop is configured from
``definition`` and delegates a turn here; tools and event sequencing live
entirely in the loop, so this stays a thin generator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import BaseMessage

from aura.application.agent_context import AgentContext
from aura.domain.agent_definition import AgentDefinition
from aura.domain.events import AgentEvent


async def run_agent(
    definition: AgentDefinition,
    messages: list[BaseMessage],
    context: AgentContext,
) -> AsyncIterator[AgentEvent]:
    # definition pins this turn's identity (prompt/tools/model/mode); the loop
    # carried by context was built from it, so dispatch is a thin delegate.
    _ = definition
    async for event in context.loop.run_turn(
        history=messages, abort=context.abort,
    ):
        yield event
