"""End-to-end multi-session scenario — two concurrent Agents in the same
process write to fully-isolated per-session journals.

This pins the contract at the Agent.astream boundary, not just the
journal unit level (which test_journal_session_scope.py covers). The
entire flow — Agent init takes session_log_dir, astream wraps its body
in session_scope, every nested journal.write inside the loop + hooks
routes correctly — is exercised here.

The key invariant: even when A and B run interleaved via asyncio.gather
in the same event loop, A's events go ONLY to A's log and B's to B's.
If this fails, either contextvars aren't scoping per-task, or astream
is leaking the scope somewhere.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _minimal_config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


@pytest.mark.asyncio
async def test_two_agents_same_process_write_to_separate_logs(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"

    # Slow fakes so the two tasks genuinely interleave — a turn that
    # completes synchronously on the first await wouldn't prove contextvars
    # kept the scope task-local, because there'd be no context switch.
    class _SlowFake(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **_: Any,
        ) -> Any:
            await asyncio.sleep(0.01)
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **_,
            )

    # Hand-roll agents with the slow model.
    def _slow_agent(session_id: str, content: str) -> Agent:
        model = _SlowFake(turns=[FakeTurn(AIMessage(content=content))])  # type: ignore[call-arg]
        storage = SessionStorage(tmp_path / f"{session_id}.db")
        return Agent(
            config=_minimal_config(),
            model=model,
            storage=storage,
            session_id=session_id,
            session_log_dir=log_dir,
            auto_compact_threshold=0,
        )

    agent_a = _slow_agent("alpha", "from A")
    agent_b = _slow_agent("beta", "from B")

    async def run(agent: Agent, prompt: str) -> list[Any]:
        events: list[Any] = []
        async for e in agent.astream(prompt):
            events.append(e)
        return events

    await asyncio.gather(
        run(agent_a, "hi A"),
        run(agent_b, "hi B"),
    )

    a_log = (log_dir / "alpha.jsonl").read_text()
    b_log = (log_dir / "beta.jsonl").read_text()

    # Key isolation: prompt preview is in ONLY its own log.
    assert "hi A" in a_log
    assert "hi A" not in b_log
    assert "hi B" in b_log
    assert "hi B" not in a_log

    # Session id tagging sanity — each event carries its session, and the
    # wrong session's id never appears in the other log.
    assert "alpha" in a_log
    assert "alpha" not in b_log
    assert "beta" in b_log
    assert "beta" not in a_log

    agent_a.close()
    agent_b.close()
