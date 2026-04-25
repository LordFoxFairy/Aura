"""run_teammate — long-lived loop driving an Agent on mailbox messages."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

from aura.core.abort import AbortController
from aura.core.persistence.storage import SessionStorage
from aura.core.teams.mailbox import Mailbox
from aura.core.teams.runtime import _format_envelope, run_teammate
from aura.core.teams.types import TeamMessage
from aura.schemas.events import Final


def _msg(body: str = "hi", kind: str = "text", sender: str = "leader") -> TeamMessage:
    return TeamMessage(
        msg_id=uuid.uuid4().hex,
        sender=sender,
        recipient="alice",
        body=body,
        kind=kind,  # type: ignore[arg-type]
    )


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "sessions.db")


class _ScriptedAgent:
    """Minimal Agent stand-in: yields a Final per astream call.

    Records every prompt it sees so tests can assert envelope shape.
    """

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = replies or ["ack"]
        self.prompts_seen: list[str] = []
        self._idx = 0

    async def astream(self, prompt: str, *, abort: Any = None) -> Any:
        self.prompts_seen.append(prompt)
        msg = self.replies[min(self._idx, len(self.replies) - 1)]
        self._idx += 1
        yield Final(message=msg, reason="natural")


def test_format_envelope_wraps_each_sender() -> None:
    msgs = [
        _msg(body="hello", sender="leader"),
        _msg(body="follow up", sender="leader"),
    ]
    out = _format_envelope(msgs)
    assert "<from-leader>" in out
    assert "</from-leader>" in out
    assert "hello" in out
    assert "follow up" in out


@pytest.mark.asyncio
async def test_runtime_processes_text_message(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    box = Mailbox(storage, "team-a")
    agent = _ScriptedAgent(replies=["got it"])
    abort = AbortController()
    stop = asyncio.Event()
    box.append(_msg(body="please work"))
    task = asyncio.create_task(run_teammate(
        agent=agent,  # type: ignore[arg-type]
        team_id="team-a",
        member_name="alice",
        storage=storage,
        stop_event=stop,
        abort=abort,
    ))
    # Give the runtime a window to consume the message.
    for _ in range(40):
        await asyncio.sleep(0.05)
        if agent.prompts_seen:
            break
    stop.set()
    await asyncio.wait_for(task, timeout=10)
    assert agent.prompts_seen, "runtime never consumed the message"
    assert "please work" in agent.prompts_seen[0]
    # And the .seen cursor advanced — no more unseen.
    assert box.read_unseen("alice") == []


@pytest.mark.asyncio
async def test_runtime_seed_prompt_runs_immediately(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    agent = _ScriptedAgent(replies=["seed-ack"])
    abort = AbortController()
    stop = asyncio.Event()
    task = asyncio.create_task(run_teammate(
        agent=agent,  # type: ignore[arg-type]
        team_id="team-a",
        member_name="alice",
        storage=storage,
        stop_event=stop,
        abort=abort,
        seed_prompt="please scan the repo",
    ))
    # Wait until seed prompt is consumed
    for _ in range(40):
        await asyncio.sleep(0.05)
        if agent.prompts_seen:
            break
    stop.set()
    await asyncio.wait_for(task, timeout=10)
    assert agent.prompts_seen[0] == "please scan the repo"


@pytest.mark.asyncio
async def test_runtime_shutdown_request_exits_cleanly(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    box = Mailbox(storage, "team-a")
    agent = _ScriptedAgent()
    abort = AbortController()
    stop = asyncio.Event()
    box.append(_msg(kind="shutdown_request", body="please go"))
    task = asyncio.create_task(run_teammate(
        agent=agent,  # type: ignore[arg-type]
        team_id="team-a",
        member_name="alice",
        storage=storage,
        stop_event=stop,
        abort=abort,
    ))
    await asyncio.wait_for(task, timeout=15)
    # No model invocation — shutdown short-circuits before the turn.
    assert agent.prompts_seen == []


@pytest.mark.asyncio
async def test_runtime_abort_stops_loop(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    agent = _ScriptedAgent()
    abort = AbortController()
    stop = asyncio.Event()
    task = asyncio.create_task(run_teammate(
        agent=agent,  # type: ignore[arg-type]
        team_id="team-a",
        member_name="alice",
        storage=storage,
        stop_event=stop,
        abort=abort,
    ))
    await asyncio.sleep(0.1)
    abort.abort("test")
    await asyncio.wait_for(task, timeout=15)
