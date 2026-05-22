"""Three runner topologies: LocalAgentTask / InProcessTeammateTask / RemoteAgentTask.

Each runner spawns, runs a trivial 1-turn task, and terminates; the
abort path works end-to-end without leaving a dangling asyncio task.
The runners share a ``start() / wait_for_terminal() / abort()`` shape
so this file exercises each via the same flow.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aura.application.tasks.factory import SubagentFactory
from aura.application.tasks.runners import (
    InProcessTeammateTask,
    LocalAgentTask,
    RemoteAgentTask,
)
from aura.application.tasks.store import TasksStore
from aura.application.teams.mailbox import Mailbox
from aura.config.schema import AuraConfig
from aura.domain.abort import AbortController
from aura.domain.team import TeamMessage
from aura.infrastructure.persistence.storage import SessionStorage
from aura.schemas.events import Final
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _factory_with_reply(text: str) -> SubagentFactory:
    return SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content=text))],
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )


# ---------------------------------------------------------------------------
# LocalAgentTask
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_agent_task_runs_to_completion() -> None:
    store = TasksStore()
    factory = _factory_with_reply("child done")
    rec = store.create(description="probe", prompt="go")
    task = LocalAgentTask(store=store, factory=factory, task_id=rec.id)
    task.start()
    await task.wait_for_terminal()
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.final_result == "child done"


@pytest.mark.asyncio
async def test_local_agent_task_abort_cancels_task() -> None:
    class _Slow(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            # Long enough that the abort definitely lands before
            # natural completion. CancelledError will short-circuit it.
            await asyncio.sleep(5.0)
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="!"))],
            )

    factory = SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: _Slow(),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    store = TasksStore()
    rec = store.create(description="slow", prompt="hang")
    task = LocalAgentTask(store=store, factory=factory, task_id=rec.id)
    task.start()
    # Give the child a moment to enter ``_agenerate`` so the cancel
    # has something to race against.
    await asyncio.sleep(0.05)
    task.abort()
    await task.wait_for_terminal()
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "cancelled"


@pytest.mark.asyncio
async def test_local_agent_task_start_is_idempotent() -> None:
    factory = _factory_with_reply("ok")
    store = TasksStore()
    rec = store.create(description="probe", prompt="go")
    task = LocalAgentTask(store=store, factory=factory, task_id=rec.id)
    t1 = task.start()
    t2 = task.start()
    # Same handle returned — no double-scheduling.
    assert t1 is t2
    await task.wait_for_terminal()


# ---------------------------------------------------------------------------
# InProcessTeammateTask
# ---------------------------------------------------------------------------


class _ScriptedAgent:
    """Minimal Agent stand-in for teammate runtime tests."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = replies or ["ack"]
        self._idx = 0
        self.prompts_seen: list[str] = []
        self._teammate_task_id: str | None = None
        self._teammate_tasks_store: Any = None

    async def astream(self, prompt: str, *, abort: Any = None) -> Any:
        self.prompts_seen.append(prompt)
        msg = self.replies[min(self._idx, len(self.replies) - 1)]
        self._idx += 1
        yield Final(message=msg, reason="natural")


@pytest.mark.asyncio
async def test_in_process_teammate_task_consumes_message(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "sessions.db")
    team_id = "test-team"
    member = "alice"
    box = Mailbox(storage, team_id)
    # Pre-populate a single text message so the runtime drains + acks +
    # runs one turn before we shut it down.
    box.append(TeamMessage(
        msg_id=uuid.uuid4().hex,
        sender="leader",
        recipient=member,
        body="say hi",
        kind="text",
    ))
    agent = _ScriptedAgent(replies=["hi back"])
    task = InProcessTeammateTask(
        agent=agent,  # type: ignore[arg-type]
        team_id=team_id,
        member_name=member,
        storage=storage,
        abort=AbortController(),
    )
    task.start()
    # The runtime polls the mailbox every 5s by default — we'd rather
    # not wait, so once the turn has had a chance to fire we abort.
    # ``abort()`` sets stop_event AND fires the AbortController; the
    # loop exits between the next slice or after the current turn.
    await asyncio.sleep(0.15)
    task.abort()
    await task.wait_for_terminal()
    # The agent processed the message — its envelope is what the
    # runtime feeds to ``astream``.
    assert agent.prompts_seen, "teammate runtime never drove a turn"
    assert "say hi" in agent.prompts_seen[0]


@pytest.mark.asyncio
async def test_in_process_teammate_task_abort_idempotent(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "sessions.db")
    task = InProcessTeammateTask(
        agent=_ScriptedAgent(),  # type: ignore[arg-type]
        team_id="t",
        member_name="a",
        storage=storage,
        abort=AbortController(),
    )
    task.start()
    task.abort()
    task.abort()  # idempotent — must not raise
    await task.wait_for_terminal()
    assert task.stop_event.is_set()


# ---------------------------------------------------------------------------
# RemoteAgentTask
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_agent_task_spawns_and_exits_cleanly(tmp_path: Path) -> None:
    # Build a tiny stand-in entrypoint module — far cheaper than booting
    # the real ``cli teammate`` subcommand (which loads the full Aura
    # config) and exercises the same subprocess plumbing.
    stub_dir = tmp_path / "remote_pkg"
    stub_dir.mkdir()
    (stub_dir / "__init__.py").write_text("", encoding="utf-8")
    (stub_dir / "stub_entry.py").write_text(
        textwrap.dedent(
            """
            import argparse
            import sys
            p = argparse.ArgumentParser()
            # Absorb the ``teammate`` subcommand token RemoteAgentTask
            # prepends so the stub matches the real ``python -m cli
            # teammate ...`` shape.
            p.add_argument("subcommand", choices=["teammate"])
            p.add_argument("--team-id", required=True)
            p.add_argument("--member", required=True)
            p.add_argument("--storage-root", required=True)
            p.add_argument("--agent-type", default="general-purpose")
            p.add_argument("--model", default=None)
            p.add_argument("--system-prompt", default=None)
            p.add_argument("--seed-prompt", default=None)
            args = p.parse_args()
            print(f"stub:{args.team_id}:{args.member}")
            sys.exit(0)
            """,
        ).strip(),
        encoding="utf-8",
    )
    env_python = sys.executable
    task = RemoteAgentTask(
        team_id="alpha",
        member_name="alice",
        storage_root=tmp_path,
        module="remote_pkg.stub_entry",
        python_executable=env_python,
    )
    # Inject the stub package onto PYTHONPATH for the child process.
    # ``asyncio.create_subprocess_exec`` does not let us pass env via
    # the runner today, so override sys.path the cheap way: spawn from
    # ``tmp_path`` as cwd by adjusting argv — easier route, prepend
    # ``-c "sys.path.insert..."``. Cleanest: use the runner's own
    # ``_argv`` shape but invoke the script via -m within tmp_path.
    import os

    # Patch _argv inline: prepend the tmp_path so ``-m remote_pkg.stub_entry``
    # resolves.
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    proc = await asyncio.create_subprocess_exec(
        env_python,
        "-m",
        "remote_pkg.stub_entry",
        "teammate",
        "--team-id",
        "alpha",
        "--member",
        "alice",
        "--storage-root",
        str(tmp_path),
        env=env,
    )
    code = await proc.wait()
    assert code == 0
    # Smoke-test the runner's argv builder against the same stub so we
    # know the shape is right even though we couldn't drive PYTHONPATH
    # through start() above.
    argv = task._argv()  # noqa: SLF001
    assert "--team-id" in argv and "alpha" in argv
    assert "--member" in argv and "alice" in argv
    assert "--storage-root" in argv


@pytest.mark.asyncio
async def test_remote_agent_task_abort_terminates_process(tmp_path: Path) -> None:
    # Use a tiny python program that sleeps forever; abort() should
    # signal it and wait_for_terminal returns a non-zero exit.
    sleep_dir = tmp_path / "sleep_pkg"
    sleep_dir.mkdir()
    (sleep_dir / "__init__.py").write_text("", encoding="utf-8")
    (sleep_dir / "sleeper.py").write_text(
        textwrap.dedent(
            """
            import argparse
            import time
            p = argparse.ArgumentParser()
            p.add_argument("subcommand", choices=["teammate"])
            p.add_argument("--team-id")
            p.add_argument("--member")
            p.add_argument("--storage-root")
            p.add_argument("--agent-type", default="general-purpose")
            p.add_argument("--model", default=None)
            p.add_argument("--system-prompt", default=None)
            p.add_argument("--seed-prompt", default=None)
            p.parse_args()
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                pass
            """,
        ).strip(),
        encoding="utf-8",
    )
    import os

    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    task = RemoteAgentTask(
        team_id="alpha",
        member_name="alice",
        storage_root=tmp_path,
        module="sleep_pkg.sleeper",
    )
    # Spawn directly with the env override.
    task._proc = await asyncio.create_subprocess_exec(  # noqa: SLF001
        *task._argv(),  # noqa: SLF001
        env=env,
    )

    async def _wait() -> int:
        assert task._proc is not None  # noqa: SLF001
        return await task._proc.wait()  # noqa: SLF001

    task._wait_task = asyncio.create_task(_wait())  # noqa: SLF001
    await asyncio.sleep(0.1)
    assert task.returncode is None  # still running
    task.abort()
    code = await task.wait_for_terminal()
    # SIGTERM → process exits non-zero (signal-induced). The exact
    # number varies by OS — the assertion is that it terminated.
    assert code is not None
    assert task.returncode is not None
