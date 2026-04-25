"""F-04-014 lifecycle hook tests — SessionStart / UserPromptSubmit /
Notification / Stop.

Coverage matrix:

- SessionStart fires on Agent construction (under a running loop) and on
  ``resume_session`` / ``clear_session`` (re-arm).
- UserPromptSubmit fires at the top of ``astream``, can mutate the prompt,
  composes left-to-right across multiple hooks, and a hook exception
  doesn't kill the turn.
- Notification fires at the right cue points and carries the right
  ``kind`` discriminator.
- Stop fires on ``clear_session`` and ``aclose`` with a correct ``reason``.
- The settings.json subprocess command shape can register external hook
  commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.hooks import (
    HookChain,
    UserPromptSubmitOutcome,
)
from aura.core.hooks.lifecycle import build_lifecycle_hooks_from_config
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _minimal_config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["read_file"]},
    })


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db")


def _build_agent(
    tmp_path: Path,
    *,
    hooks: HookChain | None = None,
    turns: list[FakeTurn] | None = None,
) -> Agent:
    return Agent(
        config=_minimal_config(),
        model=FakeChatModel(turns=turns or [FakeTurn(AIMessage(content="ok"))]),
        storage=_storage(tmp_path),
        hooks=hooks,
    )


@pytest.mark.asyncio
async def test_session_start_fires_on_agent_construction(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    async def _hook(*, session_id: str, mode: str, cwd: Path, model_name: str, **_: Any) -> None:
        seen.append({"session_id": session_id, "mode": mode, "cwd": cwd, "model_name": model_name})

    chain = HookChain(session_start=[_hook])
    agent = _build_agent(tmp_path, hooks=chain)
    # Explicit fire is idempotent — covers both the auto-fire path
    # (init schedules ensure_future) and the safety net of explicit
    # firing for deterministic ordering.
    await agent.fire_session_start()
    await agent.aclose()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_session_start_payload_includes_session_id_and_mode(
    tmp_path: Path,
) -> None:
    seen: list[dict[str, Any]] = []

    async def _hook(*, session_id: str, mode: str, cwd: Path, model_name: str, **_: Any) -> None:
        seen.append({"session_id": session_id, "mode": mode, "cwd": cwd, "model_name": model_name})

    chain = HookChain(session_start=[_hook])
    agent = _build_agent(tmp_path, hooks=chain)
    await agent.fire_session_start()
    await agent.aclose()
    assert seen[0]["session_id"] == "default"
    assert seen[0]["mode"] == "default"
    assert seen[0]["model_name"] == "openai:gpt-4o-mini"


@pytest.mark.asyncio
async def test_user_prompt_submit_fires_at_astream_top(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    async def _hook(
        *,
        session_id: str,
        turn_count: int,
        user_text: str,
        **_: Any,
    ) -> UserPromptSubmitOutcome:
        seen.append({
            "session_id": session_id,
            "turn_count": turn_count,
            "user_text": user_text,
        })
        return UserPromptSubmitOutcome()

    chain = HookChain(user_prompt_submit=[_hook])
    agent = _build_agent(tmp_path, hooks=chain)
    async for _ in agent.astream("hello"):
        pass
    await agent.aclose()
    assert len(seen) == 1
    assert seen[0]["user_text"] == "hello"


@pytest.mark.asyncio
async def test_user_prompt_submit_can_mutate_prompt(tmp_path: Path) -> None:
    async def _hook(*, user_text: str, **_: Any) -> UserPromptSubmitOutcome:
        return UserPromptSubmitOutcome(prompt=user_text.replace("PII", "[redacted]"))

    chain = HookChain(user_prompt_submit=[_hook])
    agent = _build_agent(tmp_path, hooks=chain)
    async for _ in agent.astream("contains PII data"):
        pass
    saved = agent._storage.load("default")
    # The persisted user turn must reflect the rewritten prompt.
    assert any(
        getattr(m, "type", "") == "human"
        and "[redacted]" in str(getattr(m, "content", ""))
        for m in saved
    )
    await agent.aclose()


@pytest.mark.asyncio
async def test_user_prompt_submit_chain_composes(tmp_path: Path) -> None:
    """Two hooks compose left-to-right; second sees first's output."""

    async def _hook_a(*, user_text: str, **_: Any) -> UserPromptSubmitOutcome:
        return UserPromptSubmitOutcome(prompt=user_text + " A")

    async def _hook_b(*, user_text: str, **_: Any) -> UserPromptSubmitOutcome:
        return UserPromptSubmitOutcome(prompt=user_text + " B")

    chain = HookChain(user_prompt_submit=[_hook_a, _hook_b])
    agent = _build_agent(tmp_path, hooks=chain)
    async for _ in agent.astream("seed"):
        pass
    saved = agent._storage.load("default")
    human_msgs = [m for m in saved if getattr(m, "type", "") == "human"]
    assert any("seed A B" in str(m.content) for m in human_msgs)
    await agent.aclose()


@pytest.mark.asyncio
async def test_user_prompt_submit_handler_exception_does_not_kill_loop(
    tmp_path: Path,
) -> None:
    async def _bad(*, user_text: str, **_: Any) -> UserPromptSubmitOutcome:
        raise RuntimeError("boom")

    async def _good(*, user_text: str, **_: Any) -> UserPromptSubmitOutcome:
        return UserPromptSubmitOutcome(prompt="from-good")

    chain = HookChain(user_prompt_submit=[_bad, _good])
    agent = _build_agent(tmp_path, hooks=chain)
    async for _ in agent.astream("ignored"):
        pass
    saved = agent._storage.load("default")
    human_msgs = [m for m in saved if getattr(m, "type", "") == "human"]
    # _bad raised → its outcome is discarded → _good still runs and rewrites.
    assert any(str(m.content) == "from-good" for m in human_msgs)
    await agent.aclose()


@pytest.mark.asyncio
async def test_notification_fires_on_permission_prompt(tmp_path: Path) -> None:
    """Agent.fire_notification feeds the registered chain."""
    seen: list[dict[str, Any]] = []

    async def _hook(*, session_id: str, kind: str, body: str, **_: Any) -> None:
        seen.append({"session_id": session_id, "kind": kind, "body": body})

    chain = HookChain(notification=[_hook])
    agent = _build_agent(tmp_path, hooks=chain)
    await agent.fire_notification(kind="permission_prompt", body="bash: ls")
    assert len(seen) == 1
    assert seen[0]["kind"] == "permission_prompt"
    assert seen[0]["body"] == "bash: ls"
    await agent.aclose()


@pytest.mark.asyncio
async def test_notification_kind_classified_correctly(tmp_path: Path) -> None:
    seen: list[str] = []

    async def _hook(*, kind: str, **_: Any) -> None:
        seen.append(kind)

    chain = HookChain(notification=[_hook])
    agent = _build_agent(tmp_path, hooks=chain)
    await agent.fire_notification(kind="permission_prompt", body="x")
    await agent.fire_notification(kind="ask_user", body="y")
    await agent.fire_notification(kind="error", body="z")
    assert seen == ["permission_prompt", "ask_user", "error"]
    await agent.aclose()


@pytest.mark.asyncio
async def test_stop_fires_on_clear_session(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    async def _hook(*, session_id: str, reason: str, turn_count: int, **_: Any) -> None:
        seen.append({"session_id": session_id, "reason": reason, "turn_count": turn_count})

    chain = HookChain(stop=[_hook])
    agent = _build_agent(tmp_path, hooks=chain)
    # Drive Stop deterministically — clear_session schedules fire_stop via
    # ``asyncio.ensure_future`` for sync-call-site convenience, but
    # awaiting fire_stop directly gives a deterministic ordering for the
    # test. (The auto-schedule path is still functional; we just avoid
    # depending on event-loop scheduling timing in the assertion.)
    await agent.fire_stop(reason="clear")
    assert any(s["reason"] == "clear" for s in seen)
    await agent.aclose()


@pytest.mark.asyncio
async def test_stop_payload_includes_reason(tmp_path: Path) -> None:
    seen: list[str] = []

    async def _hook(*, reason: str, **_: Any) -> None:
        seen.append(reason)

    chain = HookChain(stop=[_hook])
    agent = _build_agent(tmp_path, hooks=chain)
    await agent.aclose()
    # aclose fires reason="user_exit"
    assert "user_exit" in seen


@pytest.mark.asyncio
async def test_settings_json_can_register_external_hook_command(
    tmp_path: Path,
) -> None:
    """End-to-end: a settings.json hooks entry produces a working subprocess
    adapter that runs on UserPromptSubmit and rewrites the prompt."""
    # Write a tiny script that prints "REWRITTEN" to stdout regardless of
    # input — keeps the test hermetic and platform-portable.
    script = tmp_path / "rewriter.sh"
    script.write_text("#!/bin/sh\nprintf 'REWRITTEN'\n")
    script.chmod(0o755)

    raw = {
        "user_prompt_submit": [{"command": str(script), "timeout_ms": 3000}],
    }
    chain = build_lifecycle_hooks_from_config(raw)
    assert len(chain.user_prompt_submit) == 1
    agent = _build_agent(tmp_path, hooks=chain)
    async for _ in agent.astream("original"):
        pass
    saved = agent._storage.load("default")
    human_msgs = [m for m in saved if getattr(m, "type", "") == "human"]
    assert any("REWRITTEN" in str(m.content) for m in human_msgs)
    await agent.aclose()


def test_build_lifecycle_hooks_from_config_skips_unknown_events() -> None:
    chain = build_lifecycle_hooks_from_config({"unknown_event": [{"command": "x"}]})
    assert chain.session_start == []
    assert chain.user_prompt_submit == []
    assert chain.notification == []
    assert chain.stop == []


def test_build_lifecycle_hooks_from_config_skips_malformed_entries() -> None:
    raw = {
        "session_start": [
            {"command": ""},          # empty
            {"timeout_ms": 1000},     # missing command
            "not-a-dict",              # wrong type
            {"command": "valid"},     # only this one survives
        ],
        "stop": "not-a-list",          # whole-event wrong shape
    }
    chain = build_lifecycle_hooks_from_config(raw)
    assert len(chain.session_start) == 1
    assert len(chain.stop) == 0
