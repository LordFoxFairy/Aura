"""Tests for the CwdChanged hook + Agent.set_cwd + cwd-rules reload consumer.

Covers (V14-HOOK-CATALOG):

1. ``Agent.set_cwd(new_path)`` updates ``self._cwd``.
2. ``set_cwd`` fires ``CwdChanged`` with old + new paths.
3. ``set_cwd`` emits ``cwd_changed`` journal event.
4. Reload consumer refreshes Context's rules from the new cwd.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aura.core.hooks.auto_reload import make_cwd_rules_reload_hook
from aura.schemas.state import LoopState


def _minimal_agent(tmp_path: Path) -> Any:
    """Build a bare Agent with no tools enabled — enough for set_cwd tests."""
    from aura.config.schema import AuraConfig
    from aura.core.agent import Agent
    from aura.core.persistence.storage import SessionStorage
    from tests.conftest import FakeChatModel

    cfg = AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
            "storage": {"path": str(tmp_path / "db")},
        }
    )
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "aura.db"),
    )


@pytest.mark.asyncio
async def test_set_cwd_updates_internal_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_home = tmp_path / "_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    agent = _minimal_agent(tmp_path)
    try:
        new_dir = tmp_path / "sub"
        new_dir.mkdir()

        await agent.set_cwd(new_dir)
        assert agent._cwd == new_dir.resolve()
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_set_cwd_fires_cwd_changed_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_home = tmp_path / "_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    agent = _minimal_agent(tmp_path)
    try:
        new_dir = tmp_path / "sub"
        new_dir.mkdir()

        captured: list[tuple[Path, Path]] = []

        async def hook(
            *, old_cwd: Path, new_cwd: Path, state: LoopState, **_: Any,
        ) -> None:
            captured.append((old_cwd, new_cwd))

        agent._hooks.cwd_changed.append(hook)

        old = agent._cwd
        await agent.set_cwd(new_dir)

        assert captured == [(old, new_dir.resolve())]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_set_cwd_emits_journal_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_home = tmp_path / "_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    log_path = tmp_path / "events.jsonl"
    from aura.core.persistence import journal
    journal.configure(log_path)

    agent = _minimal_agent(tmp_path)
    try:
        new_dir = tmp_path / "sub"
        new_dir.mkdir()

        await agent.set_cwd(new_dir)
    finally:
        await agent.aclose()
        journal.reset()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    cwd_events = [e for e in events if e.get("event") == "cwd_changed"]
    assert len(cwd_events) == 1
    e = cwd_events[0]
    # ``Path.resolve`` on tmp_path may differ from the literal string on
    # macOS (``/private/var`` vs ``/var``) — assert with resolved form.
    assert e["new_cwd"] == str(new_dir.resolve())


@pytest.mark.asyncio
async def test_cwd_rules_reload_refreshes_rules_from_new_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_home = tmp_path / "_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    from aura.core.memory import project_memory, rules
    project_memory.clear_cache()
    rules.clear_cache()

    agent = _minimal_agent(tmp_path)
    try:
        # Brand-new cwd with its own rules + AURA.md.
        new_cwd = tmp_path / "project2"
        rules_dir = new_cwd / ".aura" / "rules"
        rules_dir.mkdir(parents=True)
        rule_path = rules_dir / "r.md"
        rule_path.write_text(
            "---\npaths: \"**/*.py\"\n---\nRELOADED-BODY\n",
            encoding="utf-8",
        )
        (new_cwd / "AURA.md").write_text("RELOADED-MEM", encoding="utf-8")

        # Sanity: original Agent does NOT see project2's rules yet.
        original_sources = [r.source_path for r in agent._rules.conditional]
        assert rule_path.resolve() not in original_sources

        # Move + invoke the reload consumer.
        await agent.set_cwd(new_cwd)
        consumer = make_cwd_rules_reload_hook(agent)
        await consumer(
            old_cwd=tmp_path.resolve(),
            new_cwd=new_cwd.resolve(),
            state=agent.state,
        )

        # Now the Agent's rules + memory reflect the new cwd.
        new_sources = [r.source_path for r in agent._rules.conditional]
        assert rule_path.resolve() in new_sources
        assert "RELOADED-MEM" in agent._primary_memory
    finally:
        await agent.aclose()
