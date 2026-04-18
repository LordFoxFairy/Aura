"""Integration: verify events flow from multiple layers to the journal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.journal import reset_journal, setup_file_journal
from aura.core.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


@pytest.fixture(autouse=True)
def _reset() -> Any:
    yield
    reset_journal()


@pytest.mark.asyncio
async def test_astream_emits_layered_events(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    setup_file_journal(log)

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[FakeTurn(message=AIMessage(content="hi"))]),
        storage=SessionStorage(tmp_path / "db"),
    )

    async for _ in agent.astream("hello"):
        pass
    agent.close()

    events = [json.loads(line) for line in log.read_text().splitlines()]
    event_names = [e["event"] for e in events]
    assert "astream_begin" in event_names
    assert "turn_begin" in event_names
    assert "astream_end" in event_names
    assert "storage_load" in event_names
    assert "storage_save" in event_names
