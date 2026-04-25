"""F-07-012: persistent on-disk subagent transcripts.

Each subagent's full message history is written to JSONL under
``<storage_root>/subagents/subagent-<task_id>.jsonl`` so a failed
child's diagnostic trail survives the parent process exit. This file
exercises:

- The transcript path is captured on the TaskRecord.
- The file persists after the subagent terminates.
- One JSONL line per message; the full transcript round-trips.
- ``list_subagent_transcripts`` returns recent transcripts newest-first.
- The transcripts are ISOLATED from the user-session resume picker
  (``list_sessions``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.config.schema import AuraConfig
from aura.core.persistence.storage import SessionStorage, TranscriptMeta
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from tests.conftest import FakeChatModel, FakeTurn


def _make_factory() -> SubagentFactory:
    return SubagentFactory(
        parent_config=AuraConfig.model_validate({
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="child-final"))],
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )


@pytest.mark.asyncio
async def test_subagent_writes_transcript_to_disk(tmp_path: Path) -> None:
    parent_storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d", prompt="hello world")

    await run_task(
        store, factory, rec.id, transcript_storage=parent_storage,
    )

    expected = tmp_path / "subagents" / f"subagent-{rec.id}.jsonl"
    assert expected.exists()
    assert expected.stat().st_size > 0


@pytest.mark.asyncio
async def test_transcript_path_captured_in_task_record(tmp_path: Path) -> None:
    parent_storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d", prompt="hi")
    await run_task(
        store, factory, rec.id, transcript_storage=parent_storage,
    )

    rec_after = store.get(rec.id)
    assert rec_after is not None
    assert rec_after.transcript_path is not None
    assert rec_after.transcript_path.name == f"subagent-{rec.id}.jsonl"
    assert rec_after.transcript_path.exists()


@pytest.mark.asyncio
async def test_transcript_persists_after_subagent_completes(
    tmp_path: Path,
) -> None:
    """Transcript file outlives both subagent and parent process simulation."""
    parent_storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d", prompt="hi")
    await run_task(
        store, factory, rec.id, transcript_storage=parent_storage,
    )
    path = (
        tmp_path / "subagents" / f"subagent-{rec.id}.jsonl"
    )

    # Simulate process exit: drop in-memory references.
    parent_storage.close()
    del store
    del factory

    # The file must still be on disk and readable.
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.strip(), "transcript file is empty"


@pytest.mark.asyncio
async def test_list_subagent_transcripts_returns_recent(tmp_path: Path) -> None:
    parent_storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()
    factory = _make_factory()

    rec_a = store.create(description="a", prompt="p1")
    rec_b = store.create(description="b", prompt="p2")
    await run_task(store, factory, rec_a.id, transcript_storage=parent_storage)
    await run_task(store, factory, rec_b.id, transcript_storage=parent_storage)

    metas = parent_storage.list_subagent_transcripts()
    assert len(metas) == 2
    found_ids = {m.task_id for m in metas}
    assert rec_a.id in found_ids
    assert rec_b.id in found_ids
    for m in metas:
        assert isinstance(m, TranscriptMeta)
        assert m.path.exists()
        assert m.message_count > 0


@pytest.mark.asyncio
async def test_transcripts_isolated_from_user_session_listing(
    tmp_path: Path,
) -> None:
    """Subagent transcripts must NOT pollute list_sessions output."""
    parent_storage = SessionStorage(tmp_path / "parent.db")
    # Simulate one user session.
    parent_storage.save("user-session-1", [HumanMessage(content="hi parent")])

    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d", prompt="hi")
    await run_task(
        store, factory, rec.id, transcript_storage=parent_storage,
    )

    sessions = parent_storage.list_sessions()
    session_ids = {s.session_id for s in sessions}
    # The subagent transcript must NOT show up as a resumable user
    # session.
    assert "user-session-1" in session_ids
    assert f"subagent-{rec.id}" not in session_ids
    # And the listing path's structural source (sqlite messages table)
    # must not have any subagent-prefixed sessions.
    for sid in session_ids:
        assert not sid.startswith("subagent-")

    # Still listable via the dedicated transcript surface.
    transcripts = parent_storage.list_subagent_transcripts()
    assert any(m.task_id == rec.id for m in transcripts)


@pytest.mark.asyncio
async def test_transcript_includes_full_message_stream(tmp_path: Path) -> None:
    """The persisted JSONL contains every message the child saw — one per line."""
    parent_storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d", prompt="hello universe")
    await run_task(
        store, factory, rec.id, transcript_storage=parent_storage,
    )

    path = tmp_path / "subagents" / f"subagent-{rec.id}.jsonl"
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # At minimum: HumanMessage(prompt) + AIMessage(child-final).
    assert len(lines) >= 2
    payloads = [json.loads(line) for line in lines]
    types = [p.get("type") for p in payloads]
    assert "human" in types
    assert "ai" in types

    # Round-trip: load_subagent_transcript reproduces the same messages.
    loaded = parent_storage.load_subagent_transcript(rec.id)
    assert len(loaded) == len(lines)
    # The HumanMessage carries the original prompt.
    human_messages = [
        m for m in loaded if isinstance(m, HumanMessage)
    ]
    assert any(
        isinstance(m.content, str) and "hello universe" in m.content
        for m in human_messages
    )
