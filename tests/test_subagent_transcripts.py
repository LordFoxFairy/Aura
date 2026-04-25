"""F-07-012 / V14-Persistence-V3: persistent on-disk subagent transcripts.

Each subagent's full message history is persisted as JSONL so a failed
child's diagnostic trail survives the parent process exit. Under the
v3 layout, transcripts live nested under the parent session
(``<projects>/<encoded-cwd>/<parent-session>/subagents/agent-<task_id>.jsonl``)
when ``parent_session_id`` is supplied, with a flat fallback at
``<storage_root>/subagents/agent-<task_id>.jsonl`` for callers that
haven't yet threaded the parent attribution through.

This file pins:

- The transcript file is on disk after ``run_task`` returns.
- ``list_subagent_transcripts`` enumerates the file regardless of which
  layout (flat or nested) the writer chose.
- One JSONL line per message; the full transcript round-trips through
  ``load_subagent_transcript``.
- Subagent transcripts are ISOLATED from the user-session resume picker
  (``list_sessions``) — their files never appear in the session index.
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


def _candidate_transcript_paths(
    storage: SessionStorage, task_id: str,
) -> list[Path]:
    """Return every path Track A's storage could have written under.

    Track B (run.py) currently writes a file directly at the legacy
    flat path AND calls back into ``storage.write_subagent_transcript``
    (which writes at Track A's canonical flat fallback path). Either
    file existing satisfies the "transcript persisted" invariant.
    """
    return [
        # Track A fallback when no parent_session_id is wired through.
        storage.subagent_transcript_path(task_id),
        # Run.py legacy filename (still present on disk).
        storage._path.parent / "subagents" / f"subagent-{task_id}.jsonl",
    ]


@pytest.mark.asyncio
async def test_subagent_writes_transcript_to_disk(tmp_path: Path) -> None:
    parent_storage = SessionStorage(tmp_path / "parent.db", cwd=tmp_path)
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d", prompt="hello world")

    await run_task(
        store, factory, rec.id, transcript_storage=parent_storage,
    )

    candidates = _candidate_transcript_paths(parent_storage, rec.id)
    written = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    assert written, (
        f"expected a transcript at one of {candidates}; none exist"
    )


@pytest.mark.asyncio
async def test_transcript_path_captured_in_task_record(tmp_path: Path) -> None:
    parent_storage = SessionStorage(tmp_path / "parent.db", cwd=tmp_path)
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d", prompt="hi")
    await run_task(
        store, factory, rec.id, transcript_storage=parent_storage,
    )

    rec_after = store.get(rec.id)
    assert rec_after is not None
    assert rec_after.transcript_path is not None
    # The captured path embeds the task id and lives under a
    # ``subagents/`` directory regardless of which writer pinned it.
    assert rec.id in rec_after.transcript_path.name
    assert rec_after.transcript_path.parent.name == "subagents"
    assert rec_after.transcript_path.exists()


@pytest.mark.asyncio
async def test_transcript_persists_after_subagent_completes(
    tmp_path: Path,
) -> None:
    """Transcript file outlives both subagent and parent process simulation."""
    parent_storage = SessionStorage(tmp_path / "parent.db", cwd=tmp_path)
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d", prompt="hi")
    await run_task(
        store, factory, rec.id, transcript_storage=parent_storage,
    )
    candidates = _candidate_transcript_paths(parent_storage, rec.id)

    # Simulate process exit: drop in-memory references.
    parent_storage.close()
    del store
    del factory

    surviving = [p for p in candidates if p.exists()]
    assert surviving, f"no transcript survived among {candidates}"
    for p in surviving:
        text = p.read_text(encoding="utf-8")
        assert text.strip(), f"transcript file {p} is empty"


@pytest.mark.asyncio
async def test_list_subagent_transcripts_returns_recent(tmp_path: Path) -> None:
    parent_storage = SessionStorage(tmp_path / "parent.db", cwd=tmp_path)
    store = TasksStore()
    factory = _make_factory()

    rec_a = store.create(description="a", prompt="p1")
    rec_b = store.create(description="b", prompt="p2")
    await run_task(store, factory, rec_a.id, transcript_storage=parent_storage)
    await run_task(store, factory, rec_b.id, transcript_storage=parent_storage)

    metas = parent_storage.list_subagent_transcripts()
    # Dedup by task_id — even if Track B and Track A both wrote files,
    # the listing surfaces one row per task.
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
    parent_storage = SessionStorage(tmp_path / "parent.db", cwd=tmp_path)
    parent_storage.save("user-session-1", [HumanMessage(content="hi parent")])

    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d", prompt="hi")
    await run_task(
        store, factory, rec.id, transcript_storage=parent_storage,
    )

    sessions = parent_storage.list_sessions()
    session_ids = {s.session_id for s in sessions}
    assert "user-session-1" in session_ids
    assert f"subagent-{rec.id}" not in session_ids
    assert f"agent-{rec.id}" not in session_ids
    for sid in session_ids:
        assert not sid.startswith("subagent-")
        assert not sid.startswith("agent-")

    transcripts = parent_storage.list_subagent_transcripts()
    assert any(m.task_id == rec.id for m in transcripts)


@pytest.mark.asyncio
async def test_transcript_includes_full_message_stream(tmp_path: Path) -> None:
    """The persisted JSONL contains every message the child saw — one per line."""
    parent_storage = SessionStorage(tmp_path / "parent.db", cwd=tmp_path)
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d", prompt="hello universe")
    await run_task(
        store, factory, rec.id, transcript_storage=parent_storage,
    )

    candidates = _candidate_transcript_paths(parent_storage, rec.id)
    path = next((p for p in candidates if p.exists()), None)
    assert path is not None, f"no transcript at any of {candidates}"
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
    human_messages = [
        m for m in loaded if isinstance(m, HumanMessage)
    ]
    assert any(
        isinstance(m.content, str) and "hello universe" in m.content
        for m in human_messages
    )
