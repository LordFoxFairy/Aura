"""Subagent session_id isolation (audit Tier S).

Before this fix, :meth:`SubagentFactory.spawn` hardcoded
``session_id="subagent"`` on every child :class:`Agent`. Two concurrent
subagents under the same parent both wrote into
:class:`SessionStorage` keyed by the same ``"subagent"`` literal, and
``storage.save`` is a ``DELETE WHERE session_id=? then INSERT`` — one
subagent's transcript clobbered the other's the instant it flushed.

Journal events (``turn_begin``, ``tool_execute_end``,
``microcompact_applied``, ``astream_started``, ``storage_save``, ...)
carried ``session="subagent"`` regardless of which child emitted them,
so forensics couldn't tell the two apart.

The contract exercised here:

1. Two subagents spawned in parallel from the same parent each get a
   distinct ``session_id`` of the form ``subagent-<task_id>``.
2. Each child's storage save/load round-trips its own messages — no
   cross-contamination, even when the storage is the same
   :class:`SessionStorage` instance (shared across spawns).
3. Journal events emitted by each child carry its own ``session=``
   value, matching ``^subagent-``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from aura.config.schema import AuraConfig
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().strip().split("\n")
        if line
    ]


@pytest.fixture(autouse=True)
def _reset_journal() -> Any:
    journal.reset()
    yield
    journal.reset()


@pytest.mark.asyncio
async def test_two_subagents_get_distinct_session_ids(tmp_path: Path) -> None:
    """Spawn two subagents via ``factory.spawn(..., task_id=...)`` and verify
    each child's ``_session_id`` is ``subagent-<their-task-id>``."""
    factory = SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="done"))]
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    child_a = factory.spawn("prompt-a", task_id="task-aaaa")
    child_b = factory.spawn("prompt-b", task_id="task-bbbb")
    try:
        assert child_a._session_id == "subagent-task-aaaa"
        assert child_b._session_id == "subagent-task-bbbb"
        assert child_a._session_id != child_b._session_id
    finally:
        await child_a.aclose()
        await child_b.aclose()


@pytest.mark.asyncio
async def test_subagents_storage_does_not_cross_contaminate(tmp_path: Path) -> None:
    """Two subagents sharing one :class:`SessionStorage` keep separate rows.

    Under the ``session_id="subagent"`` bug, the second child's ``save``
    would ``DELETE WHERE session_id='subagent'`` — wiping the first
    child's turn. With per-task ids, both rows coexist.
    """
    db = tmp_path / "sessions.sqlite"
    shared_storage = SessionStorage(db)

    factory = SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="child-output"))]
        ),
        storage_factory=lambda: shared_storage,
    )

    child_a = factory.spawn("prompt-a", task_id="task-aaaa")
    child_b = factory.spawn("prompt-b", task_id="task-bbbb")

    try:
        # Run one scripted turn on each child. ``astream`` triggers
        # storage.save at the end of the turn.
        async for _ in child_a.astream("prompt-a"):
            pass
        async for _ in child_b.astream("prompt-b"):
            pass

        msgs_a = shared_storage.load("subagent-task-aaaa")
        msgs_b = shared_storage.load("subagent-task-bbbb")

        # Each child's session has its own messages. Neither is empty
        # (the DELETE-clobber bug showed up as an empty load for the
        # first-saved session).
        assert msgs_a, "child-a session wiped by child-b (session_id collision)"
        assert msgs_b, "child-b session wiped by child-a (session_id collision)"

        # The per-session prompts differ — confirm they didn't swap.
        texts_a = [m.content for m in msgs_a]
        texts_b = [m.content for m in msgs_b]
        assert any("prompt-a" in str(t) for t in texts_a)
        assert any("prompt-b" in str(t) for t in texts_b)
        # And the legacy literal "subagent" session has no rows
        # (confirms nobody wrote to the old hardcoded key).
        assert shared_storage.load("subagent") == []
    finally:
        await child_a.aclose()
        await child_b.aclose()
        shared_storage.close()


@pytest.mark.asyncio
async def test_subagent_journal_events_carry_distinct_session_ids(
    tmp_path: Path,
) -> None:
    """Each subagent's journal events tag ``session=subagent-<task_id>``."""
    journal_path = tmp_path / "journal.jsonl"
    journal.configure(journal_path)

    store = TasksStore()
    factory = SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="done"))]
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )

    rec_a = store.create(description="a", prompt="alpha-prompt")
    rec_b = store.create(description="b", prompt="beta-prompt")

    # Parallel run, same parent event loop.
    await asyncio.gather(
        run_task(store, factory, rec_a.id),
        run_task(store, factory, rec_b.id),
    )

    events = _events(journal_path)
    # Collect every ``session`` value we see.
    session_values = {e.get("session") for e in events if "session" in e}
    session_values.discard(None)

    expected_a = f"subagent-{rec_a.id}"
    expected_b = f"subagent-{rec_b.id}"
    assert expected_a in session_values, (
        f"expected subagent-a's session in journal; got {session_values}"
    )
    assert expected_b in session_values, (
        f"expected subagent-b's session in journal; got {session_values}"
    )
    # No event should carry the legacy literal "subagent".
    assert "subagent" not in session_values, (
        "journal still shows hardcoded 'subagent' session — per-task ids "
        "did not reach the child Agent"
    )


@pytest.mark.asyncio
async def test_spawn_without_task_id_falls_back_to_unique_id(tmp_path: Path) -> None:
    """Legacy callers that don't pass ``task_id`` still get a unique session
    (not the hardcoded ``"subagent"``). Two such spawns must differ."""
    factory = SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="done"))]
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    child_a = factory.spawn("p")
    child_b = factory.spawn("p")
    try:
        assert child_a._session_id != child_b._session_id
        assert child_a._session_id.startswith("subagent-")
        assert child_b._session_id.startswith("subagent-")
        # Neither degrades to the bare literal.
        assert child_a._session_id != "subagent"
        assert child_b._session_id != "subagent"
    finally:
        await child_a.aclose()
        await child_b.aclose()
