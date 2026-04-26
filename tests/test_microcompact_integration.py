"""Integration tests for the microcompact layer (v0.12 G2 — T4 + T5 + T6).

Covers the wiring between the pure-function ``aura.core.compact.microcompact``
surface and the turn loop / Agent constructor:

- ``Agent(...)`` rejects misconfigurations at construction (keep_recent >=
  trigger_pairs) with ``AuraConfigError``.
- ``auto_microcompact_enabled=False`` disables the feature entirely
  (no markers appear in the outgoing prompt).
- ``microcompact_trigger_pairs=0`` also disables the feature.
- With defaults (8/5), a session carrying 10 compactable tool pairs
  emits a ``microcompact_applied`` journal event with
  ``cleared_pair_count=5`` and the 5 oldest tool_call_ids.
- Stored history in SQLite is NOT mutated by the view transform —
  loading the session back shows all 10 original tool_result payloads.
- AIMessage.tool_calls entries survive the transform unchanged
  (provider-level schema requires the call to remain visible).

Test design: pre-populate storage with N compactable pairs, then drive a
single ``astream`` turn with a recording FakeChatModel that (a) snapshots
the outgoing messages on ``_agenerate`` and (b) returns a tool-call-free
AIMessage so the turn terminates in one model round. This keeps the test
deterministic and fast while still exercising the full ``_invoke_model``
path the feature lives on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core.agent import Agent
from aura.core.compact import MICROCOMPACT_CLEAR_MARKER
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


class _RecordingFakeChatModel(FakeChatModel):
    """FakeChatModel that snapshots the outgoing ``messages`` on each call.

    Needed because the stock ``FakeChatModel`` records only bound-tools and
    ainvoke counts — not the message list — and the microcompact contract
    is observable *only* on the outgoing prompt.
    """

    def __init__(self, turns: list[FakeTurn]) -> None:
        super().__init__(turns=turns)
        self.__dict__["captured_messages"] = []

    @property
    def captured_messages(self) -> list[list[BaseMessage]]:
        return self.__dict__["captured_messages"]  # type: ignore[no-any-return]

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Snapshot before delegating — copying the list keeps the view
        # stable even if the loop mutates history afterwards.
        self.__dict__["captured_messages"].append(list(messages))
        self.__dict__["ainvoke_calls"] += 1
        turn = self._pop_turn()
        return ChatResult(generations=[ChatGeneration(message=turn.message)])


def _build_n_compactable_pairs(n: int) -> list[BaseMessage]:
    """Pre-populated history: ``n`` ``read_file`` tool_use/tool_result pairs.

    Each pair is wrapped by a HumanMessage so message-order stays realistic.
    The tool_call_ids follow ``tc-0``..``tc-{n-1}`` so assertions can talk
    about the oldest k ids trivially.
    """
    history: list[BaseMessage] = []
    for i in range(n):
        history.append(HumanMessage(content=f"read pair {i}"))
        history.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": f"/f{i}"}, "id": f"tc-{i}"},
                ],
            )
        )
        history.append(
            ToolMessage(
                content=f"payload-for-pair-{i}",
                tool_call_id=f"tc-{i}",
                name="read_file",
                status="success",
            )
        )
    return history


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db")


def _seed_history(storage: SessionStorage, session_id: str, n_pairs: int) -> None:
    """Persist ``n_pairs`` compactable pairs under ``session_id`` before astream.

    ``astream`` loads history via ``storage.load(session_id)`` before each
    turn, so seeding the DB is the cleanest way to ensure the next
    ``_invoke_model`` sees a fully-populated history.
    """
    storage.save(session_id, _build_n_compactable_pairs(n_pairs))


def _count_markers(messages: Sequence[BaseMessage]) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m, ToolMessage) and m.content == MICROCOMPACT_CLEAR_MARKER
    )


# ---------------------------------------------------------------------------
# T5 — config surface (misconfig)
# ---------------------------------------------------------------------------


def test_misconfig_keep_gte_trigger_raises_at_construction(tmp_path: Path) -> None:
    # keep_recent == trigger_pairs → impossible to ever clear anything.
    with pytest.raises(AuraConfigError) as excinfo:
        Agent(
            config=_minimal_config(),
            model=FakeChatModel(turns=[FakeTurn(AIMessage(content="x"))]),
            storage=_storage(tmp_path),
            microcompact_trigger_pairs=5,
            microcompact_keep_recent=5,
        )
    assert "microcompact_keep_recent" in str(excinfo.value)

    # keep_recent > trigger_pairs → same failure mode.
    with pytest.raises(AuraConfigError):
        Agent(
            config=_minimal_config(),
            model=FakeChatModel(turns=[FakeTurn(AIMessage(content="x"))]),
            storage=_storage(tmp_path / "db2"),
            microcompact_trigger_pairs=5,
            microcompact_keep_recent=7,
        )


# ---------------------------------------------------------------------------
# T5 — disable paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_microcompact_enabled_false_disables_feature(
    tmp_path: Path,
) -> None:
    model = _RecordingFakeChatModel(
        turns=[FakeTurn(AIMessage(content="done"))],
    )
    storage = _storage(tmp_path)
    session_id = "session-disabled-flag"
    _seed_history(storage, session_id, n_pairs=20)

    agent = Agent(
        config=_minimal_config(),
        model=model,
        storage=storage,
        session_id=session_id,
        auto_microcompact_enabled=False,
    )
    try:
        async for _ in agent.astream("next"):
            pass
    finally:
        await agent.aclose()

    assert len(model.captured_messages) == 1
    sent = model.captured_messages[0]
    # No marker anywhere — the 20 pre-seeded ToolMessage payloads flow
    # through untouched.
    assert _count_markers(sent) == 0
    # Sanity: all 20 original payloads made it to the prompt verbatim.
    payloads = [
        m.content for m in sent
        if isinstance(m, ToolMessage) and m.name == "read_file"
    ]
    assert len(payloads) == 20
    assert payloads[0] == "payload-for-pair-0"
    assert payloads[19] == "payload-for-pair-19"


@pytest.mark.asyncio
async def test_trigger_pairs_zero_disables_feature(tmp_path: Path) -> None:
    model = _RecordingFakeChatModel(
        turns=[FakeTurn(AIMessage(content="done"))],
    )
    storage = _storage(tmp_path)
    session_id = "session-disabled-zero"
    _seed_history(storage, session_id, n_pairs=20)

    agent = Agent(
        config=_minimal_config(),
        model=model,
        storage=storage,
        session_id=session_id,
        # keep_recent MUST be < trigger_pairs per validator, so set both to 0
        # to exercise the "trigger_pairs <= 0 disables" path without tripping
        # the misconfig guard.
        microcompact_trigger_pairs=0,
        microcompact_keep_recent=0,
    )
    try:
        async for _ in agent.astream("next"):
            pass
    finally:
        await agent.aclose()

    assert len(model.captured_messages) == 1
    sent = model.captured_messages[0]
    assert _count_markers(sent) == 0
    payloads = [
        m.content for m in sent
        if isinstance(m, ToolMessage) and m.name == "read_file"
    ]
    assert len(payloads) == 20


# ---------------------------------------------------------------------------
# T4 — journal event + storage invariant + AIMessage preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_microcompact_applied_journal_event_emitted(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)
    try:
        model = _RecordingFakeChatModel(
            turns=[FakeTurn(AIMessage(content="done"))],
        )
        storage = _storage(tmp_path)
        session_id = "journal-session"
        _seed_history(storage, session_id, n_pairs=10)

        agent = Agent(
            config=_minimal_config(),
            model=model,
            storage=storage,
            session_id=session_id,
            # Defaults: trigger=5, keep=3 → 10 pairs clears the oldest 7.
        )
        try:
            async for _ in agent.astream("next"):
                pass
        finally:
            await agent.aclose()

        events = [
            json.loads(line)
            for line in log_path.read_text().strip().split("\n")
            if line
        ]
        applied = [e for e in events if e["event"] == "microcompact_applied"]
        assert len(applied) == 1
        ev = applied[0]
        assert ev["session"] == session_id
        assert ev["cleared_pair_count"] == 7
        # Oldest-first ordering: pre-seeded history used ``tc-0``..``tc-9``,
        # so the oldest 7 are ``tc-0``..``tc-6``.
        assert ev["cleared_tool_call_ids"] == [f"tc-{i}" for i in range(7)]
        # ``cleared_positions`` — the ToolPair indices into the outgoing
        # ``messages`` view (what find_tool_pairs sees). Context.build
        # prepends two messages now: the system_prompt SystemMessage and
        # the ``<skills-available>`` HumanMessage from bundled skills
        # (F-0910-011, v0.18.x). Raw triples (Human @ 3i, AI @ 3i+1,
        # Tool @ 3i+2) become (3i+2, 3i+3, 3i+4) in the outgoing view.
        assert ev["cleared_positions"] == [
            [3 * i + 3, 3 * i + 4] for i in range(7)
        ]
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_stored_history_unchanged_by_microcompact(tmp_path: Path) -> None:
    model = _RecordingFakeChatModel(
        turns=[FakeTurn(AIMessage(content="done"))],
    )
    storage = _storage(tmp_path)
    session_id = "storage-invariant-session"
    _seed_history(storage, session_id, n_pairs=10)

    agent = Agent(
        config=_minimal_config(),
        model=model,
        storage=storage,
        session_id=session_id,
    )
    try:
        async for _ in agent.astream("next"):
            pass
    finally:
        await agent.aclose()

    # Reload storage — the astream flow re-saved history at end-of-turn.
    # Even with microcompact clearing 5 old payloads in the OUTGOING prompt,
    # the stored ToolMessages must retain their full original content.
    storage2 = SessionStorage(tmp_path / "aura.db")
    try:
        reloaded = storage2.load(session_id)
    finally:
        storage2.close()

    tool_messages = [m for m in reloaded if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 10
    for i, tm in enumerate(tool_messages):
        # NOT the marker — the original payload survived.
        assert tm.content == f"payload-for-pair-{i}", (
            f"stored ToolMessage {i} content was mutated "
            f"(got {tm.content!r}, expected payload-for-pair-{i})"
        )
        assert tm.tool_call_id == f"tc-{i}"


@pytest.mark.asyncio
async def test_ai_message_tool_calls_preserved_in_outgoing_prompt(
    tmp_path: Path,
) -> None:
    model = _RecordingFakeChatModel(
        turns=[FakeTurn(AIMessage(content="done"))],
    )
    storage = _storage(tmp_path)
    session_id = "ai-preservation-session"
    _seed_history(storage, session_id, n_pairs=10)

    agent = Agent(
        config=_minimal_config(),
        model=model,
        storage=storage,
        session_id=session_id,
    )
    try:
        async for _ in agent.astream("next"):
            pass
    finally:
        await agent.aclose()

    assert len(model.captured_messages) == 1
    sent = model.captured_messages[0]

    # All 10 AIMessage.tool_calls survive untouched — provider schema
    # requires the call to remain visible alongside its (now-cleared) result.
    ai_tool_call_ids: list[str] = []
    for m in sent:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                tc_id = tc.get("id")
                if tc_id is not None:
                    ai_tool_call_ids.append(tc_id)
    # Ten pre-seeded pairs + one tool-call-free reply this turn (tool_calls
    # is empty, contributes nothing) == 10 ids in encounter order.
    assert ai_tool_call_ids == [f"tc-{i}" for i in range(10)]

    # 7 markers (oldest) + 3 original payloads (recent) — defaults: trigger=5, keep=3.
    assert _count_markers(sent) == 7
    payloads = [
        m.content for m in sent
        if isinstance(m, ToolMessage) and m.name == "read_file"
    ]
    assert len(payloads) == 10
    assert payloads[:7] == [MICROCOMPACT_CLEAR_MARKER] * 7
    assert payloads[7:] == [f"payload-for-pair-{i}" for i in range(7, 10)]
