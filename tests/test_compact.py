"""Tests for ``Agent.compact`` — conversation summarization + state preservation.

Covers:
  - noop on short history
  - summary replaces middle, preserves tail
  - KEEP_LAST_N_TURNS preserved raw
  - preserved state: read_records, invoked_skills, todos
  - cleared caches: nested memory fragments, matched rules
  - must-read-first hook swap over new Context
  - journal event emission

Model interactions are driven by FakeChatModel with a single scripted turn
for the summary response.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.skills.types import Skill
from aura.schemas.todos import TodoItem
from tests.conftest import FakeChatModel, FakeTurn


def _minimal_config(enabled: list[str] | None = None) -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": enabled if enabled is not None else []},
    })


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db")


def _make_agent(tmp_path: Path, *, summary_text: str = "SUMMARY-TEXT") -> Agent:
    """Agent whose FakeChatModel yields a single scripted summary turn."""
    model = FakeChatModel(turns=[FakeTurn(AIMessage(content=summary_text))])
    return Agent(
        config=_minimal_config(),
        model=model,
        storage=_storage(tmp_path),
    )


def _seed_history(agent: Agent, *, pairs: int) -> None:
    """Seed ``pairs`` HumanMessage/AIMessage pairs into the agent's storage."""
    h: list[Any] = []
    for i in range(pairs):
        h.append(HumanMessage(content=f"user-{i}"))
        h.append(AIMessage(content=f"assistant-{i}"))
    agent._storage.save(agent.session_id, h)


@pytest.mark.asyncio
async def test_compact_noop_when_short_history(tmp_path: Path) -> None:
    # KEEP_LAST_N_TURNS=3 → need >= 6 messages before compaction does anything.
    # 4 messages (2 pairs) must be a no-op.
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=2)
    before_model_calls = agent._model.ainvoke_calls  # type: ignore[attr-defined]

    result = await agent.compact(source="manual")

    # No summary turn should have been invoked.
    assert agent._model.ainvoke_calls == before_model_calls  # type: ignore[attr-defined]
    # Returned result still structurally valid.
    assert result.source == "manual"
    # History unchanged.
    history = agent._storage.load(agent.session_id)
    assert len(history) == 4
    agent.close()


@pytest.mark.asyncio
async def test_compact_replaces_middle_with_summary_preserves_tail(
    tmp_path: Path,
) -> None:
    # 10 pairs = 20 messages; last 3 turns = 6 messages preserved raw.
    agent = _make_agent(tmp_path, summary_text="SUMMARY-BODY")
    _seed_history(agent, pairs=10)

    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    # 1 summary HumanMessage + 6 preserved tail messages = 7.
    assert len(history) == 7
    assert isinstance(history[0], HumanMessage)
    content0 = str(history[0].content)
    assert "<session-summary>" in content0
    assert "SUMMARY-BODY" in content0
    assert "</session-summary>" in content0
    # Tail preserved in order — last 6 messages of original 20.
    assert str(history[1].content) == "user-7"
    assert str(history[-1].content) == "assistant-9"
    agent.close()


@pytest.mark.asyncio
async def test_compact_keeps_last_n_turns_raw(tmp_path: Path) -> None:
    # Specifically assert KEEP_LAST_N_TURNS * 2 messages land raw at the tail.
    from aura.core.compact.constants import KEEP_LAST_N_TURNS

    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=8)  # 16 messages total

    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    tail = history[-KEEP_LAST_N_TURNS * 2 :]
    assert len(tail) == KEEP_LAST_N_TURNS * 2
    # These are the last N raw turns from the original history — not summaries.
    assert all(
        ("<session-summary>" not in str(m.content)) for m in tail
    )
    agent.close()


@pytest.mark.asyncio
async def test_compact_preserves_read_records(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    agent._context.record_read(target)
    assert agent._context.read_status(target) == "fresh"

    await agent.compact(source="manual")

    # After compact: new Context is in place but the fresh read fingerprint
    # must survive — claude-code parity for the must-read-first invariant.
    assert agent._context.read_status(target) == "fresh"
    agent.close()


@pytest.mark.asyncio
async def test_compact_preserves_invoked_skills(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    skill = Skill(
        name="ping",
        description="ping desc",
        body="PING-BODY",
        source_path=tmp_path / "ping.md",
        layer="project",
    )
    agent.record_skill_invocation(skill)
    # Before compact: rendered Context shows the invoked skill.
    blob_before = " ".join(str(m.content) for m in agent._context.build([]))
    assert '<skill-invoked name="ping">' in blob_before
    assert "PING-BODY" in blob_before

    await agent.compact(source="manual")

    # After compact: new Context, but the invoked skill must still render.
    blob_after = " ".join(str(m.content) for m in agent._context.build([]))
    assert '<skill-invoked name="ping">' in blob_after
    assert "PING-BODY" in blob_after
    agent.close()


@pytest.mark.asyncio
async def test_compact_preserves_todos(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    agent._state.custom["todos"] = [
        TodoItem(
            content="TASK-A", status="pending", active_form="Doing TASK-A",
        )
    ]

    await agent.compact(source="manual")

    todos = agent._state.custom.get("todos", [])
    assert len(todos) == 1
    assert todos[0].content == "TASK-A"
    agent.close()


@pytest.mark.asyncio
async def test_compact_clears_nested_memory_fragments(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    # Inject a synthetic nested fragment; compact must clear this — nested
    # memory is a DISCOVERY artefact and will be re-discovered on next
    # tool-touched-path.
    from aura.core.memory.context import NestedFragment

    agent._context._nested_fragments.append(
        NestedFragment(source=tmp_path / "AURA.md", content="STALE")
    )
    agent._context._loaded_nested_paths.add((tmp_path / "AURA.md").resolve())

    await agent.compact(source="manual")

    assert agent._context._nested_fragments == []
    assert agent._context._loaded_nested_paths == set()
    agent.close()


@pytest.mark.asyncio
async def test_compact_clears_matched_rules(tmp_path: Path) -> None:
    from aura.core.memory.rules import Rule

    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    fake_rule = Rule(
        source_path=tmp_path / "rules" / "old.md",
        base_dir=tmp_path,
        globs=("**/*.py",),
        content="OLD-RULE",
    )
    agent._context._matched_rules.append(fake_rule)
    agent._context._matched_rule_paths.add(fake_rule.source_path)

    await agent.compact(source="manual")

    assert agent._context._matched_rules == []
    assert agent._context._matched_rule_paths == set()
    agent.close()


@pytest.mark.asyncio
async def test_compact_reruns_must_read_first_hook_with_new_context(
    tmp_path: Path,
) -> None:
    # Regression: the must-read-first hook closure must be swapped to the NEW
    # Context AND the preserved read record must survive. If either half
    # regresses, the invariant breaks silently.
    from pydantic import BaseModel

    from aura.schemas.state import LoopState
    from aura.tools.base import build_tool

    class _PathOldNew(BaseModel):
        path: str
        old_str: str
        new_str: str

    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    target = tmp_path / "f.txt"
    target.write_text("body\n")
    agent._context.record_read(target)
    assert agent._context.read_status(target) == "fresh"

    await agent.compact(source="manual")

    # The hook closure must see the preserved record on the NEW context.
    edit_tool = build_tool(
        name="edit_file",
        description="edit",
        args_schema=_PathOldNew,
        func=lambda path, old_str, new_str: {"replacements": 1},
        is_destructive=True,
    )
    outcome = await agent._must_read_first_hook(
        tool=edit_tool,
        args={"path": str(target), "old_str": "body", "new_str": "BODY"},
        state=LoopState(),
    )
    # Preserved record = fresh → no short_circuit (no block).
    assert outcome.short_circuit is None
    agent.close()


@pytest.mark.asyncio
async def test_compact_journal_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """compact() must write a ``compact_applied`` journal event with before/after."""
    captured: list[dict[str, Any]] = []

    def _capture(event: str, /, **fields: Any) -> None:
        if event == "compact_applied":
            captured.append(fields)

    monkeypatch.setattr(journal, "write", _capture)

    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    await agent.compact(source="manual")

    assert captured, "expected a compact_applied event"
    ev = captured[0]
    assert ev["source"] == "manual"
    # Integer-ish token counts present.
    assert "before_tokens" in ev
    assert "after_tokens" in ev
    agent.close()


@pytest.mark.asyncio
async def test_compact_result_dataclass_shape(tmp_path: Path) -> None:
    """Agent.compact returns CompactResult with before/after/source fields."""
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    agent._state.total_tokens_used = 123

    result = await agent.compact(source="manual")

    # CompactResult dataclass contract.
    assert result.source == "manual"
    # before_tokens is total_tokens_used captured at entry.
    assert result.before_tokens == 123
    # after_tokens exists and is an integer (same or increased; summary turn
    # may add usage if a usage hook were wired — here it isn't, so equal).
    assert isinstance(result.after_tokens, int)
    agent.close()


# ---------------------------------------------------------------------------
# Item 1 — selective file re-injection after compact
# ---------------------------------------------------------------------------


def _touch_with_mtime(path: Path, body: str, mtime: float) -> None:
    """Write ``body`` to ``path`` then force its mtime — ordering the reads."""
    path.write_text(body)
    import os
    os.utime(path, (mtime, mtime))


@pytest.mark.asyncio
async def test_compact_reinjects_top_n_recent_files_by_mtime(
    tmp_path: Path,
) -> None:
    """Top MAX_FILES_TO_RESTORE read files (by mtime DESC) re-injected after compact."""
    from aura.core.compact.constants import MAX_FILES_TO_RESTORE

    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    # Create 7 files, staggered mtimes — file_6 is newest, file_0 oldest.
    files: list[Path] = []
    for i in range(7):
        p = tmp_path / f"file_{i}.txt"
        _touch_with_mtime(p, f"BODY-{i}", mtime=1_000_000 + i)
        agent._context.record_read(p)
        files.append(p)

    await agent.compact(source="manual")

    # History: summary + N recent-file messages + preserved tail.
    history = agent._storage.load(agent.session_id)
    recent_file_messages = [
        m for m in history
        if isinstance(m, HumanMessage) and "<recent-file" in str(m.content)
    ]
    assert len(recent_file_messages) == MAX_FILES_TO_RESTORE

    # Top 5 by mtime DESC = files 6, 5, 4, 3, 2.
    joined = "\n".join(str(m.content) for m in recent_file_messages)
    for i in (6, 5, 4, 3, 2):
        assert f"BODY-{i}" in joined, f"expected BODY-{i} in re-injected blob"
    # Older files skipped.
    for i in (1, 0):
        assert f"BODY-{i}" not in joined
    agent.close()


@pytest.mark.asyncio
async def test_compact_skips_partial_reads_in_reinjection(tmp_path: Path) -> None:
    """Files whose recorded read was partial do NOT get re-injected."""
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    full = tmp_path / "full.txt"
    part = tmp_path / "part.txt"
    _touch_with_mtime(full, "FULL-BODY", mtime=2_000_000)
    _touch_with_mtime(part, "PART-BODY", mtime=3_000_000)  # newer, but partial
    agent._context.record_read(full)
    agent._context.record_read(part, partial=True)

    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "FULL-BODY" in blob
    assert "PART-BODY" not in blob
    agent.close()


@pytest.mark.asyncio
async def test_compact_handles_deleted_file_during_reinjection(
    tmp_path: Path,
) -> None:
    """A recorded file deleted before compact is silently skipped (no crash)."""
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    alive = tmp_path / "alive.txt"
    dead = tmp_path / "dead.txt"
    _touch_with_mtime(alive, "ALIVE-BODY", mtime=1_000_000)
    _touch_with_mtime(dead, "DEAD-BODY", mtime=2_000_000)  # newer
    agent._context.record_read(alive)
    agent._context.record_read(dead)
    dead.unlink()

    # Must not raise.
    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "ALIVE-BODY" in blob
    assert "DEAD-BODY" not in blob
    agent.close()


@pytest.mark.asyncio
async def test_compact_caps_file_body_at_max_tokens_per_file(
    tmp_path: Path,
) -> None:
    """Oversize file bodies are truncated with a ``(truncated)`` marker."""
    from aura.core.compact.constants import MAX_TOKENS_PER_FILE

    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    # 4 chars/token approx → cap is MAX_TOKENS_PER_FILE * 4 chars.
    max_chars = MAX_TOKENS_PER_FILE * 4
    huge_body = "X" * (max_chars + 500)
    big = tmp_path / "big.txt"
    _touch_with_mtime(big, huge_body, mtime=1_000_000)
    agent._context.record_read(big)

    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    recent = [
        str(m.content) for m in history
        if isinstance(m, HumanMessage) and "<recent-file" in str(m.content)
    ]
    assert recent, "expected a <recent-file> block"
    blob = recent[0]
    assert "(truncated)" in blob
    # Total payload roughly bounded — tag + truncated body stays close to max_chars.
    assert len(blob) <= max_chars + 1_000
    agent.close()


@pytest.mark.asyncio
async def test_compact_recent_files_rendered_before_preserved_tail(
    tmp_path: Path,
) -> None:
    """Order: summary, *<recent-file>, *preserved_tail."""
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    p = tmp_path / "r.txt"
    _touch_with_mtime(p, "RECENT-BODY", mtime=1_500_000)
    agent._context.record_read(p)

    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    # history[0] = summary
    assert "<session-summary>" in str(history[0].content)
    # history[1] = recent-file (only one recorded)
    assert "<recent-file" in str(history[1].content)
    assert "RECENT-BODY" in str(history[1].content)
    # history[2..] = preserved tail (user-7/assistant-7 onwards).
    tail = history[2:]
    # None of the tail messages should be a <recent-file> or <session-summary>.
    assert all(
        "<recent-file" not in str(m.content)
        and "<session-summary>" not in str(m.content)
        for m in tail
    )
    agent.close()
