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
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aura.application.compact.compact import _is_prompt_too_long
from aura.application.compact.constants import MICROCOMPACT_CLEAR_MARKER
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.skills.types import Skill
from aura.schemas.todos import TodoItem
from tests.conftest import FakeChatModel, FakeTurn


def _minimal_config(
    enabled: list[str] | None = None,
    *,
    context_window: int | None = None,
) -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": enabled if enabled is not None else []},
        **({"context_window": context_window} if context_window is not None else {}),
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
    agent.storage.save(agent.session_id, h)


class SizeLimitedSummaryModel(FakeChatModel):
    """Fake summary model that rejects prompts over a provider-sized limit."""

    def __init__(self, *, max_prompt_chars: int) -> None:
        super().__init__(turns=[])
        self.__dict__["max_prompt_chars"] = max_prompt_chars
        self.__dict__["prompt_sizes"] = []
        self.__dict__["prompts"] = []

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["ainvoke_calls"] += 1
        prompt_text = "\n".join(str(m.content) for m in messages)
        self.__dict__["prompts"].append(prompt_text)
        prompt_size = len(prompt_text)
        self.__dict__["prompt_sizes"].append(prompt_size)
        if prompt_size > self.__dict__["max_prompt_chars"]:
            raise RuntimeError(
                "Error code: 400 - {'error': {'code': '1261', "
                "'message': 'Prompt exceeds max length'}}"
            )
        content = f"summary-call-{self.__dict__['ainvoke_calls']}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def prompt_sizes(self) -> list[int]:
        return self.__dict__["prompt_sizes"]  # type: ignore[no-any-return]

    @property
    def prompts(self) -> list[str]:
        return self.__dict__["prompts"]  # type: ignore[no-any-return]


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
    history = agent.storage.load(agent.session_id)
    assert len(history) == 4
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_replaces_middle_with_summary_preserves_tail(
    tmp_path: Path,
) -> None:
    # 10 pairs = 20 messages; last 3 turns = 6 messages preserved raw.
    agent = _make_agent(tmp_path, summary_text="SUMMARY-BODY")
    _seed_history(agent, pairs=10)

    await agent.compact(source="manual")

    history = agent.storage.load(agent.session_id)
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
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_keeps_last_n_turns_raw(tmp_path: Path) -> None:
    # Specifically assert KEEP_LAST_N_TURNS * 2 messages land raw at the tail.
    from aura.application.compact.constants import KEEP_LAST_N_TURNS

    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=8)  # 16 messages total

    await agent.compact(source="manual")

    history = agent.storage.load(agent.session_id)
    tail = history[-KEEP_LAST_N_TURNS * 2 :]
    assert len(tail) == KEEP_LAST_N_TURNS * 2
    # These are the last N raw turns from the original history — not summaries.
    assert all(
        ("<session-summary>" not in str(m.content)) for m in tail
    )
    await agent.aclose()


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
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_resets_progressive_state_preserves_reads(
    tmp_path: Path,
) -> None:
    """Phase 3 Task 3 — post-compact Context comes from ``fresh()``.

    Progressive fields (``_invoked_skills``, ``_loaded_nested_paths``)
    must be EMPTY on the new Context: invoked-skill bodies are surfaced
    via ``<skill-active>`` re-injection HumanMessages in history (see
    ``test_compact_skill_reinjection.py``), so keeping them on the new
    Context's ``_invoked_skills`` would double-render. ``_read_records``
    is preserved by ``fresh()`` defaults — the file is still on disk so
    the must-read-first fingerprint remains valid.
    """
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

    # Seed a read record + a synthetic loaded-nested-path so we can prove the
    # new Context preserves reads while emptying nested-load discovery state.
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    agent._context.record_read(target)
    agent._context._loaded_nested_paths.add((tmp_path / "AURA.md").resolve())

    await agent.compact(source="manual")

    # After compact: NEW Context with cleared progressive fields.
    assert agent._context._invoked_skills == []
    assert agent._context._invoked_skill_paths == set()
    assert agent._context._loaded_nested_paths == set()
    # Read fingerprints survive — must-read-first invariant.
    assert agent._context.read_status(target) == "fresh"
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_preserves_todos(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    agent.state.slots.todos.clear()
    agent.state.slots.todos.append(
        TodoItem(
            content="TASK-A", status="pending", active_form="Doing TASK-A",
        )
    )

    await agent.compact(source="manual")

    todos = agent.state.slots.todos
    assert len(todos) == 1
    assert todos[0].content == "TASK-A"
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_clears_nested_memory_fragments(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    # Inject a synthetic nested fragment; compact must clear this — nested
    # memory is a DISCOVERY artefact and will be re-discovered on next
    # tool-touched-path.
    from aura.application.memory.context import NestedFragment

    agent._context._nested_fragments.append(
        NestedFragment(source=tmp_path / "AURA.md", content="STALE")
    )
    agent._context._loaded_nested_paths.add((tmp_path / "AURA.md").resolve())

    await agent.compact(source="manual")

    assert agent._context._nested_fragments == []
    assert agent._context._loaded_nested_paths == set()
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_clears_matched_rules(tmp_path: Path) -> None:
    from aura.application.memory.rules import Rule

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
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_reruns_must_read_first_hook_with_new_context(
    tmp_path: Path,
) -> None:
    # Regression: the must-read-first hook closure must be swapped to the NEW
    # Context AND the preserved read record must survive. If either half
    # regresses, the invariant breaks silently.
    from pydantic import BaseModel

    from aura.schemas.permissions import Allow
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
    # Preserved record = fresh → hook passes through as Allow (no block).
    assert isinstance(outcome, Allow)
    await agent.aclose()


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
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_result_dataclass_shape(tmp_path: Path) -> None:
    """Agent.compact returns CompactResult with before/after/source fields."""
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=10)

    agent.state.total_tokens_used = 123

    result = await agent.compact(source="manual")

    # CompactResult dataclass contract.
    assert result.source == "manual"
    # before_tokens is total_tokens_used captured at entry.
    assert result.before_tokens == 123
    # after_tokens exists and is an integer (same or increased; summary turn
    # may add usage if a usage hook were wired — here it isn't, so equal).
    assert isinstance(result.after_tokens, int)
    await agent.aclose()


def test_compact_detects_dashscope_prompt_max_length_error() -> None:
    exc = RuntimeError(
        "Error code: 400 - {'error': {'code': '1261', "
        "'message': 'Prompt exceeds max length'}}"
    )

    assert _is_prompt_too_long(exc) is True


@pytest.mark.asyncio
async def test_compact_splits_summary_when_provider_rejects_large_prompt(
    tmp_path: Path,
) -> None:
    model = SizeLimitedSummaryModel(max_prompt_chars=9_000)
    agent = Agent(
        config=_minimal_config(context_window=15_000),
        model=model,
        storage=_storage(tmp_path),
    )
    history: list[Any] = []
    for i in range(16):
        history.append(HumanMessage(content=f"user-{i} " + ("u" * 500)))
        history.append(AIMessage(content=f"assistant-{i} " + ("a" * 500)))
    agent.storage.save(agent.session_id, history)

    await agent.compact(source="manual")

    compacted = agent.storage.load(agent.session_id)
    assert "<session-summary>" in str(compacted[0].content)
    assert model.ainvoke_calls > 1
    assert max(model.prompt_sizes) <= model.__dict__["max_prompt_chars"]
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_truncates_oversized_raw_tool_outputs_before_summary(
    tmp_path: Path,
) -> None:
    model = SizeLimitedSummaryModel(max_prompt_chars=9_000)
    agent = Agent(
        config=_minimal_config(context_window=15_000),
        model=model,
        storage=_storage(tmp_path),
    )
    history: list[Any] = []
    for i in range(8):
        history.append(HumanMessage(content=f"user-{i}"))
        history.append(AIMessage(content="assistant"))
        history.append(HumanMessage(content="TOOL-OUTPUT-" + ("x" * 50_000)))
    agent.storage.save(agent.session_id, history)

    await agent.compact(source="manual")

    assert model.prompt_sizes
    assert max(model.prompt_sizes) <= model.__dict__["max_prompt_chars"]
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_summarizes_microcompacted_dynamic_history_view(
    tmp_path: Path,
) -> None:
    model = SizeLimitedSummaryModel(max_prompt_chars=12_000)
    agent = Agent(
        config=_minimal_config(context_window=20_000),
        model=model,
        storage=_storage(tmp_path),
        microcompact_trigger_pairs=2,
        microcompact_keep_recent=1,
    )
    history: list[Any] = []
    for i in range(6):
        call_id = f"tc-{i}"
        history.append(HumanMessage(content=f"user-{i}"))
        history.append(AIMessage(
            content="",
            tool_calls=[{
                "name": "read_file",
                "args": {"path": f"file-{i}.py"},
                "id": call_id,
            }],
        ))
        history.append(ToolMessage(
            content="RAW-OLD-TOOL-RESULT-" + ("x" * 20_000),
            tool_call_id=call_id,
            name="read_file",
        ))
    agent.storage.save(agent.session_id, history)

    await agent.compact(source="manual")

    compacted = agent.storage.load(agent.session_id)
    assert "<session-summary>" in str(compacted[0].content)
    sent_summary_prompt = "\n".join(model.prompts)
    assert MICROCOMPACT_CLEAR_MARKER in sent_summary_prompt
    assert "RAW-OLD-TOOL-RESULT-" not in sent_summary_prompt
    await agent.aclose()


def _touch_with_mtime(path: Path, body: str, mtime: float) -> None:
    """Write ``body`` to ``path`` then force its mtime — ordering the reads."""
    path.write_text(body)
    import os
    os.utime(path, (mtime, mtime))


@pytest.mark.asyncio
async def test_compact_reinjects_top_n_recent_files_by_mtime(
    tmp_path: Path,
) -> None:
    """Top ``compact.max_files_to_restore`` reads (by mtime DESC) re-injected after compact."""
    agent = _make_agent(tmp_path)
    max_files_to_restore = agent.config.compact.max_files_to_restore
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
    history = agent.storage.load(agent.session_id)
    recent_file_messages = [
        m for m in history
        if isinstance(m, HumanMessage) and "<recent-file" in str(m.content)
    ]
    assert len(recent_file_messages) == max_files_to_restore

    # Top 5 by mtime DESC = files 6, 5, 4, 3, 2.
    joined = "\n".join(str(m.content) for m in recent_file_messages)
    for i in (6, 5, 4, 3, 2):
        assert f"BODY-{i}" in joined, f"expected BODY-{i} in re-injected blob"
    # Older files skipped.
    for i in (1, 0):
        assert f"BODY-{i}" not in joined
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_honors_max_files_to_restore_config_override(
    tmp_path: Path,
) -> None:
    """Phase 4 Task 2 — JSON config override on ``compact.max_files_to_restore``
    propagates end-to-end through ``run_compact`` instead of using the legacy
    constant default (5). Pinning ``2`` and recording 5 reads must yield
    exactly 2 ``<recent-file>`` HumanMessages — proves the constant→config
    migration is wired all the way to the call site.
    """
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "compact": {"max_files_to_restore": 2},
    })
    model = FakeChatModel(turns=[FakeTurn(AIMessage(content="SUMMARY-TEXT"))])
    agent = Agent(config=cfg, model=model, storage=_storage(tmp_path))
    _seed_history(agent, pairs=10)

    # 5 files, distinct mtimes — newest first (file_4) should win every slot.
    for i in range(5):
        p = tmp_path / f"file_{i}.txt"
        _touch_with_mtime(p, f"BODY-{i}", mtime=1_000_000 + i)
        agent._context.record_read(p)

    await agent.compact(source="manual")

    history = agent.storage.load(agent.session_id)
    recent_file_messages = [
        m for m in history
        if isinstance(m, HumanMessage) and "<recent-file" in str(m.content)
    ]
    # Override pinned 2 — exactly 2 files re-injected, not the legacy default 5.
    assert len(recent_file_messages) == 2
    joined = "\n".join(str(m.content) for m in recent_file_messages)
    # Top 2 by mtime DESC = file_4, file_3.
    assert "BODY-4" in joined
    assert "BODY-3" in joined
    # Older entries (would have been re-injected at the legacy default of 5)
    # are dropped under the override.
    for i in (2, 1, 0):
        assert f"BODY-{i}" not in joined
    await agent.aclose()


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

    history = agent.storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "FULL-BODY" in blob
    assert "PART-BODY" not in blob
    await agent.aclose()


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

    history = agent.storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert "ALIVE-BODY" in blob
    assert "DEAD-BODY" not in blob
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_caps_file_body_at_max_tokens_per_file(
    tmp_path: Path,
) -> None:
    """Oversize file bodies are truncated with a ``(truncated)`` marker."""
    agent = _make_agent(tmp_path)
    max_tokens_per_file = agent.config.compact.max_tokens_per_file
    _seed_history(agent, pairs=10)

    # 4 chars/token approx → cap is ``max_tokens_per_file * 4`` chars.
    max_chars = max_tokens_per_file * 4
    huge_body = "X" * (max_chars + 500)
    big = tmp_path / "big.txt"
    _touch_with_mtime(big, huge_body, mtime=1_000_000)
    agent._context.record_read(big)

    await agent.compact(source="manual")

    history = agent.storage.load(agent.session_id)
    recent = [
        str(m.content) for m in history
        if isinstance(m, HumanMessage) and "<recent-file" in str(m.content)
    ]
    assert recent, "expected a <recent-file> block"
    blob = recent[0]
    assert "(truncated)" in blob
    # Total payload roughly bounded — tag + truncated body stays close to max_chars.
    assert len(blob) <= max_chars + 1_000
    await agent.aclose()


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

    history = agent.storage.load(agent.session_id)
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
    await agent.aclose()
