"""``AgentSession.compact``: history summarization plus selective state preservation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aura.application.compact.constants import MICROCOMPACT_CLEAR_MARKER
from aura.application.compact.microcompact import MicrocompactPolicy
from aura.application.compact.reactive import _is_prompt_too_long
from aura.application.compact.summary_turn import (
    SummaryCaps,
    _fallback_summary,
    _run_summary_turn_resilient,
    _run_summary_turn_with_retry,
    _serialize_tool_args,
    compact_summary_messages,
    compact_summary_prompt_budget,
    estimate_compact_summary_tokens,
)
from aura.application.session import AgentSession
from aura.config.schema import AuraConfig
from aura.domain.skill import Skill
from aura.domain.todos import TodoItem
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage
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


def _make_agent(tmp_path: Path, *, summary_text: str = "SUMMARY-TEXT") -> AgentSession:
    """AgentSession whose FakeChatModel yields a single scripted summary turn."""
    model = FakeChatModel(turns=[FakeTurn(AIMessage(content=summary_text))])
    return AgentSession(
        config=_minimal_config(),
        model=model,
        storage=_storage(tmp_path),
    )


def _seed_history(agent: AgentSession, *, pairs: int) -> None:
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
        result: list[int] = self.__dict__["prompt_sizes"]
        return result

    @property
    def prompts(self) -> list[str]:
        result: list[str] = self.__dict__["prompts"]
        return result


@pytest.mark.asyncio
async def test_compact_noop_when_short_history(tmp_path: Path) -> None:
    # KEEP_LAST_N_TURNS=3 → need >= 6 messages before compaction does anything.
    # 4 messages (2 pairs) must be a no-op.
    agent = _make_agent(tmp_path)
    _seed_history(agent, pairs=2)
    assert isinstance(agent._model, FakeChatModel)
    before_model_calls = agent._model.ainvoke_calls

    result = await agent.compact(source="manual")

    # No summary turn should have been invoked.
    assert isinstance(agent._model, FakeChatModel)
    assert agent._model.ainvoke_calls == before_model_calls
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

    # New Context after compact, but must-read-first fingerprint survives.
    assert agent._context.read_status(target) == "fresh"
    await agent.aclose()


@pytest.mark.asyncio
async def test_compact_resets_progressive_state_preserves_reads(
    tmp_path: Path,
) -> None:
    """Post-compact Context preserves reads but resets progressive caches."""
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
    from aura.application.memory.rules_types import Rule

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

    from aura.application.loop_state import LoopState
    from aura.domain.permission.outcome import Allow
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
    """AgentSession.compact returns CompactResult with before/after/source fields."""
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
    agent = AgentSession(
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
    agent = AgentSession(
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
    agent = AgentSession(
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
    """``compact.max_files_to_restore`` overrides propagate end-to-end."""
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "compact": {"max_files_to_restore": 2},
    })
    model = FakeChatModel(turns=[FakeTurn(AIMessage(content="SUMMARY-TEXT"))])
    agent = AgentSession(config=cfg, model=model, storage=_storage(tmp_path))
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


# --- summary_turn.py unit-level boundary coverage -------------------------


class _PromptTooLongModel(FakeChatModel):
    """Summary model rejecting any prompt whose serialized chars exceed a cap.

    Mirrors a provider that hard-caps prompt size: the resilient summarizer must
    bisect the message list until each piece fits (or fall back deterministically).
    """

    def __init__(self, *, max_prompt_chars: int) -> None:
        super().__init__(turns=[])
        self.__dict__["max_prompt_chars"] = max_prompt_chars
        self.__dict__["accepted_sizes"] = []

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["ainvoke_calls"] += 1
        size = sum(len(str(m.content)) for m in messages)
        if size > self.__dict__["max_prompt_chars"]:
            raise RuntimeError("context length exceeded: prompt is too long")
        self.__dict__["accepted_sizes"].append(size)
        idx = self.__dict__["ainvoke_calls"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"S{idx}"))])

    @property
    def accepted_sizes(self) -> list[int]:
        result: list[int] = self.__dict__["accepted_sizes"]
        return result


class _AlwaysTooLongModel(FakeChatModel):
    """Summary model that rejects every prompt as too long, regardless of size.

    Exercises the floor of the resilient summarizer: when even a single message
    cannot be summarized, it must yield the deterministic fallback, never crash.
    """

    def __init__(self) -> None:
        super().__init__(turns=[])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["ainvoke_calls"] += 1
        raise RuntimeError("maximum context length is 1 token")


class _NonPtlErrorModel(FakeChatModel):
    """Summary model whose failure is NOT a prompt-size rejection.

    Non-PTL errors are real faults: they must propagate, never be silently
    swallowed by the bisect-and-fallback path.
    """

    def __init__(self) -> None:
        super().__init__(turns=[])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["ainvoke_calls"] += 1
        raise RuntimeError("upstream 503 service unavailable")


def test_serialize_tool_args_falls_back_to_str_on_non_json_value() -> None:
    """Unserializable tool args must not crash serialization — degrade to ``str``."""
    weird: dict[str, Any] = {"obj": object()}
    rendered = _serialize_tool_args(weird, caps=SummaryCaps())
    # json.dumps(default=str) actually stringifies object() — force the except
    # branch with a key that json cannot encode at all.
    circular: dict[str, Any] = {}
    circular["self"] = circular
    rendered2 = _serialize_tool_args(circular, caps=SummaryCaps())
    assert "obj" in rendered
    assert "self" in rendered2


def test_serialize_tool_args_caps_oversized_payload() -> None:
    """Tool-arg blobs over the cap are truncated so one giant call can't bloat the prompt."""
    caps = SummaryCaps(max_summary_tool_args_chars=20)
    rendered = _serialize_tool_args({"k": "v" * 1_000}, caps=caps)
    assert "truncated" in rendered
    assert len(rendered) < 1_000


def test_estimate_compact_summary_tokens_is_positive_for_content() -> None:
    """Token estimate must stay strictly positive so budget math never divides by zero."""
    tokens = estimate_compact_summary_tokens([HumanMessage(content="hello world")])
    assert tokens > 0
    # Empty history still includes the fixed prompt scaffolding → non-zero.
    assert estimate_compact_summary_tokens([]) > 0


def test_compact_summary_messages_returns_copy_when_no_policy(tmp_path: Path) -> None:
    """No microcompact policy → summary sees a verbatim COPY, never the live list."""
    model = FakeChatModel(turns=[FakeTurn(AIMessage(content="SUMMARY-TEXT"))])
    agent = AgentSession(
        config=_minimal_config(),
        model=model,
        storage=_storage(tmp_path),
        microcompact_trigger_pairs=0,  # disable microcompact → policy is None
    )
    assert agent.microcompact_policy is None
    history: list[BaseMessage] = [HumanMessage(content="a"), AIMessage(content="b")]
    view = compact_summary_messages(agent, history)
    assert view == history
    assert view is not history  # defensive copy — mutating the view can't corrupt storage


def test_compact_summary_messages_applies_policy_when_present(tmp_path: Path) -> None:
    """A microcompact policy must reshape the summary view (clear-marker injected)."""
    agent = _make_agent(tmp_path)
    # trigger_pairs=1 with 3 read_file pairs (>1) and keep_recent=1 → oldest
    # two tool payloads get cleared in the summary view.
    policy = MicrocompactPolicy(trigger_pairs=1, keep_recent=1)
    object.__setattr__(agent._loop, "_microcompact_policy", policy)
    history: list[BaseMessage] = []
    for i in range(3):
        call_id = f"tc-{i}"
        history.append(HumanMessage(content=f"u{i}"))
        history.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": f"f{i}.py"}, "id": call_id}
                ],
            )
        )
        history.append(ToolMessage(content=f"RAW-{i}", tool_call_id=call_id, name="read_file"))
    view = compact_summary_messages(agent, history)
    blob = "\n".join(str(m.content) for m in view)
    assert MICROCOMPACT_CLEAR_MARKER in blob
    assert "RAW-0" not in blob  # oldest payload cleared


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (type("PromptTooLongError", (RuntimeError,), {})("boom"), True),
        (type("PromptTooLong", (RuntimeError,), {})("boom"), True),
        (RuntimeError("Error: {'code': '1261'} something else"), True),
        (RuntimeError('Error: {"code": "1261"} other'), True),
        (RuntimeError("totally unrelated network blip"), False),
        (RuntimeError("code 9999 unknown"), False),
    ],
)
def test_is_prompt_too_long_matrix(exc: BaseException, expected: bool) -> None:
    """Provider PTL detection must catch type-name and bare error-code signals."""
    assert _is_prompt_too_long(exc) is expected


def test_compact_summary_prompt_budget_floor_on_nonpositive_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero/negative context window must not shrink the budget below the safe max."""
    agent = _make_agent(tmp_path)
    monkeypatch.setattr(AgentSession, "context_window", property(lambda _self: 0))
    assert compact_summary_prompt_budget(agent) == 16_000


def test_compact_summary_prompt_budget_clamps_small_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny positive window must clamp the prompt budget to the 2_000-token floor."""
    agent = _make_agent(tmp_path)
    monkeypatch.setattr(AgentSession, "context_window", property(lambda _self: 13_500))
    # window - 13_000 = 500 < 2_000 floor → floor wins.
    assert compact_summary_prompt_budget(agent) == 2_000


@pytest.mark.asyncio
async def test_run_summary_turn_with_retry_empty_history_returns_blank(
    tmp_path: Path,
) -> None:
    """Empty history must summarize to an empty string, not invoke the model."""
    model = FakeChatModel(turns=[])
    result = await _run_summary_turn_with_retry(model, [])
    assert result == ""
    assert model.ainvoke_calls == 0


@pytest.mark.asyncio
async def test_run_summary_turn_resilient_empty_returns_blank() -> None:
    """Recursive base case: no messages → empty summary, zero model calls."""
    model = FakeChatModel(turns=[])
    result = await _run_summary_turn_resilient(model, [], depth=0)
    assert result == ""
    assert model.ainvoke_calls == 0


@pytest.mark.asyncio
async def test_run_summary_turn_resilient_bisects_on_prompt_too_long() -> None:
    """Oversized prompt must bisect-and-merge into a single combined summary."""
    # Overhead alone is ~718 chars; cap 1100 admits single messages (~1030)
    # but rejects pairs (~1330) → forces bisection to single-message leaves.
    model = _PromptTooLongModel(max_prompt_chars=1_100)
    messages: list[BaseMessage] = [HumanMessage(content="m" * 300) for _ in range(4)]

    summary = await _run_summary_turn_resilient(model, messages, depth=0)

    # The full list was rejected, leaves were accepted, then merged → non-empty.
    assert summary != ""
    assert model.ainvoke_calls > 1
    # The model accepted at least one prompt, all within its cap.
    assert model.accepted_sizes
    assert all(size <= 1_100 for size in model.accepted_sizes)


@pytest.mark.asyncio
async def test_run_summary_turn_resilient_single_message_falls_back() -> None:
    """A lone message the provider always rejects must yield the deterministic fallback."""
    model = _AlwaysTooLongModel()
    lone: list[BaseMessage] = [HumanMessage(content="x" * 5_000)]

    summary = await _run_summary_turn_resilient(model, lone, depth=0)

    assert "<goal>" in summary
    assert "provider rejected the compact prompt size" in summary


@pytest.mark.asyncio
async def test_run_summary_turn_resilient_depth_cap_forces_fallback() -> None:
    """Hitting the split-depth ceiling must stop recursion and fall back, not loop."""
    model = _AlwaysTooLongModel()
    caps = SummaryCaps(max_summary_split_depth=1)
    messages: list[BaseMessage] = [HumanMessage(content="a"), HumanMessage(content="b")]

    summary = await _run_summary_turn_resilient(model, messages, depth=5, caps=caps)

    # depth (5) >= cap (1) with len>1 → immediate fallback, no further bisecting.
    assert "<history-excerpt>" in summary


@pytest.mark.asyncio
async def test_run_summary_turn_resilient_propagates_non_ptl_error() -> None:
    """Non-prompt-size failures must propagate — never be masked as a fallback summary."""
    model = _NonPtlErrorModel()
    messages: list[BaseMessage] = [HumanMessage(content="x")]

    with pytest.raises(RuntimeError, match="503 service unavailable"):
        await _run_summary_turn_resilient(model, messages, depth=0)


@pytest.mark.asyncio
async def test_run_summary_turn_with_retry_chains_partial_summaries() -> None:
    """Budget-split history must summarize each chunk then fold partials into one."""
    model = _PromptTooLongModel(max_prompt_chars=100_000)
    # ~600 chars/message (~150 tokens). Budget 400 (> ~180 prompt overhead) admits
    # one message per chunk but splits two → several chunks whose tiny partial
    # summaries then fold back into a single converging prompt.
    messages: list[BaseMessage] = [HumanMessage(content="z" * 600) for _ in range(6)]

    summary = await _run_summary_turn_with_retry(
        model, messages, max_prompt_tokens=400
    )

    # Multiple chunks → multiple model calls, then the partials collapse to one.
    assert summary != ""
    assert model.ainvoke_calls > 1


def test_fallback_summary_truncates_oversized_excerpt() -> None:
    """The deterministic fallback must cap its embedded excerpt to the char limit."""
    caps = SummaryCaps(fallback_summary_char_limit=100, max_summary_message_chars=10_000)
    messages: list[BaseMessage] = [HumanMessage(content="y" * 5_000)]

    out = _fallback_summary(messages, caps=caps)

    assert "(truncated)" in out
    assert "<history-excerpt>" in out


def test_fallback_summary_keeps_small_excerpt_intact() -> None:
    """A small history excerpt survives verbatim inside the fallback envelope."""
    messages: list[BaseMessage] = [HumanMessage(content="tiny")]
    out = _fallback_summary(messages)
    assert "tiny" in out
    assert "(truncated)" not in out


@pytest.mark.asyncio
async def test_run_summary_turn_with_retry_single_chunk_direct_path() -> None:
    """A history that fits one chunk takes the single-chunk path (one resilient call)."""
    model = FakeChatModel(turns=[FakeTurn(AIMessage(content="ONE"))])
    messages: list[BaseMessage] = [HumanMessage(content="short")]

    summary = await _run_summary_turn_with_retry(
        model, messages, max_prompt_tokens=16_000
    )

    assert summary == "ONE"
    assert model.ainvoke_calls == 1
