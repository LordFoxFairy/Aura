"""Unit tests for the microcompact layer (v0.12 G2).

Pure-function coverage of the per-turn prompt-view transform that replaces
old tool_use/tool_result pair payloads with a clear marker. Scope here is
``aura.core.compact.microcompact`` only — integration into ``_invoke_model``
and the Agent constructor surface are exercised by other test files.

Design anchors (from tasks/todo.md, session 2026-04-24):
- View transform, not history mutation (pure functions, new lists).
- No LLM call: literal string replacement with marker.
- Pair-count trigger + keep-recent-N policy (default 8 / 5).
- Hard floor: always keep >= 1 pair, matching claude-code
  ``Math.max(1, config.keepRecent)``.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from aura.core.compact.constants import (
    MICROCOMPACT_CLEAR_MARKER,
    MICROCOMPACT_COMPACTABLE_TOOLS,
    MICROCOMPACT_KEEP_RECENT,
    MICROCOMPACT_TRIGGER_PAIRS,
)
from aura.core.compact.microcompact import (
    MicrocompactPolicy,
    MicrocompactResult,
    ToolPair,
    apply_clear,
    apply_microcompact,
    find_tool_pairs,
    select_clear_ids,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _ai_with_call(
    tool_name: str, tool_call_id: str, args: dict[str, object] | None = None,
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": tool_name, "args": args or {}, "id": tool_call_id},
        ],
    )


def _tool_msg(
    tool_call_id: str,
    *,
    name: str,
    content: str = "result",
    artifact: object = None,
    status: Literal["success", "error"] = "success",
) -> ToolMessage:
    msg = ToolMessage(content=content, tool_call_id=tool_call_id)
    msg.name = name
    msg.status = status
    if artifact is not None:
        msg.artifact = artifact
    return msg


def _n_pairs(n: int, *, tool_name: str = "read_file") -> list[BaseMessage]:
    """Build ``n`` alternating Human/AI(tool_call)/Tool triples in order."""
    messages: list[BaseMessage] = []
    for i in range(n):
        messages.append(HumanMessage(content=f"user-{i}"))
        messages.append(_ai_with_call(tool_name, f"tc-{i}"))
        messages.append(_tool_msg(f"tc-{i}", name=tool_name, content=f"result-{i}"))
    return messages


# ---------------------------------------------------------------------------
# find_tool_pairs
# ---------------------------------------------------------------------------


def test_find_tool_pairs_empty_messages_returns_empty() -> None:
    assert find_tool_pairs([], MICROCOMPACT_COMPACTABLE_TOOLS) == []


def test_find_tool_pairs_no_tool_calls_returns_empty() -> None:
    messages: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
        HumanMessage(content="ok"),
    ]
    assert find_tool_pairs(messages, MICROCOMPACT_COMPACTABLE_TOOLS) == []


def test_find_tool_pairs_single_matched_pair() -> None:
    messages: list[BaseMessage] = [
        HumanMessage(content="read foo"),
        _ai_with_call("read_file", "tc-1"),
        _tool_msg("tc-1", name="read_file", content="contents"),
    ]
    pairs = find_tool_pairs(messages, MICROCOMPACT_COMPACTABLE_TOOLS)
    assert pairs == [ToolPair(ai_idx=1, tool_idx=2, tool_call_id="tc-1", tool_name="read_file")]


def test_find_tool_pairs_non_compactable_tool_excluded() -> None:
    # task_create is intentionally NOT in the compactable allowlist (high-signal).
    messages: list[BaseMessage] = [
        _ai_with_call("task_create", "tc-1"),
        _tool_msg("tc-1", name="task_create", content="spawned"),
    ]
    assert find_tool_pairs(messages, MICROCOMPACT_COMPACTABLE_TOOLS) == []


def test_find_tool_pairs_unmatched_call_skipped_not_errored() -> None:
    # AIMessage has a tool_call but no corresponding ToolMessage follows —
    # should be silently skipped (defensive: partial turns during streaming).
    messages: list[BaseMessage] = [
        _ai_with_call("read_file", "tc-orphan"),
        AIMessage(content="gave up"),
    ]
    assert find_tool_pairs(messages, MICROCOMPACT_COMPACTABLE_TOOLS) == []


def test_find_tool_pairs_multi_turn_encounter_order() -> None:
    messages = _n_pairs(3, tool_name="read_file")
    pairs = find_tool_pairs(messages, MICROCOMPACT_COMPACTABLE_TOOLS)
    assert [p.tool_call_id for p in pairs] == ["tc-0", "tc-1", "tc-2"]
    assert all(p.tool_name == "read_file" for p in pairs)


def test_find_tool_pairs_collision_first_match_wins() -> None:
    # Two ToolMessages with the same tool_call_id (pathological but defensively
    # handled). The first match should be bound to the first ToolMessage.
    messages: list[BaseMessage] = [
        _ai_with_call("read_file", "tc-dup"),
        _tool_msg("tc-dup", name="read_file", content="first"),
        _tool_msg("tc-dup", name="read_file", content="second"),
    ]
    pairs = find_tool_pairs(messages, MICROCOMPACT_COMPACTABLE_TOOLS)
    assert len(pairs) == 1
    assert pairs[0].tool_idx == 1  # first ToolMessage wins


# ---------------------------------------------------------------------------
# select_clear_ids
# ---------------------------------------------------------------------------


def _synthetic_pairs(n: int) -> list[ToolPair]:
    return [
        ToolPair(ai_idx=2 * i, tool_idx=2 * i + 1, tool_call_id=f"tc-{i}", tool_name="read_file")
        for i in range(n)
    ]


def test_select_clear_ids_zero_pairs_returns_empty() -> None:
    assert select_clear_ids([], trigger_pairs=8, keep_recent=5) == set()


def test_select_clear_ids_under_trigger_returns_empty() -> None:
    assert select_clear_ids(_synthetic_pairs(5), trigger_pairs=8, keep_recent=5) == set()


def test_select_clear_ids_just_over_trigger_clears_oldest_excess() -> None:
    # 9 pairs, keep_recent=5 → clear the oldest 4 (tc-0..tc-3), keep tc-4..tc-8.
    ids = select_clear_ids(_synthetic_pairs(9), trigger_pairs=8, keep_recent=5)
    assert ids == {"tc-0", "tc-1", "tc-2", "tc-3"}


def test_select_clear_ids_many_over_trigger_clears_oldest() -> None:
    # 20 pairs, keep_recent=5 → clear oldest 15 (tc-0..tc-14).
    ids = select_clear_ids(_synthetic_pairs(20), trigger_pairs=8, keep_recent=5)
    assert ids == {f"tc-{i}" for i in range(15)}


def test_select_clear_ids_keep_recent_zero_floor_to_one() -> None:
    # keep_recent=0 would clear everything; claude-code enforces max(1, keep_recent).
    # 10 pairs, trigger=3, keep_recent=0 → keep last 1 (tc-9), clear tc-0..tc-8.
    ids = select_clear_ids(_synthetic_pairs(10), trigger_pairs=3, keep_recent=0)
    assert ids == {f"tc-{i}" for i in range(9)}


def test_select_clear_ids_trigger_one_keep_one() -> None:
    # 2 pairs, trigger=1, keep_recent=1 → clear 1 (tc-0).
    ids = select_clear_ids(_synthetic_pairs(2), trigger_pairs=1, keep_recent=1)
    assert ids == {"tc-0"}


# ---------------------------------------------------------------------------
# apply_clear
# ---------------------------------------------------------------------------


def test_apply_clear_empty_ids_returns_references() -> None:
    messages: list[BaseMessage] = [
        HumanMessage(content="hi"),
        _ai_with_call("read_file", "tc-1"),
        _tool_msg("tc-1", name="read_file"),
    ]
    out = apply_clear(messages, set(), MICROCOMPACT_CLEAR_MARKER)
    assert out == messages
    # Reference-equal preservation for untouched path.
    for a, b in zip(out, messages, strict=True):
        assert a is b


def test_apply_clear_single_id_replaces_content_preserves_metadata() -> None:
    tm = _tool_msg(
        "tc-1",
        name="read_file",
        content="LONG RESULT" * 100,
        status="success",
    )
    messages: list[BaseMessage] = [
        _ai_with_call("read_file", "tc-1"),
        tm,
    ]
    out = apply_clear(messages, {"tc-1"}, MICROCOMPACT_CLEAR_MARKER)
    replaced = out[1]
    assert isinstance(replaced, ToolMessage)
    assert replaced.content == MICROCOMPACT_CLEAR_MARKER
    assert replaced.tool_call_id == "tc-1"
    assert replaced.name == "read_file"
    assert replaced.status == "success"
    # Original message reference must NOT be mutated.
    assert tm.content != MICROCOMPACT_CLEAR_MARKER


def test_apply_clear_preserves_artifact_field() -> None:
    artifact = {"path": "/tmp/x", "bytes": 1024}
    messages: list[BaseMessage] = [
        _ai_with_call("read_file", "tc-1"),
        _tool_msg("tc-1", name="read_file", content="...", artifact=artifact),
    ]
    out = apply_clear(messages, {"tc-1"}, MICROCOMPACT_CLEAR_MARKER)
    replaced = out[1]
    assert isinstance(replaced, ToolMessage)
    assert replaced.artifact == artifact


def test_apply_clear_leaves_aimessage_tool_calls_untouched() -> None:
    # Even if the AIMessage's tool_call id is in clear_ids, the call itself
    # must remain visible to the model (it must see the call was issued).
    ai = _ai_with_call("read_file", "tc-1")
    messages: list[BaseMessage] = [
        ai,
        _tool_msg("tc-1", name="read_file"),
    ]
    out = apply_clear(messages, {"tc-1"}, MICROCOMPACT_CLEAR_MARKER)
    # AIMessage untouched, reference-preserved.
    assert out[0] is ai
    assert isinstance(out[0], AIMessage)
    assert out[0].tool_calls[0]["id"] == "tc-1"


def test_apply_clear_only_touches_matching_tool_messages() -> None:
    t0 = _tool_msg("tc-0", name="read_file", content="keep-0")
    t1 = _tool_msg("tc-1", name="read_file", content="clear-1")
    t2 = _tool_msg("tc-2", name="read_file", content="keep-2")
    messages: list[BaseMessage] = [
        _ai_with_call("read_file", "tc-0"),
        t0,
        _ai_with_call("read_file", "tc-1"),
        t1,
        _ai_with_call("read_file", "tc-2"),
        t2,
    ]
    out = apply_clear(messages, {"tc-1"}, MICROCOMPACT_CLEAR_MARKER)
    assert out[1] is t0  # untouched → reference-equal
    assert out[5] is t2  # untouched → reference-equal
    # Only tc-1 got replaced (new object, not the original reference).
    assert out[3] is not t1
    assert isinstance(out[3], ToolMessage)
    assert out[3].content == MICROCOMPACT_CLEAR_MARKER
    # Other two keep original content.
    assert t0.content == "keep-0"
    assert t2.content == "keep-2"


# ---------------------------------------------------------------------------
# apply_microcompact (facade)
# ---------------------------------------------------------------------------


def test_apply_microcompact_trigger_zero_is_noop() -> None:
    messages = _n_pairs(10, tool_name="read_file")
    policy = MicrocompactPolicy(trigger_pairs=0)
    result = apply_microcompact(messages, policy)
    assert isinstance(result, MicrocompactResult)
    assert result.cleared_pair_count == 0
    assert result.cleared_tool_call_ids == ()
    # Pass-through: every message reference preserved.
    for a, b in zip(result.messages, messages, strict=True):
        assert a is b


def test_apply_microcompact_ten_pairs_clears_five_oldest() -> None:
    messages = _n_pairs(10, tool_name="read_file")
    policy = MicrocompactPolicy(trigger_pairs=8, keep_recent=5)
    result = apply_microcompact(messages, policy)
    assert result.cleared_pair_count == 5
    # Oldest-first ordering: tc-0..tc-4 are the cleared ids.
    assert result.cleared_tool_call_ids == ("tc-0", "tc-1", "tc-2", "tc-3", "tc-4")
    # Verify the actual ToolMessages for tc-0..tc-4 got their content swapped
    # and tc-5..tc-9 kept their original content.
    for i in range(5):
        tm = result.messages[3 * i + 2]
        assert isinstance(tm, ToolMessage)
        assert tm.content == MICROCOMPACT_CLEAR_MARKER
    for i in range(5, 10):
        tm = result.messages[3 * i + 2]
        assert isinstance(tm, ToolMessage)
        assert tm.content == f"result-{i}"


def test_apply_microcompact_defaults_match_constants() -> None:
    # Sanity: MicrocompactPolicy() should match the module-level defaults.
    policy = MicrocompactPolicy()
    assert policy.trigger_pairs == MICROCOMPACT_TRIGGER_PAIRS
    assert policy.keep_recent == MICROCOMPACT_KEEP_RECENT
    assert policy.clear_marker == MICROCOMPACT_CLEAR_MARKER
    assert policy.compactable_tools == MICROCOMPACT_COMPACTABLE_TOOLS
