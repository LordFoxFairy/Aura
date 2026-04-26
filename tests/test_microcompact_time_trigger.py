"""F-0910-004 — time-based microcompact trigger.

When the wall-clock gap since ``last_assistant_ts`` exceeds
``gap_threshold_minutes``, microcompact fires regardless of pair count
(force-clear surplus pairs above the keep-recent floor).
"""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from aura.core.compact.microcompact import MicrocompactPolicy, apply_microcompact


def _ai_call(name: str, tc_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": tc_id}])


def _tool(tc_id: str, *, name: str, content: str = "result") -> ToolMessage:
    msg = ToolMessage(content=content, tool_call_id=tc_id)
    msg.name = name
    msg.status = "success"
    return msg


def _three_pairs() -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for i in range(3):
        out.append(HumanMessage(content=f"u-{i}"))
        out.append(_ai_call("read_file", f"tc-{i}"))
        out.append(_tool(f"tc-{i}", name="read_file", content=f"r-{i}"))
    return out


def test_no_time_trigger_when_under_pair_threshold() -> None:
    # 3 pairs < trigger_pairs=5 → no clear without time trigger.
    policy = MicrocompactPolicy(
        trigger_pairs=5, keep_recent=2, last_assistant_ts=None,
    )
    result = apply_microcompact(_three_pairs(), policy)
    assert result.cleared_pair_count == 0


def test_time_trigger_forces_clear_when_gap_exceeded() -> None:
    # Same 3 pairs, but the gap is 90 minutes (> 60-min default) → force clear.
    long_ago = time.time() - 90 * 60
    policy = MicrocompactPolicy(
        trigger_pairs=5,
        keep_recent=2,
        last_assistant_ts=long_ago,
        gap_threshold_minutes=60,
    )
    result = apply_microcompact(_three_pairs(), policy)
    # 3 pairs, keep_recent=2 → exactly 1 oldest pair cleared.
    assert result.cleared_pair_count == 1
    assert "tc-0" in result.cleared_tool_call_ids


def test_time_trigger_no_op_when_gap_below_threshold() -> None:
    recent = time.time() - 5 * 60  # 5 minutes ago
    policy = MicrocompactPolicy(
        trigger_pairs=5,
        keep_recent=2,
        last_assistant_ts=recent,
        gap_threshold_minutes=60,
    )
    result = apply_microcompact(_three_pairs(), policy)
    assert result.cleared_pair_count == 0


def test_time_trigger_disabled_when_threshold_zero() -> None:
    long_ago = time.time() - 100 * 60
    policy = MicrocompactPolicy(
        trigger_pairs=5,
        keep_recent=2,
        last_assistant_ts=long_ago,
        gap_threshold_minutes=0,  # disabled
    )
    result = apply_microcompact(_three_pairs(), policy)
    assert result.cleared_pair_count == 0


def test_time_trigger_respects_keep_recent_floor() -> None:
    # 5 pairs, keep_recent=2, time-trigger fires.
    msgs: list[BaseMessage] = []
    for i in range(5):
        msgs.append(HumanMessage(content=f"u-{i}"))
        msgs.append(_ai_call("read_file", f"tc-{i}"))
        msgs.append(_tool(f"tc-{i}", name="read_file", content=f"r-{i}"))
    long_ago = time.time() - 90 * 60
    policy = MicrocompactPolicy(
        trigger_pairs=10,
        keep_recent=2,
        last_assistant_ts=long_ago,
        gap_threshold_minutes=60,
    )
    result = apply_microcompact(msgs, policy)
    # 5 pairs, keep_recent=2 → 3 oldest cleared.
    assert result.cleared_pair_count == 3
    assert set(result.cleared_tool_call_ids) == {"tc-0", "tc-1", "tc-2"}
