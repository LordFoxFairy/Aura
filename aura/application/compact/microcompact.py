"""Microcompact: per-turn view-only compression of old tool pairs (storage stays raw)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from aura.application.compact.constants import (
    MICROCOMPACT_CLEAR_MARKER,
    MICROCOMPACT_COMPACTABLE_TOOLS,
    MICROCOMPACT_KEEP_RECENT,
    MICROCOMPACT_TRIGGER_PAIRS,
)


@dataclass(frozen=True)
class ToolPair:
    """One matched AIMessage.tool_call → ToolMessage pair."""

    ai_idx: int
    tool_idx: int
    tool_call_id: str
    tool_name: str


@dataclass(frozen=True)
class MicrocompactPolicy:
    """Knobs for one microcompact application.

    ``last_assistant_ts=None`` disables the time trigger.
    """

    trigger_pairs: int = MICROCOMPACT_TRIGGER_PAIRS
    keep_recent: int = MICROCOMPACT_KEEP_RECENT
    compactable_tools: frozenset[str] = field(
        default_factory=lambda: MICROCOMPACT_COMPACTABLE_TOOLS,
    )
    clear_marker: str = MICROCOMPACT_CLEAR_MARKER
    last_assistant_ts: float | None = None
    gap_threshold_minutes: int = 60


@dataclass(frozen=True)
class MicrocompactResult:
    """``cleared_tool_call_ids`` is oldest-first (load-bearing for journal)."""

    messages: list[BaseMessage]
    cleared_pair_count: int
    cleared_tool_call_ids: tuple[str, ...]
    cleared_pairs: tuple[ToolPair, ...]


def find_tool_pairs(
    messages: list[BaseMessage],
    compactable_tools: frozenset[str],
) -> list[ToolPair]:
    """Return AIMessage.tool_call → ToolMessage pairs in encounter order.

    Unmatched calls (streaming interrupted, retried) are skipped silently.
    """
    pairs: list[ToolPair] = []
    matched_tool_indices: set[int] = set()

    for ai_idx, msg in enumerate(messages):
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = msg.tool_calls or []
        for tc in tool_calls:
            tool_name = tc.get("name") or ""
            tool_call_id = tc.get("id") or ""
            if not tool_call_id:
                continue
            if tool_name not in compactable_tools:
                continue
            for tool_idx in range(ai_idx + 1, len(messages)):
                if tool_idx in matched_tool_indices:
                    continue
                candidate = messages[tool_idx]
                if not isinstance(candidate, ToolMessage):
                    continue
                if candidate.tool_call_id != tool_call_id:
                    continue
                pairs.append(
                    ToolPair(
                        ai_idx=ai_idx,
                        tool_idx=tool_idx,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                    ),
                )
                matched_tool_indices.add(tool_idx)
                break
    return pairs


def select_clear_ids(
    pairs: list[ToolPair],
    trigger_pairs: int,
    keep_recent: int,
) -> set[str]:
    """Keep last ``max(1, keep_recent)`` pairs; clear older ones."""
    if len(pairs) <= trigger_pairs:
        return set()
    effective_keep = max(1, keep_recent)
    to_clear = pairs[:-effective_keep]
    return {pair.tool_call_id for pair in to_clear}


def _rebuild_tool_message(original: ToolMessage, marker: str) -> ToolMessage:
    """Fresh ToolMessage carrying ``marker`` content, preserving other fields."""
    new_msg = ToolMessage(
        content=marker,
        tool_call_id=original.tool_call_id,
    )
    if original.name is not None:
        new_msg.name = original.name
    new_msg.status = original.status
    if original.artifact is not None:
        new_msg.artifact = original.artifact
    if original.additional_kwargs:
        new_msg.additional_kwargs = dict(original.additional_kwargs)
    return new_msg


def apply_clear(
    messages: list[BaseMessage],
    clear_ids: set[str],
    marker: str,
) -> list[BaseMessage]:
    """Return a new list with matching ToolMessage payloads replaced by ``marker``.

    Untouched messages are returned by reference (no deep copy).
    AIMessage.tool_calls is never modified (provider schema constraint).
    """
    if not clear_ids:
        return messages
    out: list[BaseMessage] = []
    for msg in messages:
        if (
            isinstance(msg, ToolMessage)
            and msg.tool_call_id in clear_ids
        ):
            out.append(_rebuild_tool_message(msg, marker))
        else:
            out.append(msg)
    return out


def apply_microcompact(
    messages: list[BaseMessage],
    policy: MicrocompactPolicy,
) -> MicrocompactResult:
    """Run the full pipeline; ``trigger_pairs<=0`` disables the feature."""
    if policy.trigger_pairs <= 0:
        return MicrocompactResult(
            messages=messages,
            cleared_pair_count=0,
            cleared_tool_call_ids=(),
            cleared_pairs=(),
        )
    pairs = find_tool_pairs(messages, policy.compactable_tools)
    effective_trigger = policy.trigger_pairs
    if (
        policy.last_assistant_ts is not None
        and policy.gap_threshold_minutes > 0
    ):
        gap_minutes = (time.time() - policy.last_assistant_ts) / 60.0
        if gap_minutes >= policy.gap_threshold_minutes:
            effective_trigger = max(1, policy.keep_recent)
    clear_ids = select_clear_ids(
        pairs,
        trigger_pairs=effective_trigger,
        keep_recent=policy.keep_recent,
    )
    if not clear_ids:
        return MicrocompactResult(
            messages=messages,
            cleared_pair_count=0,
            cleared_tool_call_ids=(),
            cleared_pairs=(),
        )
    ordered_cleared_pairs = tuple(
        pair for pair in pairs if pair.tool_call_id in clear_ids
    )
    new_messages = apply_clear(messages, clear_ids, policy.clear_marker)
    return MicrocompactResult(
        messages=new_messages,
        cleared_pair_count=len(ordered_cleared_pairs),
        cleared_tool_call_ids=tuple(p.tool_call_id for p in ordered_cleared_pairs),
        cleared_pairs=ordered_cleared_pairs,
    )
