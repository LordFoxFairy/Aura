"""Microcompact — per-turn prompt-view compression of old tool pairs.

Ported from claude-code v2.1.88's ``messagesForQuery`` microcompact pass
(``src/query.ts:412-468`` + ``timeBasedMCConfig.ts:30-50``). The goal is to
cut 30–50 % of mid-session context pressure by replacing the *payload* of
old ``tool_use``/``tool_result`` pairs with a compact marker, WITHOUT
mutating stored session history.

Design rules (see ``tasks/todo.md`` — user pre-approved):

1. **View transform, not history mutation.** All functions here return new
   lists; the caller's ``messages`` list is never mutated in place. Storage
   (``SessionStorage``) keeps full payloads — resume/export/replay stay
   pristine. Only the outgoing prompt to the LLM drops the payloads.
2. **No LLM call.** Pure textual substitution with a fixed marker. Adding a
   summary LLM turn per main-loop iteration would regress latency + cost
   with no user-visible upside (claude-code's microcompact is also purely
   textual).
3. **Pair-count trigger + keep-recent-N policy.** Deterministic, cheap,
   predictable. Default 8 / 5 matches claude-code.
4. **Hard floor on keep-recent.** Even if ``keep_recent=0`` is configured,
   we always keep the single most recent pair intact — mirrors
   ``Math.max(1, config.keepRecent)`` in ``timeBasedMCConfig.ts``. Clearing
   *every* tool result in the outgoing prompt would strand the model with
   no observable results for the last turn.
5. **AIMessage.tool_calls untouched.** Even when clearing a pair's result,
   the AIMessage that issued the call is left as-is. The model must still
   see that the call was issued — removing the call itself would create an
   orphaned ToolMessage (provider-level schema violation on replay).

This module is pure: no I/O, no hooks, no journal writes. Side effects
(journal emission, state updates) are the caller's responsibility — see
``aura/core/loop.py``'s ``_invoke_model`` integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from aura.core.compact.constants import (
    MICROCOMPACT_CLEAR_MARKER,
    MICROCOMPACT_COMPACTABLE_TOOLS,
    MICROCOMPACT_KEEP_RECENT,
    MICROCOMPACT_TRIGGER_PAIRS,
)


@dataclass(frozen=True)
class ToolPair:
    """One matched AIMessage.tool_call → ToolMessage pair.

    ``ai_idx`` and ``tool_idx`` index into the message list that was passed
    to :func:`find_tool_pairs`. They're carried on the dataclass for
    observability / debugging — callers don't need them for ``apply_clear``
    (which keys off ``tool_call_id``), but they make journal events and
    failure diagnostics materially easier to read.
    """

    ai_idx: int
    tool_idx: int
    tool_call_id: str
    tool_name: str


@dataclass(frozen=True)
class MicrocompactPolicy:
    """Knobs for a single microcompact application.

    Constructed fresh per ``_invoke_model`` call (cheap — frozen dataclass
    with two ints + one frozenset + one str). Callers override only the
    fields they care about; everything else tracks the module-level
    constants so a single edit to ``constants.py`` propagates.
    """

    trigger_pairs: int = MICROCOMPACT_TRIGGER_PAIRS
    keep_recent: int = MICROCOMPACT_KEEP_RECENT
    compactable_tools: frozenset[str] = field(
        default_factory=lambda: MICROCOMPACT_COMPACTABLE_TOOLS,
    )
    clear_marker: str = MICROCOMPACT_CLEAR_MARKER
    # F-0910-004: time-based force trigger. When the wall-clock gap since
    # ``last_assistant_ts`` exceeds ``gap_threshold_minutes``, microcompact
    # fires regardless of pair count — long idle gaps invalidate the model's
    # cached working set anyway, so a forced compress on resume is cheaper
    # than rehydrating stale tool payloads. ``last_assistant_ts=None``
    # disables the time trigger (parity with ``trigger_pairs<=0`` for the
    # pair trigger).
    last_assistant_ts: float | None = None
    gap_threshold_minutes: int = 60


@dataclass(frozen=True)
class MicrocompactResult:
    """Outcome of an :func:`apply_microcompact` call.

    ``cleared_tool_call_ids`` is a tuple (not set) because callers want
    *ordered* evidence of which pairs got cleared — the journal event
    serializes this verbatim, and tests/dogfood sessions benefit from
    stable oldest-first ordering. Using a tuple also keeps the dataclass
    hashable, matching the rest of the compact module's conventions.

    ``cleared_pairs`` carries the full :class:`ToolPair` records (with
    ``ai_idx`` / ``tool_idx``) so callers that emit journal events or
    render diagnostics can pinpoint *which positions* in the message list
    were cleared, not just which tool_call_ids. Same oldest-first order
    as ``cleared_tool_call_ids``.
    """

    messages: list[BaseMessage]
    cleared_pair_count: int
    cleared_tool_call_ids: tuple[str, ...]
    cleared_pairs: tuple[ToolPair, ...]


# ---------------------------------------------------------------------------
# T1 — pair detection
# ---------------------------------------------------------------------------


def find_tool_pairs(
    messages: list[BaseMessage],
    compactable_tools: frozenset[str],
) -> list[ToolPair]:
    """Return ``(AIMessage.tool_call → ToolMessage)`` pairs in encounter order.

    Algorithm: walk the message list once, for every ``AIMessage.tool_calls``
    entry whose tool name is in ``compactable_tools`` search the remainder of
    the list for the first ``ToolMessage`` whose ``tool_call_id`` matches.
    Unmatched calls (streaming interrupted mid-turn, model retried, etc.)
    are silently skipped — they're not an error at this layer.

    Complexity is O(n·m) worst case where n = AIMessage count and m = messages
    per tool_call search window. In practice n is small (one AIMessage per
    model turn) and the search window is bounded by the next AIMessage, so
    this is effectively linear. We don't pre-index because:

    - The ``tool_call_id`` collision case (first-match-wins) requires
      positional scanning anyway.
    - A dict pre-index would eagerly allocate for the common case of very
      short outgoing prompts; the walk is fast enough that it's not worth it.
    """
    pairs: list[ToolPair] = []
    matched_tool_indices: set[int] = set()  # so we honour first-match-wins

    for ai_idx, msg in enumerate(messages):
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = msg.tool_calls or []
        for tc in tool_calls:
            tool_name = tc.get("name") or ""
            tool_call_id = tc.get("id") or ""
            if not tool_call_id:
                # Defensive: LangChain can in theory emit a partial tool_call
                # without an id during streaming. Nothing to match against.
                continue
            if tool_name not in compactable_tools:
                continue
            # Search forward for the first ToolMessage with matching id that
            # hasn't already been claimed by an earlier pair.
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
                break  # first match wins — stop scanning for this tool_call
    return pairs


# ---------------------------------------------------------------------------
# T2 — selection policy
# ---------------------------------------------------------------------------


def select_clear_ids(
    pairs: list[ToolPair],
    trigger_pairs: int,
    keep_recent: int,
) -> set[str]:
    """Pick tool_call_ids to clear per keep-recent-N policy.

    Contract:
      - ``len(pairs) <= trigger_pairs`` → empty set (under threshold, no-op).
      - Otherwise, keep the last ``max(1, keep_recent)`` pairs; clear the rest.

    The ``max(1, keep_recent)`` floor mirrors claude-code's
    ``Math.max(1, config.keepRecent)`` in ``timeBasedMCConfig.ts``. Intent:
    even with a pathologically low config, never starve the outgoing prompt
    of *all* tool results — the model would lose its footing immediately.
    """
    if len(pairs) <= trigger_pairs:
        return set()
    effective_keep = max(1, keep_recent)
    # Slicing with a negative index is safe: Python clamps on bounds, so
    # ``pairs[-effective_keep:]`` never errors. The cleared slice is
    # everything BEFORE the last ``effective_keep`` entries — i.e., the
    # oldest compactable pairs.
    to_clear = pairs[:-effective_keep]
    return {pair.tool_call_id for pair in to_clear}


# ---------------------------------------------------------------------------
# T3 — view transform
# ---------------------------------------------------------------------------


def _rebuild_tool_message(original: ToolMessage, marker: str) -> ToolMessage:
    """Construct a new ToolMessage mirroring ``original`` but with marker content.

    Preserved: ``tool_call_id``, ``name``, ``status``, ``artifact``,
    ``additional_kwargs``. The new ToolMessage is a fresh object — mutating
    it must not leak back to the stored history.

    Implementation: construct with the overload-typed positional signature
    (``content=``) then assign the remaining fields on the instance. LangChain's
    ``ToolMessage.__init__`` overloads don't include typed kwargs for
    ``artifact``/``status``/``name`` — they come through Pydantic model
    validation — so passing them inline to the constructor doesn't satisfy
    mypy even though it runs fine. Building then assigning sidesteps that,
    and it's still a single object allocation.
    """
    new_msg = ToolMessage(
        content=marker,
        tool_call_id=original.tool_call_id,
    )
    if original.name is not None:
        new_msg.name = original.name
    # ``status`` has a non-None default on ToolMessage ("success"), so just
    # forward whatever the original carries — it's always a valid value.
    new_msg.status = original.status
    if original.artifact is not None:
        new_msg.artifact = original.artifact
    if original.additional_kwargs:
        # Shallow copy — callers who stored mutable structures in
        # additional_kwargs already accept shared-reference semantics.
        new_msg.additional_kwargs = dict(original.additional_kwargs)
    return new_msg


def apply_clear(
    messages: list[BaseMessage],
    clear_ids: set[str],
    marker: str,
) -> list[BaseMessage]:
    """Return a new list with matching ToolMessage payloads replaced by ``marker``.

    Contract:
      - Untouched messages are returned by reference (no deep copy). This
        matters for memory pressure — outgoing prompts can carry MBs of
        file bodies that would be cloned per-turn otherwise.
      - Matched ToolMessages become a *new* ToolMessage object whose
        ``content`` is ``marker`` but whose ``tool_call_id``, ``name``,
        ``status``, ``artifact``, and ``additional_kwargs`` track the
        original.
      - The corresponding ``AIMessage.tool_calls`` entries are never
        modified — the provider-level schema requires the call to remain
        visible alongside its (now-cleared) result.
      - Empty ``clear_ids`` short-circuits to the original list reference.
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


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


def apply_microcompact(
    messages: list[BaseMessage],
    policy: MicrocompactPolicy,
) -> MicrocompactResult:
    """Apply the full microcompact pipeline to ``messages``.

    Pipeline:
      1. ``find_tool_pairs`` — enumerate compactable pairs in order.
      2. ``select_clear_ids`` — decide which ids to clear per policy.
      3. ``apply_clear`` — produce a new message list with cleared payloads.

    Disabled path: ``policy.trigger_pairs <= 0`` returns the input
    ``messages`` list by reference with ``cleared_pair_count=0``. This
    matches the parity pattern used by ``auto_compact_threshold=0`` — a
    single knob disables the feature without requiring a separate enabled
    flag at this layer (the Agent constructor gates enablement explicitly).

    ``cleared_tool_call_ids`` is ordered oldest-first. The order is
    load-bearing for journal events and for downstream observability: a
    reader scanning the journal can reconstruct which span of turns was
    compressed without resorting the ids.
    """
    if policy.trigger_pairs <= 0:
        return MicrocompactResult(
            messages=messages,
            cleared_pair_count=0,
            cleared_tool_call_ids=(),
            cleared_pairs=(),
        )
    pairs = find_tool_pairs(messages, policy.compactable_tools)
    # F-0910-004: time-based force trigger — when the wall-clock gap since
    # ``last_assistant_ts`` exceeds ``gap_threshold_minutes``, force a
    # clear by lowering the effective trigger to ``keep_recent`` so any
    # surplus pairs above the keep-recent floor get compressed.
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
    # Preserve oldest-first order in the returned tuple — ``pairs`` is already
    # in encounter order (T1's contract), so filtering it by clear_ids keeps
    # the ordering without another sort.
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
