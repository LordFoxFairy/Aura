"""Conversation compaction — summarize old history, preserve session state.

Public surface: :class:`CompactResult` + :func:`run_compact`. ``Agent.compact``
is a thin wrapper around ``run_compact`` — the free function shape keeps the
Agent method body readable and the logic independently testable.

Flow (authoritative — mirrors the plan in v0.4.0):

1. If history is too short (fewer than KEEP_LAST_N_TURNS * 2 messages), no-op.
2. Split history into ``to_summarize`` (middle) and ``preserved_tail``
   (last KEEP_LAST_N_TURNS turns raw).
3. Run a *single-turn* summary: SystemMessage + HumanMessage containing a
   text serialization of ``to_summarize``. The prompt tells the model
   explicitly to respond with TEXT ONLY — tool calls are discarded.
4. Rebuild history as ``[<session-summary> HumanMessage, *preserved_tail]``.
5. Clear discovery caches (project memory, rules, nested fragments, matched
   rules) — these are derived state and will be re-discovered next turn.
   Preserve genuine session state (read_records, invoked skills, todos,
   session rules, accumulated token count).
6. Write a ``compact_applied`` journal event.

The Context after compact is a NEW instance — its progressive fields are
frozen on write (nested fragments, matched rules) and must be empty after
compact, which is easier to guarantee with a fresh object than an in-place
reset (and matches the docstring invariant in
``aura/core/memory/context.py``).
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from aura.core.compact.constants import (
    KEEP_LAST_N_TURNS,
    MAX_FILES_TO_RESTORE,
    MAX_TOKENS_PER_FILE,
)
from aura.core.compact.microcompact import MicrocompactPolicy, apply_microcompact
from aura.core.compact.prompt import SUMMARY_SYSTEM, SUMMARY_USER_PREFIX
from aura.core.memory import project_memory, rules
from aura.core.memory.context import Context, _ReadRecord
from aura.core.persistence import journal
from aura.core.tokens import estimate_text_tokens

if TYPE_CHECKING:
    from aura.core.agent import Agent

CompactSource = Literal["manual", "auto", "reactive"]


@dataclass(frozen=True)
class CompactResult:
    """Outcome of a compaction.

    ``before_tokens`` is the running ``LoopState.total_tokens_used`` at the
    moment compact() was called; ``after_tokens`` is the same counter after
    the summary turn (which may add usage if a usage tracking hook is
    wired — in Aura it's wired via ``default_hooks``, so auto-compact runs
    in production will show the delta).
    """

    before_tokens: int
    after_tokens: int
    source: CompactSource


def _build_recent_file_messages(
    read_records: dict[Path, _ReadRecord],
) -> list[HumanMessage]:
    """Render up to MAX_FILES_TO_RESTORE <recent-file> HumanMessages.

    Selection rule: sort by mtime DESC, drop entries whose recorded read was
    partial, take top ``MAX_FILES_TO_RESTORE``. Files that can no longer be
    read from disk (deleted / permission-changed post-read) are silently
    skipped — this path runs AFTER the summary turn already captured them by
    text, so losing their body is not catastrophic.

    Each file's body is capped at ``MAX_TOKENS_PER_FILE * 4`` characters
    (4 chars/token ballpark for English/code mix); oversize bodies get a
    trailing ``… (truncated)`` marker so the model knows content is missing.
    """
    ranked = sorted(
        read_records.items(), key=lambda kv: kv[1].mtime, reverse=True,
    )
    max_chars = MAX_TOKENS_PER_FILE * 4
    messages: list[HumanMessage] = []
    for path, record in ranked:
        if len(messages) >= MAX_FILES_TO_RESTORE:
            break
        if record.partial:
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # File deleted, renamed, or unreadable since the original read.
            # Benign — skip rather than fail the whole compact.
            continue
        if len(body) > max_chars:
            body = body[:max_chars] + "\n… (truncated)"
        messages.append(
            HumanMessage(
                content=f'<recent-file path="{path}">\n{body}\n</recent-file>',
            )
        )
    return messages


# F-0910-008: per-skill re-injection cap (4 chars/token ballpark).
_MAX_TOKENS_PER_SKILL_BODY = 5_000


def _build_skill_reinjection_messages(
    invoked_skills: Sequence[object],
) -> list[HumanMessage]:
    """Render one ``<skill-active>`` HumanMessage per preserved invoked skill.

    Body is capped at ``_MAX_TOKENS_PER_SKILL_BODY * 4`` characters; oversize
    bodies get a trailing ``… (truncated)`` marker. Skipped silently if the
    object doesn't carry a ``body`` attribute (defensive — same tolerance as
    ``_build_recent_file_messages`` applies to read records).
    """
    max_chars = _MAX_TOKENS_PER_SKILL_BODY * 4
    out: list[HumanMessage] = []
    for skill in invoked_skills:
        name = getattr(skill, "name", None)
        body = getattr(skill, "body", None)
        if not isinstance(name, str) or not isinstance(body, str):
            continue
        if len(body) > max_chars:
            body = body[:max_chars] + "\n… (truncated)"
        out.append(
            HumanMessage(
                content=f'<skill-active name="{name}">\n{body}\n</skill-active>',
            ),
        )
    return out


def _build_active_task_messages(agent: Agent) -> list[HumanMessage]:
    """F-0910-020: surface still-relevant subagent tasks across compact.

    Walks ``agent._tasks_store.list()`` and emits one HumanMessage per task
    whose status is ``running`` or whose terminal result has not been observed
    via ``task_get`` / ``task_output``. Each message renders
    ``<active-task id=... status=... last_seen_at=...>`` so a model resuming
    after compact knows what work is still open.

    Tolerates a missing ``_tasks_store`` (synthetic / partial Agent in tests)
    by returning an empty list.
    """
    store = getattr(agent, "_tasks_store", None)
    if store is None:
        return []
    out: list[HumanMessage] = []
    for rec in store.list():
        if rec.status != "running" and rec.observed_at is not None:
            continue
        last_seen = (
            rec.progress.last_activity_at
            if rec.progress.last_activity_at is not None
            else rec.started_at
        )
        out.append(
            HumanMessage(
                content=(
                    f'<active-task id="{rec.id}" status="{rec.status}" '
                    f'last_seen_at="{last_seen}">\n'
                    f"{rec.description}\n"
                    "</active-task>"
                ),
            ),
        )
    return out


_MAX_SUMMARY_MESSAGE_CHARS = 6_000
_MAX_SUMMARY_TOOL_ARGS_CHARS = 2_000


def _cap_summary_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n... (truncated; {omitted} chars omitted)"


def _serialize_tool_args(args: object) -> str:
    try:
        import json

        rendered = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = str(args)
    return _cap_summary_text(rendered, max_chars=_MAX_SUMMARY_TOOL_ARGS_CHARS)


def _serialize_history(messages: list[BaseMessage]) -> str:
    """Flatten messages into a role-tagged text block for the summary prompt.

    The summary model doesn't need round-trip-perfect message schema — it
    just needs to read what happened. Tool calls are shown as their raw
    args (JSON-ish via str) to keep the encoding cheap.
    """
    lines: list[str] = []
    for m in messages:
        role = m.__class__.__name__.replace("Message", "").lower()
        content = str(m.content) if m.content else ""
        content = _cap_summary_text(content, max_chars=_MAX_SUMMARY_MESSAGE_CHARS)
        lines.append(f"[{role}] {content}")
        # Tool calls live on AIMessage; serialize inline for legibility.
        tool_calls = getattr(m, "tool_calls", None) or []
        for tc in tool_calls:
            lines.append(
                "    -> tool_call "
                f"{tc.get('name')!r} args={_serialize_tool_args(tc.get('args'))}"
            )
    return "\n".join(lines)


def _summary_turn_estimated_tokens(messages: list[BaseMessage]) -> int:
    return estimate_text_tokens(
        SUMMARY_SYSTEM + "\n" + SUMMARY_USER_PREFIX + _serialize_history(messages)
    )


def estimate_compact_summary_tokens(messages: list[BaseMessage]) -> int:
    """Public-for-command estimate of the manual compact summary prompt."""
    return _summary_turn_estimated_tokens(messages)


def compact_summary_messages(agent: Agent, history: list[BaseMessage]) -> list[BaseMessage]:
    """Return the history view that manual/reactive compact should summarize.

    Stored history remains raw. The summary model should see the same dynamic
    conversation surface the main loop would send to the provider, including
    microcompact's view-only clearing of old tool results.
    """
    policy = _microcompact_policy_for_agent(agent)
    if policy is None:
        return list(history)
    return apply_microcompact(list(history), policy).messages


def _microcompact_policy_for_agent(agent: Agent) -> MicrocompactPolicy | None:
    loop = getattr(agent, "_loop", None)
    policy = getattr(loop, "_microcompact_policy", None)
    return policy if isinstance(policy, MicrocompactPolicy) else None


def _split_for_summary_budget(
    messages: list[BaseMessage],
    *,
    max_prompt_tokens: int,
) -> list[list[BaseMessage]]:
    chunks: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for message in messages:
        candidate = [*current, message]
        if current and _summary_turn_estimated_tokens(candidate) > max_prompt_tokens:
            chunks.append(current)
            current = [message]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


async def run_compact(agent: Agent, *, source: CompactSource = "manual") -> CompactResult:
    """Execute a compaction cycle on ``agent``.

    Exposed as a free function for testability and to keep
    ``Agent.compact`` short. ``agent`` is typed as the concrete Agent
    rather than a Protocol because we need access to private
    attributes (``_history``, ``_context``, ``_state``, …) — this is the
    intra-module pair, not an external API surface.
    """
    from aura.core.hooks.must_read_first import make_must_read_first_hook

    before_tokens = agent._state.total_tokens_used
    history = agent._storage.load(agent.session_id)

    # Nothing to summarize — short-circuit.
    if len(history) < KEEP_LAST_N_TURNS * 2:
        journal.write(
            "compact_applied",
            source=source,
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            noop=True,
            history_len_before=len(history),
            history_len_after=len(history),
        )
        return CompactResult(
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            source=source,
        )

    tail_count = KEEP_LAST_N_TURNS * 2
    preserved_tail = history[-tail_count:]
    summary_history = compact_summary_messages(agent, history)
    to_summarize = summary_history[:-tail_count]

    # Run the summary turn in isolation — no hooks, no tools. Using
    # ``ainvoke`` on the raw model sidesteps the bound tools so the model
    # CAN'T call one even if tempted. F-0910-003: on PromptTooLong /
    # context-overflow during the summary call itself, drop the oldest
    # 20% of ``to_summarize`` and retry up to 3 times before raising.
    summary_text = await _run_summary_turn_with_retry(
        agent._model,
        to_summarize,
        max_prompt_tokens=_compact_summary_prompt_budget(agent),
    )

    # --- Pre-cleanup state capture (lifted BEFORE we build new_history so
    # the re-injection step below can consult the outgoing read_records). ---
    old_ctx = agent._context
    preserved_read_records = dict(old_ctx._read_records)
    preserved_invoked_skills = list(old_ctx._invoked_skills)
    preserved_invoked_paths = set(old_ctx._invoked_skill_paths)

    # Selective file re-injection: the summary may be too lean to continue
    # work on a specific file. Re-inject the most-recently-touched FULL
    # reads' bodies as <recent-file> HumanMessages. Partial reads are skipped
    # (an incomplete view confuses the model more than omitting the body).
    recent_file_msgs = _build_recent_file_messages(preserved_read_records)

    # F-0910-008 / Phase 1 Task 6: stash the pre-compact invoked-skill list
    # on the typed ``state.slots.preserved_invoked_skills`` slot (was
    # ``state.custom``) so the producer side is observable from the test
    # surface + so the SUBAGENT-STOP / future post-compact hooks can find
    # it without reaching back into the (now-replaced) Context. In-place
    # ``[:] = ...`` keeps the list identity stable (the slot is frozen at
    # the attribute level but the contained list accepts mutation; same
    # pattern as ``slots.turn_denials.clear()``).
    agent._state.slots.preserved_invoked_skills[:] = list(
        preserved_invoked_skills,
    )
    # F-0910-008: re-inject one HumanMessage per active skill (body capped
    # at MAX_TOKENS_PER_SKILL_BODY tokens). The static <skill-invoked>
    # rendering (lifted onto new_ctx below) is enough for catalogue
    # awareness, but skill bodies often carry instructions the model needs
    # AS PART OF the conversation flow — not just preamble — so we hoist
    # them into history alongside the summary.
    skill_msgs = _build_skill_reinjection_messages(preserved_invoked_skills)

    # F-0910-020: SUBAGENT-STOP semantics — surface still-running and
    # completed-but-not-retrieved subagent tasks as one HumanMessage each so
    # the model resuming after compact knows about open work it spawned.
    active_task_msgs = _build_active_task_messages(agent)

    # Rebuild history: summary tag + recent files + skill bodies + active
    # tasks + preserved tail.
    new_history: list[BaseMessage] = [
        HumanMessage(
            content=f"<session-summary>\n{summary_text}\n</session-summary>",
        ),
        *recent_file_msgs,
        *skill_msgs,
        *active_task_msgs,
        *preserved_tail,
    ]
    agent._storage.save(agent.session_id, new_history)

    # Drop module-level caches so the next build re-reads CLAUDE.md / rules.
    project_memory.clear_cache(agent._cwd)
    rules.clear_cache(agent._cwd)
    agent._primary_memory = project_memory.load_project_memory(agent._cwd)
    agent._rules = rules.load_rules(agent._cwd)

    # New Context — progressive fields (nested fragments, matched rules) are
    # fresh-empty by construction; no in-place reset needed.
    new_ctx = Context(
        cwd=agent._cwd,
        system_prompt=agent._system_prompt,
        primary_memory=agent._primary_memory,
        rules=agent._rules,
        skills=agent._skill_registry.list(),
        todos_provider=lambda: agent._state.slots.todos,
    )

    # Lift preserved state onto the new instance.
    new_ctx._read_records = preserved_read_records
    new_ctx._invoked_skills = preserved_invoked_skills
    new_ctx._invoked_skill_paths = preserved_invoked_paths

    agent._context = new_ctx

    # Re-swap the must_read_first hook so its closure sees the NEW Context
    # (which carries the preserved records). Same pattern as clear_session.
    # ``suppress(ValueError)`` for the (unusual) case where a caller removed
    # the hook out-of-band — the fresh append below keeps the chain valid.
    with contextlib.suppress(ValueError):
        agent._hooks.pre_tool.remove(agent._must_read_first_hook)
    agent._must_read_first_hook = make_must_read_first_hook(agent._context)
    agent._hooks.pre_tool.append(agent._must_read_first_hook)

    # Rebuild the loop so it points at the new context; keeps model + hooks
    # + registry + shared state references.
    agent._loop = agent._build_loop()

    after_tokens = agent._state.total_tokens_used
    journal.write(
        "compact_applied",
        source=source,
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        noop=False,
        history_len_before=len(history),
        history_len_after=len(new_history),
    )
    return CompactResult(
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        source=source,
    )


_PTL_PHRASES: tuple[str, ...] = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "prompt is too long",
    "prompt exceeds max length",
    "exceeds max length",
    "input too long",
    "too many tokens",
    "prompttoolong",
)

_PTL_CODES: tuple[str, ...] = (
    "1261",  # DashScope: "Prompt exceeds max length"
)


def _is_prompt_too_long(exc: BaseException) -> bool:
    """True iff ``exc``'s message looks like a context-overflow signature.

    Mirrors ``aura.core.agent._is_context_overflow`` — kept duplicated to
    avoid an import cycle (compact already gets reached by Agent).
    """
    msg = str(exc).lower()
    if any(phrase in msg for phrase in _PTL_PHRASES):
        return True
    if type(exc).__name__.lower() in {"prompttoolongerror", "prompttoolong"}:
        return True
    return any(
        f"'code': '{code}'" in msg or f'"code": "{code}"' in msg
        for code in _PTL_CODES
    )


_MAX_SUMMARY_SPLIT_DEPTH = 12
_FALLBACK_SUMMARY_CHAR_LIMIT = 12_000
_MAX_COMPACT_SUMMARY_PROMPT_TOKENS = 16_000


def _compact_summary_prompt_budget(agent: Agent) -> int:
    """Keep manual summary calls well below the live model window."""
    window = getattr(agent, "context_window", 0) or 0
    if window <= 0:
        return _MAX_COMPACT_SUMMARY_PROMPT_TOKENS
    # Leave the same 13k scratch headroom used by auto-compact, then cap
    # summary prompts to a conservative ceiling. Compact is a maintenance
    # operation; multiple small calls are better than one provider 400.
    return max(2_000, min(window - 13_000, _MAX_COMPACT_SUMMARY_PROMPT_TOKENS))


async def _run_summary_turn_with_retry(
    model: BaseChatModel,
    to_summarize: list[BaseMessage],
    *,
    max_prompt_tokens: int = _MAX_COMPACT_SUMMARY_PROMPT_TOKENS,
) -> str:
    """Summarize history, splitting recursively if the provider rejects size.

    The old retry strategy dropped the oldest 20% three times. That avoided
    one class of context errors but could still lose useful state and still
    fail on providers with smaller windows. The resilient path below mirrors
    code-agent compaction behavior more closely: summarize chunks, then
    summarize the chunk summaries.
    """
    chunks = _split_for_summary_budget(
        list(to_summarize),
        max_prompt_tokens=max_prompt_tokens,
    )
    if not chunks:
        return ""
    if len(chunks) == 1:
        return await _run_summary_turn_resilient(model, chunks[0], depth=0)

    partials: list[BaseMessage] = []
    for idx, chunk in enumerate(chunks, start=1):
        text = await _run_summary_turn_resilient(model, chunk, depth=0)
        partials.append(
            HumanMessage(
                content=(
                    f'<partial-summary index="{idx}" total="{len(chunks)}">\n'
                    f"{text}\n"
                    "</partial-summary>"
                ),
            ),
        )
    return await _run_summary_turn_with_retry(
        model,
        partials,
        max_prompt_tokens=max_prompt_tokens,
    )


async def _run_summary_turn_resilient(
    model: BaseChatModel,
    messages: list[BaseMessage],
    *,
    depth: int,
) -> str:
    if not messages:
        return ""
    try:
        return await _run_summary_turn(model, messages)
    except Exception as exc:  # noqa: BLE001 — providers vary widely
        if not _is_prompt_too_long(exc):
            raise
        if len(messages) == 1 or depth >= _MAX_SUMMARY_SPLIT_DEPTH:
            return _fallback_summary(messages)

    midpoint = max(1, len(messages) // 2)
    left = await _run_summary_turn_resilient(
        model, messages[:midpoint], depth=depth + 1,
    )
    right = await _run_summary_turn_resilient(
        model, messages[midpoint:], depth=depth + 1,
    )
    merged: list[BaseMessage] = [
        HumanMessage(
            content=(
                '<partial-summary index="1">\n'
                f"{left}\n"
                "</partial-summary>"
            ),
        ),
        HumanMessage(
            content=(
                '<partial-summary index="2">\n'
                f"{right}\n"
                "</partial-summary>"
            ),
        ),
    ]
    return await _run_summary_turn_resilient(model, merged, depth=depth + 1)


def _fallback_summary(messages: list[BaseMessage]) -> str:
    """Deterministic last resort when even a single message is too large."""
    serialized = _serialize_history(messages)
    if len(serialized) > _FALLBACK_SUMMARY_CHAR_LIMIT:
        serialized = serialized[:_FALLBACK_SUMMARY_CHAR_LIMIT] + "\n... (truncated)"
    return (
        "<goal>Conversation history was compacted without a model summary "
        "because the provider rejected the compact prompt size.</goal>\n"
        "<decisions>Preserved a deterministic excerpt of the oldest "
        "history instead of crashing the session.</decisions>\n"
        "<files-touched>See excerpt if file paths were present.</files-touched>\n"
        "<tools-used>See excerpt if tool calls were present.</tools-used>\n"
        "<open-threads>Some older detail may be truncated; preserved tail "
        "messages remain raw.</open-threads>\n"
        "<next-steps>Continue from the preserved recent turns.</next-steps>\n"
        "<history-excerpt>\n"
        f"{serialized}\n"
        "</history-excerpt>"
    )


async def _run_summary_turn(
    model: BaseChatModel, to_summarize: list[BaseMessage],
) -> str:
    """Invoke ``model`` once with SUMMARY_SYSTEM + serialized history.

    Uses the raw (unbound) model so tool calls aren't even an option for
    the model to attempt. Returns the text content of the assistant reply
    (best-effort — tool calls that somehow appear are ignored by design).
    """
    serialized = _serialize_history(to_summarize)
    messages: list[BaseMessage] = [
        SystemMessage(content=SUMMARY_SYSTEM),
        HumanMessage(content=SUMMARY_USER_PREFIX + serialized),
    ]
    ai = await model.ainvoke(messages)
    if isinstance(ai, AIMessage):
        return str(ai.content) if ai.content else ""
    # Defensive fallback — some providers may return a bare string-like
    # object. Keep the compact flow robust.
    return str(getattr(ai, "content", ai))
