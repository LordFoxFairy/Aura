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
from aura.core.compact.prompt import SUMMARY_SYSTEM, SUMMARY_USER_PREFIX
from aura.core.memory import project_memory, rules
from aura.core.memory.context import Context, _ReadRecord
from aura.core.persistence import journal

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
        lines.append(f"[{role}] {content}")
        # Tool calls live on AIMessage; serialize inline for legibility.
        tool_calls = getattr(m, "tool_calls", None) or []
        for tc in tool_calls:
            lines.append(
                f"    -> tool_call {tc.get('name')!r} args={tc.get('args')!r}"
            )
    return "\n".join(lines)


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
    to_summarize = history[:-tail_count]

    # Run the summary turn in isolation — no hooks, no tools. Using
    # ``ainvoke`` on the raw model sidesteps the bound tools so the model
    # CAN'T call one even if tempted.
    summary_text = await _run_summary_turn(agent._model, to_summarize)

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

    # Rebuild history: summary tag + recent files + preserved tail.
    new_history: list[BaseMessage] = [
        HumanMessage(
            content=f"<session-summary>\n{summary_text}\n</session-summary>",
        ),
        *recent_file_msgs,
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
        todos_provider=lambda: agent._state.custom.get("todos", []),
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
