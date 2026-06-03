"""Summary compaction: rebuild = [<session-summary>, recent_files, skills, tasks, *tail]."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage

from aura.application.compact.constants import KEEP_LAST_N_TURNS
from aura.application.compact.summary_turn import (
    _CompactSession,
    _is_prompt_too_long,
    _run_summary_turn_with_retry,
    compact_summary_messages,
    compact_summary_prompt_budget,
    estimate_compact_summary_tokens,
    summary_caps_from_agent,
)
from aura.application.memory import project_memory, rules
from aura.application.memory.context import ReadRecord
from aura.domain.skill import Skill
from aura.infrastructure.persistence import journal

__all__ = [
    "CompactResult",
    "CompactSource",
    "_CompactSession",
    "_is_prompt_too_long",
    "_run_summary_turn_with_retry",
    "compact_summary_messages",
    "estimate_compact_summary_tokens",
    "run_compact",
]

CompactSource = Literal["manual", "auto", "reactive"]


@dataclass(frozen=True)
class CompactResult:
    before_tokens: int
    after_tokens: int
    source: CompactSource


def _build_recent_file_messages(
    read_records: dict[Path, ReadRecord],
    *,
    max_files_to_restore: int,
    max_tokens_per_file: int,
) -> list[HumanMessage]:
    """Render <recent-file> for FULL reads; partial reads omitted (incomplete view misleads)."""
    ranked = sorted(
        read_records.items(), key=lambda kv: kv[1].mtime, reverse=True,
    )
    max_chars = max_tokens_per_file * 4
    messages: list[HumanMessage] = []
    for path, record in ranked:
        if len(messages) >= max_files_to_restore:
            break
        if record.partial:
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(body) > max_chars:
            body = body[:max_chars] + "\n… (truncated)"
        messages.append(
            HumanMessage(
                content=f'<recent-file path="{path}">\n{body}\n</recent-file>',
            )
        )
    return messages


_MAX_TOKENS_PER_SKILL_BODY = 5_000


def _build_skill_reinjection_messages(
    invoked_skills: list[Skill],
) -> list[HumanMessage]:
    max_chars = _MAX_TOKENS_PER_SKILL_BODY * 4
    out: list[HumanMessage] = []
    for skill in invoked_skills:
        body = skill.body
        if len(body) > max_chars:
            body = body[:max_chars] + "\n… (truncated)"
        out.append(
            HumanMessage(
                content=f'<skill-active name="{skill.name}">\n{body}\n</skill-active>',
            ),
        )
    return out


def _build_active_task_messages(agent: _CompactSession) -> list[HumanMessage]:
    """Surface still-running / un-observed subagent tasks across the compact boundary."""
    out: list[HumanMessage] = []
    for rec in agent.tasks_store.list():
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


async def run_compact(
    agent: _CompactSession, *, source: CompactSource = "manual",
) -> CompactResult:
    before_tokens = agent.state.total_tokens_used
    history = agent.storage.load(agent.session_id)

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

    summary_caps = summary_caps_from_agent(agent)
    summary_text = await _run_summary_turn_with_retry(
        agent.model,
        to_summarize,
        max_prompt_tokens=compact_summary_prompt_budget(agent),
        caps=summary_caps,
    )

    old_ctx = agent.context
    preserved_read_records = dict(old_ctx.read_records)
    preserved_invoked_skills = list(old_ctx.invoked_skills)

    compact_cfg = agent.config.compact
    recent_file_msgs = _build_recent_file_messages(
        preserved_read_records,
        max_files_to_restore=compact_cfg.max_files_to_restore,
        max_tokens_per_file=compact_cfg.max_tokens_per_file,
    )

    skill_msgs = _build_skill_reinjection_messages(preserved_invoked_skills)
    active_task_msgs = _build_active_task_messages(agent)

    new_history: list[BaseMessage] = [
        HumanMessage(
            content=f"<session-summary>\n{summary_text}\n</session-summary>",
        ),
        *recent_file_msgs,
        *skill_msgs,
        *active_task_msgs,
        *preserved_tail,
    ]

    # Reload disk state for future rebuilds; THIS Context keeps old state (summary encoded it).
    project_memory.clear_cache(agent.cwd)
    rules.clear_cache(agent.cwd)
    agent.reload_memory_and_rules()

    new_ctx = old_ctx.fresh()
    new_ctx.bind_read_records(preserved_read_records)

    agent.apply_compaction(
        new_history=new_history,
        new_context=new_ctx,
        preserved_skills=preserved_invoked_skills,
    )

    after_tokens = agent.state.total_tokens_used
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
