"""Summary-turn mechanics: serialize history, budget-split, resilient model summary, fallback."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from aura.application.compact.microcompact import MicrocompactPolicy, apply_microcompact
from aura.application.compact.prompt import SUMMARY_SYSTEM, SUMMARY_USER_PREFIX
from aura.application.loop_state import LoopState
from aura.application.memory.context import Context
from aura.config.schema import AuraConfig
from aura.domain.skill import Skill
from aura.domain.tokens import estimate_text_tokens
from aura.infrastructure.persistence.storage import SessionStorage


class _TaskProgress(Protocol):
    @property
    def last_activity_at(self) -> float | None: ...


class _TaskRecord(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def status(self) -> str: ...
    @property
    def observed_at(self) -> float | None: ...
    @property
    def started_at(self) -> float: ...
    @property
    def description(self) -> str: ...
    @property
    def progress(self) -> _TaskProgress: ...


class _TasksStore(Protocol):
    def list(self) -> Iterable[_TaskRecord]: ...


class _CompactSession(Protocol):
    """Narrow contract reactive.py needs from AgentSession."""

    @property
    def tasks_store(self) -> _TasksStore: ...
    @property
    def config(self) -> AuraConfig: ...
    @property
    def microcompact_policy(self) -> MicrocompactPolicy | None: ...
    @property
    def context_window(self) -> int: ...
    @property
    def state(self) -> LoopState: ...
    @property
    def storage(self) -> SessionStorage: ...
    @property
    def session_id(self) -> str: ...
    @property
    def model(self) -> BaseChatModel: ...
    @property
    def context(self) -> Context: ...
    @property
    def cwd(self) -> Path: ...

    def reload_memory_and_rules(self) -> None: ...
    def apply_compaction(
        self,
        *,
        new_history: list[BaseMessage],
        new_context: Context,
        preserved_skills: list[Skill],
    ) -> None: ...


@dataclass(frozen=True)
class SummaryCaps:
    max_summary_message_chars: int = 6_000
    max_summary_tool_args_chars: int = 2_000
    fallback_summary_char_limit: int = 12_000
    max_summary_split_depth: int = 12


_DEFAULT_SUMMARY_CAPS = SummaryCaps()


def summary_caps_from_agent(agent: _CompactSession) -> SummaryCaps:
    cfg = agent.config.compact
    return SummaryCaps(
        max_summary_message_chars=cfg.max_summary_message_chars,
        max_summary_tool_args_chars=cfg.max_summary_tool_args_chars,
        fallback_summary_char_limit=cfg.fallback_summary_char_limit,
        max_summary_split_depth=cfg.max_summary_split_depth,
    )


def _cap_summary_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n... (truncated; {omitted} chars omitted)"


def _serialize_tool_args(args: dict[str, Any], *, caps: SummaryCaps) -> str:
    try:
        rendered = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = str(args)
    return _cap_summary_text(rendered, max_chars=caps.max_summary_tool_args_chars)


def _serialize_history(
    messages: list[BaseMessage],
    *,
    caps: SummaryCaps = _DEFAULT_SUMMARY_CAPS,
) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.__class__.__name__.replace("Message", "").lower()
        content = str(m.content) if m.content else ""
        content = _cap_summary_text(content, max_chars=caps.max_summary_message_chars)
        lines.append(f"[{role}] {content}")
        tool_calls = m.tool_calls if isinstance(m, AIMessage) else []
        for tc in tool_calls:
            lines.append(
                "    -> tool_call "
                f"{tc.get('name')!r} args={_serialize_tool_args(tc.get('args'), caps=caps)}"
            )
    return "\n".join(lines)


def _summary_turn_estimated_tokens(
    messages: list[BaseMessage],
    *,
    caps: SummaryCaps = _DEFAULT_SUMMARY_CAPS,
) -> int:
    return estimate_text_tokens(
        SUMMARY_SYSTEM
        + "\n"
        + SUMMARY_USER_PREFIX
        + _serialize_history(messages, caps=caps)
    )


def estimate_compact_summary_tokens(messages: list[BaseMessage]) -> int:
    return _summary_turn_estimated_tokens(messages)


def compact_summary_messages(
    agent: _CompactSession, history: list[BaseMessage],
) -> list[BaseMessage]:
    """View the summary turn sees; stored history stays raw."""
    policy = agent.microcompact_policy
    if policy is None:
        return list(history)
    return apply_microcompact(list(history), policy).messages


def _split_for_summary_budget(
    messages: list[BaseMessage],
    *,
    max_prompt_tokens: int,
    caps: SummaryCaps = _DEFAULT_SUMMARY_CAPS,
) -> list[list[BaseMessage]]:
    chunks: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for message in messages:
        candidate = [*current, message]
        if (
            current
            and _summary_turn_estimated_tokens(candidate, caps=caps) > max_prompt_tokens
        ):
            chunks.append(current)
            current = [message]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


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
    "1261",  # DashScope
)


def _is_prompt_too_long(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(phrase in msg for phrase in _PTL_PHRASES):
        return True
    if type(exc).__name__.lower() in {"prompttoolongerror", "prompttoolong"}:
        return True
    return any(
        f"'code': '{code}'" in msg or f'"code": "{code}"' in msg
        for code in _PTL_CODES
    )


_MAX_COMPACT_SUMMARY_PROMPT_TOKENS = 16_000


def compact_summary_prompt_budget(agent: _CompactSession) -> int:
    window = agent.context_window
    if window <= 0:
        return _MAX_COMPACT_SUMMARY_PROMPT_TOKENS
    return max(2_000, min(window - 13_000, _MAX_COMPACT_SUMMARY_PROMPT_TOKENS))


async def _run_summary_turn_with_retry(
    model: BaseChatModel,
    to_summarize: list[BaseMessage],
    *,
    max_prompt_tokens: int = _MAX_COMPACT_SUMMARY_PROMPT_TOKENS,
    caps: SummaryCaps = _DEFAULT_SUMMARY_CAPS,
) -> str:
    """Summarize; split recursively when the provider rejects prompt size."""
    chunks = _split_for_summary_budget(
        list(to_summarize),
        max_prompt_tokens=max_prompt_tokens,
        caps=caps,
    )
    if not chunks:
        return ""
    if len(chunks) == 1:
        return await _run_summary_turn_resilient(model, chunks[0], depth=0, caps=caps)

    partials: list[BaseMessage] = []
    for idx, chunk in enumerate(chunks, start=1):
        text = await _run_summary_turn_resilient(model, chunk, depth=0, caps=caps)
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
        caps=caps,
    )


async def _run_summary_turn_resilient(
    model: BaseChatModel,
    messages: list[BaseMessage],
    *,
    depth: int,
    caps: SummaryCaps = _DEFAULT_SUMMARY_CAPS,
) -> str:
    if not messages:
        return ""
    try:
        return await _run_summary_turn(model, messages, caps=caps)
    except Exception as exc:  # noqa: BLE001
        if not _is_prompt_too_long(exc):
            raise
        if len(messages) == 1 or depth >= caps.max_summary_split_depth:
            return _fallback_summary(messages, caps=caps)

    midpoint = max(1, len(messages) // 2)
    left = await _run_summary_turn_resilient(
        model, messages[:midpoint], depth=depth + 1, caps=caps,
    )
    right = await _run_summary_turn_resilient(
        model, messages[midpoint:], depth=depth + 1, caps=caps,
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
    return await _run_summary_turn_resilient(model, merged, depth=depth + 1, caps=caps)


def _fallback_summary(
    messages: list[BaseMessage],
    *,
    caps: SummaryCaps = _DEFAULT_SUMMARY_CAPS,
) -> str:
    """Deterministic last resort when even a single message is too large."""
    serialized = _serialize_history(messages, caps=caps)
    if len(serialized) > caps.fallback_summary_char_limit:
        serialized = (
            serialized[: caps.fallback_summary_char_limit] + "\n... (truncated)"
        )
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
    model: BaseChatModel,
    to_summarize: list[BaseMessage],
    *,
    caps: SummaryCaps = _DEFAULT_SUMMARY_CAPS,
) -> str:
    serialized = _serialize_history(to_summarize, caps=caps)
    messages: list[BaseMessage] = [
        SystemMessage(content=SUMMARY_SYSTEM),
        HumanMessage(content=SUMMARY_USER_PREFIX + serialized),
    ]
    ai = await model.ainvoke(messages)
    return str(ai.content) if ai.content else ""
