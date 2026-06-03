"""Transcript/metadata persistence + child-message capture for the local-agent runner."""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import BaseMessage

from aura.application.tasks.spawn_port import SpawnedAgent
from aura.application.tasks.store import TasksStore
from aura.domain.task import TaskRecord, TaskStatus
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage


class SubagentMeta(TypedDict):
    """Fixed shape of the ``.meta.json`` companion written next to a transcript."""

    agent_type: str
    task_id: str
    description: str
    model_spec: str
    parent_session_id: str
    cwd: str
    started_at: float
    started_at_iso: str
    ended_at: float | None
    ended_at_iso: str | None
    status: TaskStatus
    input_tokens: int
    output_tokens: int


def flush_transcript(
    *,
    transcript_storage: SessionStorage,
    task_id: str,
    messages: list[BaseMessage],
    store: TasksStore,
    parent_session_id: str = "",
    cwd: str = "",
) -> Path | None:
    """Write the child's transcript JSONL via the storage's path API."""
    try:
        cwd_arg: Path | None = Path(cwd) if cwd else None
        parent_arg: str | None = parent_session_id or None
        path = transcript_storage.write_subagent_transcript(
            task_id,
            messages,
            parent_session_id=parent_arg,
            cwd=cwd_arg,
        )
        store.set_transcript_path(task_id, path)
        return path
    except Exception as exc:  # noqa: BLE001  # persistence failure is non-fatal best-effort
        journal.write(
            "subagent_transcript_flush_error",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None


def maybe_cleanup_completed_transcript(
    *,
    agent: SpawnedAgent,
    transcript_storage: SessionStorage,
    task_id: str,
    parent_session_id: str,
    cwd: str,
) -> None:
    """Delete a completed subagent's transcript + meta files (opt-in; failures survive)."""
    try:
        if not agent.config.tools.cleanup_completed_subagent_transcripts:
            return
        cwd_arg: Path | None = Path(cwd) if cwd else None
        parent_arg: str | None = parent_session_id or None
        path_resolvers = (
            ("subagent_transcript_path", transcript_storage.subagent_transcript_path),
            ("subagent_metadata_path", transcript_storage.subagent_metadata_path),
        )
        for fn_name, resolve_path in path_resolvers:
            target = resolve_path(
                task_id, parent_session_id=parent_arg, cwd=cwd_arg,
            )
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                journal.write(
                    "subagent_transcript_cleanup_error",
                    task_id=task_id,
                    file=fn_name,
                    error=f"{type(exc).__name__}: {exc}",
                )
        journal.write(
            "subagent_transcript_cleaned",
            task_id=task_id,
            parent_session_id=parent_arg or "",
        )
    except Exception as exc:  # noqa: BLE001  # log + swallow; logging path must never crash caller
        journal.write(
            "subagent_transcript_cleanup_error",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )


def flush_metadata(
    *,
    transcript_storage: SessionStorage,
    record: TaskRecord,
    parent_session_id: str,
    cwd: str,
) -> Path | None:
    """Write the ``.meta.json`` companion file alongside the transcript."""
    try:
        cwd_arg: Path | None = Path(cwd) if cwd else None
        parent_arg: str | None = parent_session_id or None
        meta_path = transcript_storage.subagent_metadata_path(
            record.id,
            parent_session_id=parent_arg,
            cwd=cwd_arg,
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        started_iso = datetime.fromtimestamp(
            record.started_at, tz=UTC,
        ).isoformat()
        ended_iso: str | None = None
        if record.finished_at is not None:
            ended_iso = datetime.fromtimestamp(
                record.finished_at, tz=UTC,
            ).isoformat()
        meta: SubagentMeta = {
            "agent_type": record.agent_type or "general-purpose",
            "task_id": record.id,
            "description": record.description,
            "model_spec": record.model_spec or "",
            "parent_session_id": parent_session_id or "",
            "cwd": cwd or "",
            "started_at": record.started_at,
            "started_at_iso": started_iso,
            "ended_at": record.finished_at,
            "ended_at_iso": ended_iso,
            "status": record.status,
            "input_tokens": record.progress.input_tokens,
            "output_tokens": record.progress.output_tokens,
        }
        tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp.replace(meta_path)
        return meta_path
    except Exception as exc:  # noqa: BLE001  # persistence failure is non-fatal best-effort
        journal.write(
            "subagent_metadata_flush_error",
            task_id=record.id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None


async def capture_child_messages(
    agent: SpawnedAgent | None, store: TasksStore, task_id: str,
) -> None:
    """Pull the child's full message list onto the TaskRecord."""
    if agent is None:
        return
    try:
        msgs = agent.storage.load(agent.session_id)
        rec = store.get(task_id)
        if rec is None:
            return
        rec.messages = list(msgs)
    except Exception as exc:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
        journal.write(
            "subagent_capture_messages_error",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )


def load_child_messages(
    agent: SpawnedAgent | None, store: TasksStore, task_id: str,
) -> list[BaseMessage]:
    """Read the child's transcript for transcript-flush."""
    rec = store.get(task_id)
    if rec is not None and rec.messages:
        return list(rec.messages)
    if agent is None:
        return []
    with contextlib.suppress(Exception):
        return list(agent.storage.load(agent.session_id))
    return []


__all__ = [
    "SubagentMeta",
    "capture_child_messages",
    "flush_metadata",
    "flush_transcript",
    "load_child_messages",
    "maybe_cleanup_completed_transcript",
]
