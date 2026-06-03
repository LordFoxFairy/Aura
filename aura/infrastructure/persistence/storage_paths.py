"""Path computation for session/subagent/team storage layout + cwd encoding."""

from __future__ import annotations

import os
from pathlib import Path


def encode_cwd(cwd: Path) -> str:
    """Encode an absolute cwd into the ``projects/`` bucket name."""
    abs_cwd = cwd if cwd.is_absolute() else (Path.cwd() / cwd).resolve()
    return str(abs_cwd).replace(os.sep, "-")


def validate_session_id(session_id: str) -> None:
    if not session_id or "/" in session_id or ".." in session_id:
        raise ValueError(f"invalid session_id: {session_id!r}")


def validate_task_id(task_id: str) -> None:
    if not task_id or "/" in task_id or ".." in task_id:
        raise ValueError(f"invalid task_id: {task_id!r}")


def projects_dir(root: Path) -> Path:
    return root / "projects"


def project_dir(root: Path, cwd: Path) -> Path:
    return projects_dir(root) / encode_cwd(cwd)


def session_jsonl_path(root: Path, cwd: Path, session_id: str) -> Path:
    validate_session_id(session_id)
    return project_dir(root, cwd) / f"{session_id}.jsonl"


def memory_dir(root: Path, cwd: Path) -> Path:
    return project_dir(root, cwd) / "memory"


def session_dir(root: Path, cwd: Path, session_id: str) -> Path:
    validate_session_id(session_id)
    return project_dir(root, cwd) / session_id


def subagent_transcript_path(
    root: Path,
    cwd: Path,
    task_id: str,
    parent_session_id: str | None,
) -> Path:
    """``parent_session_id=None`` falls back to the flat ad-hoc bucket."""
    validate_task_id(task_id)
    if parent_session_id is None:
        return root / "subagents" / f"agent-{task_id}.jsonl"
    return session_dir(root, cwd, parent_session_id) / "subagents" / f"agent-{task_id}.jsonl"


def index_path(root: Path) -> Path:
    return root / "index.sqlite"


def teams_root(root: Path) -> Path:
    return root / "teams"


def team_dir(root: Path, team_id: str) -> Path:
    return teams_root(root) / team_id
