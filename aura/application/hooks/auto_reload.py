"""Default V14-HOOK-CATALOG consumers — live-reload project memory + rules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aura.application.hooks import CwdChangedHook, FileChangedHook
from aura.infrastructure.persistence import journal
from aura.schemas.state import LoopState

if TYPE_CHECKING:
    from aura.core.agent import Agent


_AURA_MD_NAMES = {"AURA.md", "AURA.local.md"}


def _is_aura_md_path(path: Path) -> bool:
    return path.name in _AURA_MD_NAMES


def make_aura_md_reload_hook(agent: Agent) -> FileChangedHook:
    """FileChangedHook that rebuilds project memory on AURA.md changes."""

    async def _hook(
        *,
        path: Path,
        kind: str,
        state: LoopState,
        **_: Any,
    ) -> None:
        if not _is_aura_md_path(path):
            return
        agent.apply_aura_md_reload()
        journal.write(
            "aura_md_reloaded",
            session=agent._session_id,
            path=str(path),
            kind=kind,
        )

    return _hook


def make_cwd_rules_reload_hook(agent: Agent) -> CwdChangedHook:
    """CwdChangedHook that refreshes rules + project memory for new cwd."""

    async def _hook(
        *,
        old_cwd: Path,
        new_cwd: Path,
        state: LoopState,
        **_: Any,
    ) -> None:
        agent.change_cwd_and_reload(new_cwd)
        journal.write(
            "cwd_rules_reloaded",
            session=agent._session_id,
            old_cwd=str(old_cwd),
            new_cwd=str(new_cwd),
        )

    return _hook
