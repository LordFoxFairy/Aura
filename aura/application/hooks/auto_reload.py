"""Live-reload project memory + rules on AURA.md / cwd changes."""

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

    async def _hook(
        *,
        path: Path,
        kind: str,
        state: LoopState,  # noqa: ARG001  # Protocol kw arg; unused by this hook
        **_: Any,
    ) -> None:
        if not _is_aura_md_path(path):
            return
        agent.apply_aura_md_reload()
        journal.write(
            "aura_md_reloaded",
            session=agent.session_id,
            path=str(path),
            kind=kind,
        )

    return _hook


def make_cwd_rules_reload_hook(agent: Agent) -> CwdChangedHook:

    async def _hook(
        *,
        old_cwd: Path,
        new_cwd: Path,
        state: LoopState,  # noqa: ARG001  # Protocol kw arg; unused by this hook
        **_: Any,
    ) -> None:
        agent.change_cwd_and_reload(new_cwd)
        journal.write(
            "cwd_rules_reloaded",
            session=agent.session_id,
            old_cwd=str(old_cwd),
            new_cwd=str(new_cwd),
        )

    return _hook
