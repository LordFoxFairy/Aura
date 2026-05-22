"""Default V14-HOOK-CATALOG consumers — live-reload project memory + rules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aura.application.hooks import CwdChangedHook, FileChangedHook
from aura.application.memory import project_memory, rules
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
        project_memory.clear_cache(agent._cwd)
        agent._primary_memory = project_memory.load_project_memory(agent._cwd)
        agent._context = agent._build_context()
        agent._hooks.pre_tool.remove(agent._must_read_first_hook)
        from aura.application.hooks.must_read_first import make_must_read_first_hook
        agent._must_read_first_hook = make_must_read_first_hook(agent._context)
        agent._hooks.pre_tool.append(agent._must_read_first_hook)
        agent._loop = agent._build_loop()
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
        agent._cwd = new_cwd
        project_memory.clear_cache(new_cwd)
        rules.clear_cache(new_cwd)
        agent._primary_memory = project_memory.load_project_memory(new_cwd)
        agent._rules = rules.load_rules(new_cwd)
        agent._context = agent._build_context()
        agent._hooks.pre_tool.remove(agent._must_read_first_hook)
        from aura.application.hooks.must_read_first import make_must_read_first_hook
        agent._must_read_first_hook = make_must_read_first_hook(agent._context)
        agent._hooks.pre_tool.append(agent._must_read_first_hook)
        agent._loop = agent._build_loop()
        journal.write(
            "cwd_rules_reloaded",
            session=agent._session_id,
            old_cwd=str(old_cwd),
            new_cwd=str(new_cwd),
        )

    return _hook
