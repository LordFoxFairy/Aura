"""Default consumers for the V14-HOOK-CATALOG hooks.

- :func:`make_aura_md_reload_hook` — on ``FileChanged`` for ``AURA.md``,
  invalidate the project_memory cache and refresh
  :attr:`Agent._primary_memory` + :attr:`Agent._context`. Live-reload
  of project memory without ``/clear``.
- :func:`make_cwd_rules_reload_hook` — on ``CwdChanged``, invalidate
  the rules + project_memory caches and refresh both for the new cwd.

Both are consumed by ``default_hooks()`` so every Aura agent ships
them out of the box (per the V14-HOOK-CATALOG quality bar: no
SDK-only scaffolding).

The factories take an :class:`Agent` directly so the closure can reach
into ``_primary_memory`` / ``_rules`` / ``_context`` / ``_cwd``. Yes,
that's tighter coupling than a pure function, but the alternative is
threading a half-dozen mutators through the hook signature for a
single load-bearing consumer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aura.core.hooks import CwdChangedHook, FileChangedHook
from aura.core.memory import project_memory, rules
from aura.core.persistence import journal
from aura.schemas.state import LoopState

if TYPE_CHECKING:
    from aura.core.agent import Agent


_AURA_MD_NAMES = {"AURA.md", "AURA.local.md"}


def _is_aura_md_path(path: Path) -> bool:
    """True iff ``path`` is one of the AURA memory files we live-reload.

    ``.aura/AURA.md`` (the nested form) qualifies too. We match on
    filename rather than full path so per-ancestor variants are all
    covered without enumerating ancestors here.
    """
    return path.name in _AURA_MD_NAMES


def make_aura_md_reload_hook(agent: Agent) -> FileChangedHook:
    """Build a FileChangedHook that live-reloads project memory.

    Closes over ``agent`` so the hook can mutate the live
    ``_primary_memory`` / ``_context`` fields directly. Non-AURA-md
    paths are ignored — the watcher fans every event out, this hook
    decides whether to act.
    """

    async def _hook(
        *,
        path: Path,
        kind: str,
        state: LoopState,
        **_: Any,
    ) -> None:
        if not _is_aura_md_path(path):
            return
        # Drop the cache for the agent's resolved cwd; ``load_project_memory``
        # will rescan walk-up + nested + local, exactly mirroring the
        # construction-time path.
        project_memory.clear_cache(agent._cwd)
        agent._primary_memory = project_memory.load_project_memory(agent._cwd)
        # Rebuild context so the system prompt's project memory section
        # picks up the new content. Inherited reads stay where they
        # are — this is a memory refresh, not a /clear.
        agent._context = agent._build_context()
        # Swap must_read_first hook to close over the rebuilt Context
        # (the old one's _read_records is now decoupled from the live
        # Agent state; same dance as ``clear_session``).
        agent._hooks.pre_tool.remove(agent._must_read_first_hook)
        from aura.core.hooks.must_read_first import make_must_read_first_hook
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
    """Build a CwdChangedHook that refreshes rules + project memory.

    On a cwd change the previously-cached project memory + rules are
    bound to the OLD path; they're not just stale, they're wrong for
    the new project. Drop the caches, reload from ``new_cwd``, and
    rebuild Context so the next turn sees the right surface.
    """

    async def _hook(
        *,
        old_cwd: Path,
        new_cwd: Path,
        state: LoopState,
        **_: Any,
    ) -> None:
        # ``Agent.set_cwd`` already updated ``agent._cwd``, but be
        # defensive — if a custom caller fired the hook without going
        # through ``set_cwd``, prefer the kwarg.
        agent._cwd = new_cwd
        project_memory.clear_cache(new_cwd)
        rules.clear_cache(new_cwd)
        agent._primary_memory = project_memory.load_project_memory(new_cwd)
        agent._rules = rules.load_rules(new_cwd)
        agent._context = agent._build_context()
        # Keep must_read_first attached to the live Context.
        agent._hooks.pre_tool.remove(agent._must_read_first_hook)
        from aura.core.hooks.must_read_first import make_must_read_first_hook
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
