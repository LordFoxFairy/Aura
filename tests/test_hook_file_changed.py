"""Tests for the FileChanged hook + FileWatcher producer + AURA.md reload consumer.

Covers (V14-HOOK-CATALOG):

1. Watcher fires hook on file modification.
2. Watcher fires ``kind="created"`` for newly-appearing watched files.
3. Watcher fires ``kind="deleted"`` for removed files.
4. AURA.md reload consumer mutates Context.primary_memory after a write.
5. ``stop()`` cancels the polling task without warnings.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from aura.core.hooks import HookChain
from aura.core.hooks.auto_reload import make_aura_md_reload_hook
from aura.core.hooks.file_watcher import FileWatcher
from aura.schemas.state import LoopState

# Use a tight polling interval so the tests don't drag — the watcher's
# default interval is human-scale (1.0s), but unit tests should not be.
_FAST_POLL = 0.05


async def _wait_for(predicate: Any, timeout: float = 2.0) -> None:
    """Poll ``predicate`` until truthy or raise TimeoutError after ``timeout``."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise TimeoutError(f"predicate {predicate!r} stayed false within {timeout}s")


@pytest.mark.asyncio
async def test_watcher_fires_modified_when_file_changes(tmp_path: Path) -> None:
    target = tmp_path / "AURA.md"
    target.write_text("v1", encoding="utf-8")

    captured: list[tuple[Path, str]] = []

    async def hook(
        *, path: Path, kind: str, state: LoopState, **_: Any,
    ) -> None:
        captured.append((path, kind))

    chain = HookChain(file_changed=[hook])
    state = LoopState()

    watcher = FileWatcher(
        paths=[target],
        chain=chain,
        state=state,
        poll_interval=_FAST_POLL,
    )
    await watcher.start()
    try:
        # Sleep a few polling intervals so the watcher latches the
        # initial mtime, then write fresh content with an mtime bump.
        await asyncio.sleep(_FAST_POLL * 3)
        # Ensure the second mtime differs from the first — some
        # filesystems have coarse mtime granularity.
        new_mtime = target.stat().st_mtime + 1
        target.write_text("v2", encoding="utf-8")
        import os
        os.utime(target, (new_mtime, new_mtime))

        await _wait_for(
            lambda: any(kind == "modified" for _, kind in captured)
        )
    finally:
        await watcher.stop()

    kinds = [k for _, k in captured]
    assert "modified" in kinds


@pytest.mark.asyncio
async def test_watcher_fires_created_when_file_appears(tmp_path: Path) -> None:
    target = tmp_path / "AURA.md"
    # ``target`` does NOT exist when the watcher starts.

    captured: list[tuple[Path, str]] = []

    async def hook(
        *, path: Path, kind: str, state: LoopState, **_: Any,
    ) -> None:
        captured.append((path, kind))

    chain = HookChain(file_changed=[hook])
    state = LoopState()

    watcher = FileWatcher(
        paths=[target],
        chain=chain,
        state=state,
        poll_interval=_FAST_POLL,
    )
    await watcher.start()
    try:
        await asyncio.sleep(_FAST_POLL * 3)
        target.write_text("hello", encoding="utf-8")
        await _wait_for(lambda: any(k == "created" for _, k in captured))
    finally:
        await watcher.stop()

    assert any(k == "created" for _, k in captured)


@pytest.mark.asyncio
async def test_watcher_fires_deleted_when_file_removed(tmp_path: Path) -> None:
    target = tmp_path / "AURA.md"
    target.write_text("hello", encoding="utf-8")

    captured: list[tuple[Path, str]] = []

    async def hook(
        *, path: Path, kind: str, state: LoopState, **_: Any,
    ) -> None:
        captured.append((path, kind))

    chain = HookChain(file_changed=[hook])
    state = LoopState()

    watcher = FileWatcher(
        paths=[target],
        chain=chain,
        state=state,
        poll_interval=_FAST_POLL,
    )
    await watcher.start()
    try:
        await asyncio.sleep(_FAST_POLL * 3)
        target.unlink()
        await _wait_for(lambda: any(k == "deleted" for _, k in captured))
    finally:
        await watcher.stop()

    assert any(k == "deleted" for _, k in captured)


@pytest.mark.asyncio
async def test_aura_md_reload_consumer_refreshes_primary_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: AURA.md change → reload consumer → Context refresh.

    The Agent is built first with one AURA.md content, then the test
    rewrites the file and invokes the reload consumer directly (no
    watcher in the loop — that's covered by the producer tests above).
    The consumer must clear the project_memory cache and re-load the
    primary memory string on the Agent's Context.
    """
    from aura.config.schema import AuraConfig
    from aura.core.agent import Agent
    from aura.core.persistence.storage import SessionStorage

    monkeypatch.chdir(tmp_path)
    fake_home = tmp_path / "_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    from aura.core.memory import project_memory, rules
    project_memory.clear_cache()
    rules.clear_cache()

    aura_md = tmp_path / "AURA.md"
    aura_md.write_text("ORIGINAL", encoding="utf-8")

    cfg = AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
            "storage": {"path": str(tmp_path / "db")},
        }
    )

    from tests.conftest import FakeChatModel

    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "aura.db"),
    )
    try:
        assert "ORIGINAL" in agent._primary_memory

        aura_md.write_text("UPDATED", encoding="utf-8")
        # Bump mtime so any cache busting that uses mtime will see the change.
        import os
        new_mtime = aura_md.stat().st_mtime + 1
        os.utime(aura_md, (new_mtime, new_mtime))

        consumer = make_aura_md_reload_hook(agent)
        await consumer(
            path=aura_md, kind="modified", state=agent.state,
        )

        assert "UPDATED" in agent._primary_memory
        assert "ORIGINAL" not in agent._primary_memory
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_watcher_stop_cancels_polling_task_without_warnings(
    tmp_path: Path, recwarn: pytest.WarningsRecorder,
) -> None:
    target = tmp_path / "AURA.md"
    target.write_text("v1", encoding="utf-8")

    chain = HookChain()
    state = LoopState()

    watcher = FileWatcher(
        paths=[target], chain=chain, state=state, poll_interval=_FAST_POLL,
    )
    await watcher.start()
    # Let the polling loop tick at least once.
    await asyncio.sleep(_FAST_POLL * 2)
    await watcher.stop()

    # Idempotent stop.
    await watcher.stop()

    # No "Task was destroyed but it is pending" / "coroutine was never
    # awaited" warnings should have surfaced.
    bad = [
        w for w in recwarn.list
        if "was destroyed" in str(w.message)
        or "was never awaited" in str(w.message)
    ]
    assert not bad, f"unexpected warnings: {bad}"
