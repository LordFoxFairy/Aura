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

from aura.application.hooks import HookChain
from aura.application.hooks.auto_reload import make_aura_md_reload_hook
from aura.application.hooks.file_watcher import (
    FileWatcher,
    _Snap,
    default_watch_paths,
)
from aura.application.hooks.protocols import FileChangeKind
from aura.application.loop_state import LoopState

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
        *,
        path: Path,
        kind: str,
        state: LoopState,
        **_: Any,
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

        await _wait_for(lambda: any(kind == "modified" for _, kind in captured))
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
        *,
        path: Path,
        kind: str,
        state: LoopState,
        **_: Any,
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
        *,
        path: Path,
        kind: str,
        state: LoopState,
        **_: Any,
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: AURA.md change → reload consumer → Context refresh.

    The AgentSession is built first with one AURA.md content, then the test
    rewrites the file and invokes the reload consumer directly (no
    watcher in the loop — that's covered by the producer tests above).
    The consumer must clear the project_memory cache and re-load the
    primary memory string on the AgentSession's Context.
    """
    from aura.application.session import AgentSession
    from aura.config.schema import AuraConfig
    from aura.infrastructure.persistence.storage import SessionStorage

    monkeypatch.chdir(tmp_path)
    fake_home = tmp_path / "_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    from aura.application.memory import project_memory, rules

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

    agent = AgentSession(
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
            path=aura_md,
            kind="modified",
            state=agent.state,
        )

        assert "UPDATED" in agent._primary_memory
        assert "ORIGINAL" not in agent._primary_memory
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_watcher_stop_cancels_polling_task_without_warnings(
    tmp_path: Path,
    recwarn: pytest.WarningsRecorder,
) -> None:
    target = tmp_path / "AURA.md"
    target.write_text("v1", encoding="utf-8")

    chain = HookChain()
    state = LoopState()

    watcher = FileWatcher(
        paths=[target],
        chain=chain,
        state=state,
        poll_interval=_FAST_POLL,
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
        w
        for w in recwarn.list
        if "was destroyed" in str(w.message) or "was never awaited" in str(w.message)
    ]
    assert not bad, f"unexpected warnings: {bad}"


# Below: drive _tick/_take_snapshot/_poll_loop directly for deterministic branch coverage.


def _make_watcher(
    paths: list[Path],
    chain: HookChain,
    *,
    poll_interval: float = _FAST_POLL,
) -> FileWatcher:
    """Builder so the boundary tests share one wiring point for FileWatcher."""
    return FileWatcher(
        paths=paths,
        chain=chain,
        state=LoopState(),
        poll_interval=poll_interval,
    )


def _capture_chain() -> tuple[HookChain, list[tuple[Path, str]]]:
    """A HookChain whose single file_changed hook records (path, kind) tuples."""
    captured: list[tuple[Path, str]] = []

    async def hook(*, path: Path, kind: str, state: LoopState, **_: Any) -> None:
        captured.append((path, kind))

    return HookChain(file_changed=[hook]), captured


def _spy_journal(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, Any]]]:
    """Replace journal.write with an in-memory recorder of (event, fields)."""
    events: list[tuple[str, dict[str, Any]]] = []

    def fake_write(event: str, **fields: Any) -> None:
        events.append((event, dict(fields)))

    # Patch journal.write at the source module file_watcher calls through.
    monkeypatch.setattr("aura.infrastructure.persistence.journal.write", fake_write)
    return events


async def test_start_is_idempotent_while_running(tmp_path: Path) -> None:
    """A double start must not spawn a second poll task — duplicate watchers
    would double-fire every hook and leak a task on stop."""
    target = tmp_path / "AURA.md"
    target.write_text("v1", encoding="utf-8")
    chain, _ = _capture_chain()
    watcher = _make_watcher([target], chain)

    await watcher.start()
    first_task = watcher._task
    await watcher.start()  # early-return: task is not None and not done
    try:
        assert watcher._task is first_task
    finally:
        await watcher.stop()


async def test_stop_before_start_is_a_noop() -> None:
    """Stopping a never-started watcher must not raise — shutdown paths call
    stop() unconditionally and a None task is the normal pre-start state."""
    chain, _ = _capture_chain()
    watcher = _make_watcher([Path("/nonexistent")], chain)
    # No task exists; stop() must take the ``task is None`` guard and return.
    await watcher.stop()
    assert watcher._task is None


async def test_start_then_stop_journals_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start/stop must emit the audit trail the operator relies on to know the
    watcher is live; a silent watcher is indistinguishable from a dead one."""
    events = _spy_journal(monkeypatch)
    target = tmp_path / "AURA.md"
    target.write_text("v1", encoding="utf-8")
    chain, _ = _capture_chain()
    watcher = _make_watcher([target], chain)

    await watcher.start()
    await watcher.stop()

    names = [name for name, _ in events]
    assert "file_watcher_started" in names
    assert "file_watcher_stopped" in names
    started = next(f for n, f in events if n == "file_watcher_started")
    assert started["poll_interval"] == _FAST_POLL
    assert started["paths"] == [str(watcher._roots[0])]


@pytest.mark.parametrize(
    ("prev_mtime", "new_mtime", "expected"),
    [
        (None, 1.0, ["created"]),  # absent -> present
        (None, None, []),  # absent -> still absent: no event
        (1.0, None, ["deleted"]),  # present -> vanished (same key)
        (1.0, 2.0, ["modified"]),  # mtime advanced
        (2.0, 1.0, ["modified"]),  # mtime moved backwards still differs
        (1.0, 1.0, []),  # unchanged: idempotent, no event
        (0.0, 0.0, []),  # zero mtime, unchanged
        (-1.0, 1.0, ["modified"]),  # negative epoch boundary
        (1.0, float("inf"), ["modified"]),  # huge mtime
    ],
)
async def test_tick_state_machine_matrix(
    monkeypatch: pytest.MonkeyPatch,
    prev_mtime: float | None,
    new_mtime: float | None,
    expected: list[str],
) -> None:
    """The created/modified/deleted decision is the watcher's whole contract;
    each transition must map to exactly one (or zero) hook event."""
    p = Path("/seed/file")
    chain, captured = _capture_chain()
    watcher = _make_watcher([p], chain)
    watcher._snapshots = {p: _Snap(mtime=prev_mtime)}

    monkeypatch.setattr(
        watcher,
        "_take_snapshot",
        lambda: {p: _Snap(mtime=new_mtime)},
    )
    await watcher._tick()

    assert [k for _, k in captured] == expected
    # Post-tick snapshot must reflect the new world so the next tick is stable.
    assert watcher._snapshots[p].mtime == new_mtime


async def test_tick_nan_mtime_is_always_modified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NaN never equals itself, so a NaN mtime read would forever re-fire
    'modified'; this documents that real-world non-comparable mtimes leak
    events (a known sharp edge of the != comparison, not a guard)."""
    p = Path("/seed/file")
    chain, captured = _capture_chain()
    watcher = _make_watcher([p], chain)
    nan = float("nan")
    watcher._snapshots = {p: _Snap(mtime=nan)}
    monkeypatch.setattr(
        watcher,
        "_take_snapshot",
        lambda: {p: _Snap(mtime=nan)},
    )
    await watcher._tick()
    assert [k for _, k in captured] == ["modified"]


async def test_tick_deleted_when_path_drops_out_of_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path present last tick but absent from the new snapshot dict must
    still emit 'deleted'; otherwise a removed watched root is silently lost."""
    gone = Path("/was/here")
    stays = Path("/still/here")
    chain, captured = _capture_chain()
    watcher = _make_watcher([stays], chain)
    watcher._snapshots = {
        gone: _Snap(mtime=5.0),
        stays: _Snap(mtime=5.0),
    }
    # New snapshot omits ``gone`` entirely (the disappeared-key branch).
    monkeypatch.setattr(
        watcher,
        "_take_snapshot",
        lambda: {stays: _Snap(mtime=5.0)},
    )
    await watcher._tick()
    assert (gone, "deleted") in captured
    assert (stays, "modified") not in captured


async def test_tick_deleted_skipped_when_prev_already_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path that was already a 'missing' tombstone (mtime=None) and drops out
    must NOT re-fire 'deleted' — deletion is a one-shot edge, not a level."""
    gone = Path("/already/gone")
    chain, captured = _capture_chain()
    watcher = _make_watcher([gone], chain)
    watcher._snapshots = {gone: _Snap(mtime=None)}
    monkeypatch.setattr(watcher, "_take_snapshot", dict)
    await watcher._tick()
    assert captured == []


async def test_tick_swallows_consumer_error_and_journals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A throwing hook must never break the poll loop; the watcher logs the
    failure and keeps watching so one bad consumer can't blind the others."""
    events = _spy_journal(monkeypatch)
    p = Path("/seed/file")

    async def boom(*, path: Path, kind: str, state: LoopState, **_: Any) -> None:
        raise RuntimeError("consumer exploded")

    chain = HookChain(file_changed=[boom])
    watcher = _make_watcher([p], chain)
    watcher._snapshots = {p: _Snap(mtime=None)}
    monkeypatch.setattr(
        watcher,
        "_take_snapshot",
        lambda: {p: _Snap(mtime=1.0)},
    )
    # Must not raise despite the hook raising.
    await watcher._tick()

    consumer_errs = [f for n, f in events if n == "file_watcher_consumer_error"]
    assert len(consumer_errs) == 1
    assert consumer_errs[0]["kind"] == "created"
    assert "RuntimeError" in consumer_errs[0]["error"]


async def test_tick_called_twice_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-ticking an unchanged tree must yield no new events — the watcher is
    edge-triggered, so a stable filesystem must stay quiet (no event storm)."""
    p = Path("/seed/file")
    chain, captured = _capture_chain()
    watcher = _make_watcher([p], chain)
    watcher._snapshots = {p: _Snap(mtime=None)}
    snap = {p: _Snap(mtime=7.0)}
    monkeypatch.setattr(watcher, "_take_snapshot", lambda: snap)

    await watcher._tick()  # None -> 7.0 == created
    await watcher._tick()  # 7.0 -> 7.0 == no change
    assert [k for _, k in captured] == ["created"]


async def test_take_snapshot_walks_directory_tree(tmp_path: Path) -> None:
    """Watching a directory must snapshot every nested file recursively; AURA
    skills live in a tree, so a shallow scan would miss sub-skill edits."""
    root = tmp_path / "skills"
    (root / "nested").mkdir(parents=True)
    leaf = root / "nested" / "skill.md"
    leaf.write_text("x", encoding="utf-8")
    top = root / "top.md"
    top.write_text("y", encoding="utf-8")

    chain, _ = _capture_chain()
    watcher = _make_watcher([root], chain)
    snap = watcher._take_snapshot()

    assert leaf.resolve() in snap
    assert top.resolve() in snap
    # The directory root itself is also snapshotted (its own mtime).
    assert root.resolve() in snap
    assert snap[leaf.resolve()].mtime is not None


async def test_take_snapshot_missing_root_is_tombstone(tmp_path: Path) -> None:
    """A non-existent watched path must snapshot as mtime=None, not be dropped,
    so its later appearance is detectable as 'created'."""
    missing = tmp_path / "not-there"
    chain, _ = _capture_chain()
    watcher = _make_watcher([missing], chain)
    snap = watcher._take_snapshot()
    assert snap[missing.resolve()] == _Snap(mtime=None)


async def test_take_snapshot_skips_child_on_stat_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child whose stat() races a delete is skipped while its siblings still
    snapshot — TOCTOU on a busy tree must degrade per-file, not abort the scan."""
    root = tmp_path / "dir"
    root.mkdir()
    flaky = root / "flaky.md"
    flaky.write_text("x", encoding="utf-8")
    stable = root / "stable.md"
    stable.write_text("y", encoding="utf-8")

    chain, _ = _capture_chain()
    watcher = _make_watcher([root], chain)
    flaky_resolved = flaky.resolve()
    stable_resolved = stable.resolve()

    real_stat = Path.stat

    def flaky_stat(self: Path, *a: Any, **k: Any) -> Any:
        if self.name == flaky.name:
            raise OSError("vanished mid-scan")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    snap = watcher._take_snapshot()
    assert flaky_resolved not in snap
    assert stable_resolved in snap


async def test_take_snapshot_skips_root_on_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OSError reading a whole root (e.g. permission denied) must skip that
    root and continue scanning the rest — one bad mount can't kill the watcher."""
    good = tmp_path / "good.md"
    good.write_text("ok", encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text("no", encoding="utf-8")

    chain, _ = _capture_chain()
    watcher = _make_watcher([bad, good], chain)

    real_is_dir = Path.is_dir

    def flaky_is_dir(self: Path) -> bool:
        if self.resolve() == bad.resolve():
            raise OSError("permission denied")
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", flaky_is_dir)
    snap = watcher._take_snapshot()
    assert bad.resolve() not in snap
    assert good.resolve() in snap


async def test_poll_loop_reraises_and_journals_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected error inside a tick must be journaled AND re-raised, not
    swallowed — a watcher silently dead after a bug is worse than a crash."""
    events = _spy_journal(monkeypatch)
    chain, _ = _capture_chain()
    watcher = _make_watcher([Path("/x")], chain, poll_interval=0.0)

    async def explode() -> None:
        raise ValueError("disk on fire")

    monkeypatch.setattr(watcher, "_tick", explode)
    with pytest.raises(ValueError, match="disk on fire"):
        await watcher._poll_loop()

    errs = [f for n, f in events if n == "file_watcher_error"]
    assert len(errs) == 1
    assert "ValueError" in errs[0]["error"]


async def test_poll_loop_propagates_cancellation_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation must propagate unchanged (not be logged as an error), so
    stop() can tear the loop down without polluting the audit trail."""
    events = _spy_journal(monkeypatch)
    chain, _ = _capture_chain()
    watcher = _make_watcher([Path("/x")], chain, poll_interval=0.01)

    task = asyncio.create_task(watcher._poll_loop())
    await asyncio.sleep(0)  # let the loop reach its first await
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # CancelledError must NOT be recorded as file_watcher_error.
    assert not [n for n, _ in events if n == "file_watcher_error"]


def test_default_watch_paths_returns_expected_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The default watch set must cover the three reload-relevant locations:
    global AURA.md, the skills tree, and the project-local .aura dir."""
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    cwd = tmp_path / "proj"

    paths = default_watch_paths(cwd)

    assert paths == [
        fake_home / ".aura" / "AURA.md",
        fake_home / ".aura" / "skills",
        cwd / ".aura",
    ]


def test_watcher_resolves_and_expands_input_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Constructor must normalise paths (expanduser + resolve) so '~' and
    relative inputs key the snapshot dict consistently with stat() results."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    chain, _ = _capture_chain()

    watcher = _make_watcher([Path("~/notes.md")], chain)
    assert watcher._roots == [(fake_home / "notes.md").resolve()]


@pytest.mark.parametrize("kind", ["created", "modified", "deleted"])
async def test_run_file_changed_forwards_every_kind(kind: FileChangeKind) -> None:
    """HookChain.run_file_changed must pass each FileChangeKind through verbatim
    so consumers can branch on creation vs modification vs deletion."""
    chain, captured = _capture_chain()
    p = Path("/some/file")
    await chain.run_file_changed(path=p, kind=kind, state=LoopState())
    assert captured == [(p, kind)]
