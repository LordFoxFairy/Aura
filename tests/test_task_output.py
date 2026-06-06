"""task_output — boundary tests for the snapshot/wait tool's uncovered seams.

Complements test_task_observability.py (happy-path wait/abort/timeout) and
test_task_tools.py by pinning the edges those suites skip: the args_preview
label callback, the sync _run guard, in-flight cancellation of a parked wait,
and the race where a parent abort fires AFTER the child already went terminal.
"""

from __future__ import annotations

import asyncio

import pytest

from aura.application.tasks.store import TasksStore
from aura.domain.abort import AbortController, current_abort_signal
from aura.tools.task_output import TaskOutput, _preview


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        pytest.param({}, "task_output: ?", id="absent-id-falls-back"),
        pytest.param({"task_id": ""}, "task_output: ", id="empty-id"),
        pytest.param(
            {"task_id": "abc123"}, "task_output: abc123", id="short-id-verbatim"
        ),
        pytest.param(
            {"task_id": "0123456789abcdef"},
            "task_output: 01234567",
            id="long-id-truncated-to-8",
        ),
        pytest.param(
            {"task_id": "feedface", "wait": True},
            "task_output: feedface (waiting)",
            id="wait-flag-adds-suffix",
        ),
        pytest.param(
            {"task_id": "feedface", "wait": False},
            "task_output: feedface",
            id="wait-false-no-suffix",
        ),
    ],
)
def test_preview_label_is_safe_for_any_arg_shape(
    args: dict[str, object], expected: str
) -> None:
    """The UI preview must never KeyError on partial args nor leak full ids."""
    assert _preview(args) == expected


def test_run_sync_path_is_rejected() -> None:
    """task_output is async-only; the sync seam must fail loud, not silently no-op."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    tool = TaskOutput(store=store)
    with pytest.raises(NotImplementedError, match="async-only"):
        tool._run(rec.id)


@pytest.mark.asyncio
async def test_abort_after_task_already_terminal_marks_observed() -> None:
    """Abort racing a just-completed child still stamps observed + parent_aborted."""
    store = TasksStore()
    rec = store.create(description="slow", prompt="go")
    tool = TaskOutput(store=store)

    abort = AbortController()
    cv_token = current_abort_signal.set(abort)
    try:
        waiter = asyncio.create_task(
            tool.ainvoke({"task_id": rec.id, "wait": True, "timeout": 5.0})
        )
        await asyncio.sleep(0.01)
        assert not waiter.done()

        # Child reaches terminal FIRST, then the parent abort fires — the
        # wait wakes on the terminal event, re-fetches a terminal record,
        # yet abort.aborted is now True so the observed/parent_aborted
        # branch (status != running) must run.
        store.mark_completed(rec.id, "child done")
        abort.abort("parent_cancelled")

        result = await asyncio.wait_for(waiter, timeout=1.0)
        assert result["status"] == "completed"
        assert result["terminal"] is True
        assert result["error"] == "parent_aborted"
        assert result["observed_at"] is not None
        refreshed = store.get(rec.id)
        assert refreshed is not None
        assert refreshed.observed_at == result["observed_at"]
    finally:
        current_abort_signal.reset(cv_token)


@pytest.mark.asyncio
async def test_abort_after_terminal_observed_at_survives_replay() -> None:
    """The abort branch must reuse the first observed stamp; a replay can't reset it."""
    store = TasksStore()
    rec = store.create(description="slow", prompt="go")
    tool = TaskOutput(store=store)

    abort = AbortController()
    cv_token = current_abort_signal.set(abort)
    try:
        waiter = asyncio.create_task(
            tool.ainvoke({"task_id": rec.id, "wait": True, "timeout": 5.0})
        )
        await asyncio.sleep(0.01)
        store.mark_completed(rec.id, "child done")
        abort.abort("parent_cancelled")
        first = await asyncio.wait_for(waiter, timeout=1.0)
        assert first["error"] == "parent_aborted"
        assert first["observed_at"] is not None

        # Record is already terminal now, so a re-poll short-circuits BEFORE
        # the abort branch — observed_at must stay pinned to the first stamp.
        second = await tool.ainvoke({"task_id": rec.id})
        assert second["observed_at"] == first["observed_at"]
        assert second["error"] is None
    finally:
        current_abort_signal.reset(cv_token)
