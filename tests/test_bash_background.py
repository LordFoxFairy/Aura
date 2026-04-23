"""bash_background — fire-and-forget long-running shell as a TaskRecord.

The tool spawns a shell subprocess DETACHED from the tool invocation,
returns a ``task_id`` immediately, and writes rolling output into the
parent Agent's ``TasksStore`` so ``task_get`` / ``task_stop`` /
``task_list`` can observe / kill it. These tests exercise the lifecycle
axes: fast completion, long-running polling, timeout kill, task_stop
kill, stream prefixes, safety rejection, and ring-buffer boundedness.

All tests use real subprocesses (``/bin/sh -c``) because mocking asyncio
subprocess plumbing is worse than just running short commands. Tests
are gated on POSIX semantics (sh available, SIGTERM delivery) — same
target platforms as the blocking ``bash`` tool.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from aura.core.tasks.store import TasksStore
from aura.core.tasks.types import _SHELL_RECENT_ACTIVITIES_CAP
from aura.schemas.tool import ToolError
from aura.tools.bash_background import BashBackground
from aura.tools.task_get import TaskGet
from aura.tools.task_stop import TaskStop


def _make_tool() -> (
    tuple[
        BashBackground,
        TasksStore,
        dict[str, asyncio.subprocess.Process],
        dict[str, asyncio.Task[None]],
    ]
):
    store = TasksStore()
    running_shells: dict[str, asyncio.subprocess.Process] = {}
    running_tasks: dict[str, asyncio.Task[None]] = {}
    tool = BashBackground(
        store=store,
        running_shells=running_shells,
        running_tasks=running_tasks,
    )
    return tool, store, running_shells, running_tasks


async def _wait_for_terminal(
    store: TasksStore, task_id: str, timeout: float = 10.0,
) -> Any:
    """Poll until record.status != 'running'; return the record."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        rec = store.get(task_id)
        if rec is not None and rec.status != "running":
            return rec
        await asyncio.sleep(0.02)
    last = store.get(task_id)
    raise AssertionError(
        f"task {task_id[:8]} never reached terminal; status="
        f"{last.status if last is not None else 'missing'}",
    )


@pytest.mark.asyncio
async def test_short_command_completes_with_stdout_captured() -> None:
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "echo hello-aura"})
    assert out["status"] == "running"
    assert out["command"] == "echo hello-aura"
    task_id = out["task_id"]
    # Wait for the watcher to fully flip the record.
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id)
    assert rec.kind == "shell"
    assert rec.status == "completed"
    # final_result holds exit_code + tail.
    assert "exit_code=0" in rec.final_result
    assert "hello-aura" in rec.final_result
    assert any("[out] hello-aura" in a for a in rec.progress.recent_activities)


@pytest.mark.asyncio
async def test_long_running_returns_immediately_and_status_progresses() -> None:
    # sleep 3 then echo — tool must return while the child is still
    # running; a polled task_get must surface the running status.
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "sleep 2; echo ok"})
    task_id = out["task_id"]
    # Immediately after the tool returns: still running.
    rec = store.get(task_id)
    assert rec is not None
    assert rec.status == "running"
    # Let it finish.
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id, timeout=10.0)
    assert rec.status == "completed"
    assert any("[out] ok" in a for a in rec.progress.recent_activities)


@pytest.mark.asyncio
async def test_timeout_kills_child_and_marks_failed() -> None:
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "sleep 60", "timeout_sec": 1})
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id, timeout=10.0)
    assert rec.status == "failed"
    assert rec.error is not None
    assert "timed out after 1s" in rec.error


@pytest.mark.asyncio
async def test_task_stop_kills_shell_subprocess() -> None:
    tool, store, running_shells, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "sleep 60"})
    task_id = out["task_id"]
    # Give the spawn a tick so the process is registered.
    await asyncio.sleep(0.1)
    assert task_id in running_shells
    stop = TaskStop(
        store=store,
        running=running_tasks,
        running_shells=running_shells,
    )
    stop_out = await stop.ainvoke({"task_id": task_id})
    assert stop_out["status"] == "cancelled"
    rec = store.get(task_id)
    assert rec is not None
    assert rec.status == "cancelled"
    # Let the watcher finish so the test doesn't leak pending tasks. The
    # watcher's finally block may raise CancelledError once SIGKILL
    # closes the pipes and its awaits unwind — that's expected.
    for t in list(running_tasks.values()):
        with contextlib.suppress(asyncio.CancelledError):
            await t


@pytest.mark.asyncio
async def test_stderr_captured_with_err_prefix() -> None:
    tool, store, _, running_tasks = _make_tool()
    # ``>&2`` redirects to stderr in /bin/sh.
    out = await tool.ainvoke(
        {"command": "echo fail-msg >&2"},
    )
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id)
    assert rec.status == "completed"
    assert any(
        a.startswith("[err] ") and "fail-msg" in a
        for a in rec.progress.recent_activities
    ), rec.progress.recent_activities


@pytest.mark.asyncio
async def test_stdout_captured_with_out_prefix() -> None:
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "echo greet"})
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id)
    assert rec.status == "completed"
    assert any(
        a.startswith("[out] ") and "greet" in a
        for a in rec.progress.recent_activities
    ), rec.progress.recent_activities


@pytest.mark.asyncio
async def test_safety_rejects_command_substitution() -> None:
    tool, store, _, _ = _make_tool()
    with pytest.raises(ToolError, match="command substitution"):
        await tool.ainvoke({"command": "echo $(whoami)"})
    with pytest.raises(ToolError, match="command substitution"):
        await tool.ainvoke({"command": "echo `whoami`"})
    with pytest.raises(ToolError, match="command substitution"):
        await tool.ainvoke({"command": "bash -c 'echo hi'"})
    # No task records should have been created for these rejects.
    assert store.list() == []


@pytest.mark.asyncio
async def test_progress_ring_bounded_at_shell_cap() -> None:
    tool, store, _, running_tasks = _make_tool()
    # Emit 40 lines of stdout — well past the 20-line cap. ``seq`` is
    # POSIX and produces deterministic numbered lines without needing
    # $((...)) arithmetic (which trips the bash-safety rule).
    out = await tool.ainvoke({
        "command": "seq 1 40",
        "timeout_sec": 30,
    })
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id, timeout=15.0)
    assert rec.status == "completed"
    # Ring is bounded to 20 entries. line_count tracks the monotonic total.
    assert len(rec.progress.recent_activities) <= _SHELL_RECENT_ACTIVITIES_CAP
    assert rec.progress.line_count >= 40


@pytest.mark.asyncio
async def test_nonzero_exit_marks_failed_with_exit_code() -> None:
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "exit 7"})
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id)
    assert rec.status == "failed"
    assert "exit_code=7" in (rec.error or "")


@pytest.mark.asyncio
async def test_task_list_filters_by_shell_kind() -> None:
    # Cross-check: task_list's new ``kind`` filter surfaces only shell
    # tasks when requested. Mixes a subagent (via store.create) and a
    # shell (via bash_background) in the same store.
    from aura.tools.task_list import TaskList

    tool, store, _, running_tasks = _make_tool()
    # Inject a fake subagent record directly into the store so we don't
    # need to spin up a real SubagentFactory here.
    store.create(description="sub", prompt="p")
    out = await tool.ainvoke({"command": "echo sh"})
    await asyncio.gather(*running_tasks.values())
    listing = await TaskList(store=store).ainvoke({"kind": "shell"})
    kinds = {t["kind"] for t in listing["tasks"]}
    assert kinds == {"shell"}
    descs = [t["description"] for t in listing["tasks"]]
    assert any(d.startswith("bg: ") for d in descs)
    # And the subagent filter sees only the injected one.
    listing2 = await TaskList(store=store).ainvoke({"kind": "subagent"})
    assert {t["kind"] for t in listing2["tasks"]} == {"subagent"}
    assert out["task_id"] not in {t["id"] for t in listing2["tasks"]}


@pytest.mark.asyncio
async def test_task_get_surfaces_shell_kind_and_line_count() -> None:
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "echo a; echo b; echo c"})
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id)
    assert rec.status == "completed"
    get_out = await TaskGet(store=store).ainvoke({"task_id": task_id})
    assert get_out["kind"] == "shell"
    assert get_out["progress"]["line_count"] >= 3
    assert get_out["progress"]["tool_count"] == 0  # shell never fires tool events


