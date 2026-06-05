"""TasksStore — keyed store for subagent TaskRecords.

Phase E (0.5.0) MVP. Append-only semantics for a completed record's fields:
once a terminal state is set (completed/failed/cancelled) + ``finished_at``
is stamped, subsequent mutations must not silently corrupt the record.

Covers the pure store; integration with the dispatch / tool layer lives in
``test_task_tools.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.application.tasks.store import TasksStore
from aura.domain.task import (
    SHELL_RECENT_ACTIVITIES_CAP,
    TaskKind,
    TaskRecord,
    TaskStatus,
)
from aura.infrastructure.persistence import journal


def test_create_record_and_list() -> None:
    store = TasksStore()
    rec = store.create(description="scan repo", prompt="look for TODOs")
    assert isinstance(rec, TaskRecord)
    assert rec.status == "running"
    assert rec.description == "scan repo"
    assert rec.prompt == "look for TODOs"
    assert rec.final_result is None
    assert rec.error is None
    assert store.list() == [rec]


def test_append_message_mutates_record() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    msg = HumanMessage(content="hi")
    store.append_message(rec.id, msg)
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.messages == [msg]


def test_mark_completed_transitions_status_and_sets_final_result() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.append_message(rec.id, AIMessage(content="partial"))
    store.mark_completed(rec.id, "final-answer")
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "completed"
    assert r.final_result == "final-answer"
    assert r.finished_at is not None


def test_mark_failed_sets_error_and_status() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.mark_failed(rec.id, "RuntimeError: boom")
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "failed"
    assert r.error == "RuntimeError: boom"
    assert r.finished_at is not None


def test_get_returns_none_for_unknown_id() -> None:
    store = TasksStore()
    assert store.get("no-such-id") is None


def test_list_filters_by_status() -> None:
    store = TasksStore()
    a = store.create(description="a", prompt="pa")
    b = store.create(description="b", prompt="pb")
    c = store.create(description="c", prompt="pc")
    store.mark_completed(a.id, "done")
    store.mark_failed(b.id, "bad")
    running = store.list(status="running")
    assert running == [c]
    completed = store.list(status="completed")
    assert [r.id for r in completed] == [a.id]
    failed = store.list(status="failed")
    assert [r.id for r in failed] == [b.id]


def test_mark_cancelled_sets_status() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.mark_cancelled(rec.id)
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled"
    assert r.finished_at is not None


def test_terminal_transition_is_idempotent_after_completed() -> None:
    store = TasksStore()
    rec = store.create(description="x", prompt="p")
    seen: list[str] = []
    store.add_terminal_listener(lambda r: seen.append(r.status))

    store.mark_completed(rec.id, "done")
    finished_at = rec.finished_at
    store.mark_cancelled(rec.id)
    store.mark_failed(rec.id, "boom")

    assert rec.status == "completed"
    assert rec.final_result == "done"
    assert rec.error is None
    assert rec.finished_at == finished_at
    assert seen == ["completed"]


def test_terminal_transition_is_idempotent_after_failed() -> None:
    store = TasksStore()
    rec = store.create(description="x", prompt="p")
    seen: list[str] = []
    store.add_terminal_listener(lambda r: seen.append(r.status))

    store.mark_failed(rec.id, "boom")
    finished_at = rec.finished_at
    store.mark_completed(rec.id, "done")
    store.mark_cancelled(rec.id)

    assert rec.status == "failed"
    assert rec.final_result is None
    assert rec.error == "boom"
    assert rec.finished_at == finished_at
    assert seen == ["failed"]


def test_terminal_transition_is_idempotent_after_cancelled() -> None:
    store = TasksStore()
    rec = store.create(description="x", prompt="p")
    seen: list[str] = []
    store.add_terminal_listener(lambda r: seen.append(r.status))

    store.mark_cancelled(rec.id)
    finished_at = rec.finished_at
    store.mark_completed(rec.id, "done")
    store.mark_failed(rec.id, "boom")

    assert rec.status == "cancelled"
    assert rec.final_result is None
    assert rec.error is None
    assert rec.finished_at == finished_at
    assert seen == ["cancelled"]


# --- mutator no-op guards: unknown task_id must never raise (callers fire-and-forget) ---


def test_mutators_silently_noop_for_unknown_id() -> None:
    """Dispatch fires progress events by id; a stale/unknown id must not crash the agent."""
    store = TasksStore()
    missing = "no-such-task"
    # None of these may raise, and none may materialise a record.
    store.record_activity(missing, "tool")
    store.record_started(missing)
    store.record_activity_note(missing, "note")
    store.record_shell_line(missing, "line")
    store.record_shell_marker(missing, "marker")
    store.append_message(missing, AIMessage(content="x"))
    store.record_token_usage(missing, input_tokens=10, output_tokens=20)
    store.set_transcript_path(missing, Path("/tmp/x.jsonl"))
    assert store.mark_observed(missing) is None
    store.mark_completed(missing, "r")
    store.mark_failed(missing, "e")
    store.mark_cancelled(missing)
    assert store.get(missing) is None
    assert store.list() == []


# --- record_activity: tool bookkeeping + listener fan-out + ring bound ---


def test_record_activity_bumps_tool_count_and_fires_listeners() -> None:
    """task_get serialises tool_count + recent tail; both must advance per child tool event."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    seen: list[tuple[str, str]] = []
    store.add_activity_listener(lambda r, a: seen.append((r.id, a)))
    store.record_activity(rec.id, "Read")
    store.record_activity(rec.id, "Bash")
    assert rec.progress.tool_count == 2
    assert rec.progress.last_activity_at is not None
    assert rec.progress.recent_activities == ["Read", "Bash"]
    assert seen == [(rec.id, "Read"), (rec.id, "Bash")]


def test_record_activity_caps_recent_ring_at_default_cap() -> None:
    """recent_activities is serialised into task_get; an unbounded ring would bloat the payload."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    for i in range(12):
        store.record_activity(rec.id, f"act-{i}")
    # Default cap is 5; only the newest tail survives.
    assert rec.progress.recent_activities == [f"act-{i}" for i in range(7, 12)]
    assert rec.progress.tool_count == 12


def test_record_activity_listener_exception_is_swallowed_and_journaled(
    tmp_path: Path,
) -> None:
    """A buggy observer must not abort a running subagent; the failure is audited, not raised."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")

    def boom(_r: TaskRecord, _a: str) -> None:
        raise RuntimeError("listener-kaboom")

    good_seen: list[str] = []
    store.add_activity_listener(boom)
    store.add_activity_listener(lambda _r, a: good_seen.append(a))

    log = tmp_path / "journal.jsonl"
    with journal.session_scope(log):
        store.record_activity(rec.id, "Read")  # must not raise

    # The healthy listener still ran even though the first one blew up.
    assert good_seen == ["Read"]
    events = [json.loads(line) for line in log.read_text().splitlines()]
    err = next(e for e in events if e["event"] == "tasks_activity_listener_error")
    assert err["task_id"] == rec.id
    assert "RuntimeError: listener-kaboom" in err["error"]


# --- record_started ---


def test_record_started_fires_started_listeners_only() -> None:
    """The /tasks UI watches started transitions; this hook delivers the record, not progress."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    started: list[str] = []
    store.add_started_listener(lambda r: started.append(r.id))
    store.record_started(rec.id)
    store.record_started(rec.id)
    assert started == [rec.id, rec.id]


def test_record_started_listener_exception_is_swallowed_and_journaled(
    tmp_path: Path,
) -> None:
    """A failing start observer must not stop the task from running; failure is journaled."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.add_started_listener(lambda _r: (_ for _ in ()).throw(ValueError("start-bad")))
    log = tmp_path / "journal.jsonl"
    with journal.session_scope(log):
        store.record_started(rec.id)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    err = next(e for e in events if e["event"] == "tasks_started_listener_error")
    assert err["task_id"] == rec.id
    assert "ValueError: start-bad" in err["error"]


# --- record_activity_note: non-tool activity must NOT inflate tool_count ---


def test_record_activity_note_preserves_tool_count_semantics() -> None:
    """Non-tool notes share the recent ring but must not be miscounted as tool invocations."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.record_activity_note(rec.id, "thinking")
    assert rec.progress.tool_count == 0
    assert rec.progress.last_activity_at is not None
    assert rec.progress.recent_activities == ["thinking"]


# --- record_shell_line / record_shell_marker: line bookkeeping, stall-detector contract ---


def test_record_shell_line_bumps_line_count_and_uses_shell_cap() -> None:
    """Shell output is chattier than tool names, so it rides the wider SHELL cap, not default."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p", kind="shell")
    for i in range(SHELL_RECENT_ACTIVITIES_CAP + 3):
        store.record_shell_line(rec.id, f"out-{i}")
    assert rec.progress.line_count == SHELL_RECENT_ACTIVITIES_CAP + 3
    assert len(rec.progress.recent_activities) == SHELL_RECENT_ACTIVITIES_CAP
    assert rec.progress.recent_activities[-1] == f"out-{SHELL_RECENT_ACTIVITIES_CAP + 2}"
    assert rec.progress.last_activity_at is not None


def test_record_shell_marker_does_not_touch_last_activity_at() -> None:
    """The stall detector reads last_activity_at; a marker must not reset the stall clock."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p", kind="shell")
    assert rec.progress.last_activity_at is None
    store.record_shell_marker(rec.id, "<<MARK>>")
    assert rec.progress.last_activity_at is None  # untouched on purpose
    assert rec.progress.line_count == 0  # markers are not output lines
    assert rec.progress.recent_activities == ["<<MARK>>"]


def test_record_shell_marker_uses_shell_cap_not_default() -> None:
    """Markers share the shell ring; they must honour the 20-slot cap, not the 5-slot tool cap."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p", kind="shell")
    for i in range(SHELL_RECENT_ACTIVITIES_CAP + 2):
        store.record_shell_marker(rec.id, f"m-{i}")
    assert len(rec.progress.recent_activities) == SHELL_RECENT_ACTIVITIES_CAP


# --- record_token_usage: accumulation, negative clamp, idempotent re-add ---


@pytest.mark.parametrize(
    ("in_t", "out_t", "exp_in", "exp_out"),
    [
        (0, 0, 0, 0),  # zero
        (-1, -50, 0, 0),  # negatives clamp to 0
        (10, 20, 10, 20),  # plain
        (10**12, 10**12, 10**12, 10**12),  # huge ints don't overflow (Python bigint)
    ],
)
def test_record_token_usage_clamps_and_sums(
    in_t: int, out_t: int, exp_in: int, exp_out: int
) -> None:
    """Token totals feed budget/quotas; one garbage negative must not corrupt the running sum."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.record_token_usage(rec.id, input_tokens=in_t, output_tokens=out_t)
    assert rec.progress.input_tokens == exp_in
    assert rec.progress.output_tokens == exp_out
    assert rec.progress.token_count == exp_in + exp_out


def test_record_token_usage_accumulates_across_calls() -> None:
    """Usage arrives per-chunk; calling twice must add, token_count stays the derived total."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.record_token_usage(rec.id, input_tokens=5, output_tokens=7)
    store.record_token_usage(rec.id, input_tokens=3, output_tokens=2)
    assert rec.progress.input_tokens == 8
    assert rec.progress.output_tokens == 9
    assert rec.progress.token_count == 17


# --- mark_observed: stamp-once stability + running guard ---


def test_mark_observed_returns_none_while_running() -> None:
    """A still-running task has no final result to observe, so observed_at must stay unstamped."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    assert store.mark_observed(rec.id) is None
    assert rec.observed_at is None


def test_mark_observed_is_stable_on_repeated_reads() -> None:
    """observed_at drives the 'new result' badge; re-reading a terminal task must not restamp."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.mark_completed(rec.id, "done")
    first = store.mark_observed(rec.id)
    assert first is not None
    second = store.mark_observed(rec.id)
    assert second == first  # idempotent stamp
    assert rec.observed_at == first


@pytest.mark.parametrize(
    "terminal",
    [
        lambda s, tid: s.mark_completed(tid, "ok"),
        lambda s, tid: s.mark_failed(tid, "boom"),
        lambda s, tid: s.mark_cancelled(tid),
    ],
)
def test_mark_observed_stamps_for_every_terminal_status(
    terminal: object,
) -> None:
    """Failed/cancelled results are observable too, not just completed ones."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    assert callable(terminal)
    terminal(store, rec.id)
    stamped = store.mark_observed(rec.id)
    assert stamped is not None
    assert rec.observed_at == stamped


# --- set_transcript_path ---


def test_set_transcript_path_pins_location() -> None:
    """task_get exposes the JSONL path for replay; the store must pin exactly what it is given."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    p = Path("/var/tmp/aura/run-1.jsonl")
    store.set_transcript_path(rec.id, p)
    assert rec.transcript_path == p


# --- terminal_event / _LazyEvent: lazy loop binding, pre-set short-circuit ---


def test_terminal_event_is_memoised_per_task() -> None:
    """Multiple waiters on one task must share a single event so set() wakes them all."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    ev1 = store.terminal_event(rec.id)
    ev2 = store.terminal_event(rec.id)
    assert ev1 is ev2
    assert ev1.is_set() is False  # running task → unset


def test_terminal_event_preset_for_already_terminal_record() -> None:
    """A consumer that asks for the event AFTER the task finished must see it already set."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.mark_completed(rec.id, "done")
    ev = store.terminal_event(rec.id)  # requested post-terminal
    assert ev.is_set() is True


async def test_terminal_event_wait_unblocks_on_mark_completed() -> None:
    """A waiter parked before completion must wake exactly when the terminal mark fires."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    ev = store.terminal_event(rec.id)

    async def waiter() -> str:
        await ev.wait()
        return "woke"

    task = asyncio.ensure_future(waiter())
    await asyncio.sleep(0)  # let the waiter park on the event
    assert not task.done()
    store.mark_completed(rec.id, "done")
    assert await asyncio.wait_for(task, timeout=0.1) == "woke"
    assert ev.is_set() is True


def test_fire_terminal_creates_preset_placeholder_when_no_event_requested() -> None:
    """If a task finishes before anyone asked for its event, a later waiter must short-circuit."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    # No terminal_event() requested before the mark → _fire_terminal builds a placeholder.
    store.mark_failed(rec.id, "boom")
    ev = store.terminal_event(rec.id)
    assert ev.is_set() is True


async def test_lazy_event_preset_before_loop_binding_then_wait_returns() -> None:
    """The event is created outside any loop; a pre-mark set() must survive the first wait()."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.mark_cancelled(rec.id)  # placeholder set() runs with no bound asyncio.Event yet
    ev = store.terminal_event(rec.id)
    assert ev.is_set() is True
    # wait() must return immediately, not deadlock, because the pre_set flag carries over.
    await asyncio.wait_for(ev.wait(), timeout=0.1)


# --- listener registry: add / remove / remove-missing tolerance ---


@pytest.mark.parametrize(
    ("add", "remove", "fire"),
    [
        (
            "add_terminal_listener",
            "remove_terminal_listener",
            lambda s, tid: s.mark_completed(tid, "ok"),
        ),
        (
            "add_started_listener",
            "remove_started_listener",
            lambda s, tid: s.record_started(tid),
        ),
    ],
)
def test_removed_listener_stops_receiving_records(add: str, remove: str, fire: object) -> None:
    """Unsubscribing (e.g. a closed panel) stops callbacks — a stale closure would leak/raise."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    hits: list[str] = []

    def cb(r: TaskRecord) -> None:
        hits.append(r.id)

    getattr(store, add)(cb)
    getattr(store, remove)(cb)
    assert callable(fire)
    fire(store, rec.id)
    assert hits == []


def test_remove_activity_listener_present_and_then_missing() -> None:
    """Double-unsubscribe must be a tolerant no-op, not a ValueError that crashes teardown."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    hits: list[str] = []

    def cb(r: TaskRecord, _a: str) -> None:
        hits.append(r.id)

    store.add_activity_listener(cb)
    store.remove_activity_listener(cb)  # present → removed
    store.remove_activity_listener(cb)  # already gone → silent
    store.record_activity(rec.id, "Read")
    assert hits == []


@pytest.mark.parametrize(
    "remove",
    ["remove_terminal_listener", "remove_started_listener", "remove_activity_listener"],
)
def test_remove_unknown_listener_is_silent(remove: str) -> None:
    """Removing a never-registered callback must not raise (idempotent teardown contract)."""
    store = TasksStore()

    def never(_r: TaskRecord, *_rest: object) -> None: ...

    getattr(store, remove)(never)  # must not raise


def test_fire_terminal_listener_exception_is_swallowed_and_journaled(
    tmp_path: Path,
) -> None:
    """A failing terminal observer must not block the task's terminal transition; it is audited."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.add_terminal_listener(lambda _r: (_ for _ in ()).throw(KeyError("term-bad")))
    after: list[str] = []
    store.add_terminal_listener(lambda r: after.append(r.status))

    log = tmp_path / "journal.jsonl"
    with journal.session_scope(log):
        store.mark_completed(rec.id, "done")  # must not raise

    assert rec.status == "completed"  # transition still applied
    assert after == ["completed"]  # later healthy listener still ran
    events = [json.loads(line) for line in log.read_text().splitlines()]
    err = next(e for e in events if e["event"] == "tasks_terminal_listener_error")
    assert err["task_id"] == rec.id
    assert "KeyError" in err["error"]


# --- list: kind filter + newest-first limit ordering ---


def test_list_filters_by_kind() -> None:
    """The /tasks view tabs by kind; the filter must not leak shell tasks into the subagent list."""
    store = TasksStore()
    sub = store.create(description="s", prompt="p", kind="subagent")
    sh = store.create(description="h", prompt="p", kind="shell")
    store.create(description="t", prompt="p", kind="teammate")
    assert [r.id for r in store.list(kind="subagent")] == [sub.id]
    assert [r.id for r in store.list(kind="shell")] == [sh.id]


def test_list_limit_returns_newest_first() -> None:
    """A capped /tasks list must surface the most recent tasks, not an arbitrary insertion slice."""
    store = TasksStore()
    a = store.create(description="a", prompt="p")
    b = store.create(description="b", prompt="p")
    c = store.create(description="c", prompt="p")
    # started_at is wall-clock and may collide; force a strict ordering for determinism.
    a.started_at, b.started_at, c.started_at = 100.0, 200.0, 300.0
    limited = store.list(limit=2)
    assert [r.id for r in limited] == [c.id, b.id]
    assert len(store.list(limit=10)) == 3  # limit > size returns all


@pytest.mark.parametrize(
    ("status", "kind"),
    [("running", "subagent"), ("completed", "shell")],
)
def test_list_combined_status_and_kind_filters_intersect(
    status: TaskStatus, kind: TaskKind
) -> None:
    """Both filters must AND together; a status-only or kind-only match must be excluded."""
    store = TasksStore()
    match = store.create(description="m", prompt="p", kind=kind)
    if status != "running":
        store.mark_completed(match.id, "ok")
    # Decoys that satisfy exactly one predicate.
    store.create(description="wrong-kind", prompt="p", kind="teammate")
    other = store.create(description="wrong-status", prompt="p", kind=kind)
    if status == "running":
        store.mark_completed(other.id, "ok")
    result = store.list(status=status, kind=kind)
    assert [r.id for r in result] == [match.id]


# --- create: metadata isolation (defensive copy) ---


def test_create_copies_metadata_so_caller_mutation_does_not_leak() -> None:
    """The store snapshots metadata; later caller-side edits must not rewrite the record."""
    store = TasksStore()
    src: dict[str, object] = {"k": "v"}
    rec = store.create(description="d", prompt="p", metadata=src)
    src["k"] = "mutated"
    src["new"] = 1
    assert rec.metadata == {"k": "v"}  # frozen at creation time


def test_create_none_metadata_yields_empty_dict() -> None:
    """Omitted metadata must normalise to an empty dict, never None, so downstream .get is safe."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p", metadata=None)
    assert rec.metadata == {}
