# Large Task Reliability Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Aura's large-task orchestration reliable enough for long-running delegated coding work without stale task state, hidden teammate failures, or compact-related context pollution.

**Architecture:** Strengthen existing state contracts rather than rewriting the agent. `TasksStore` owns terminal lifecycle and observed-result state; task tools expose and update observation state; `TeamManager` owns teammate task lifecycle/model metadata; compact and context rendering read those contracts without inferring state from text.

**Tech Stack:** Python 3.11+, pytest/pytest-asyncio, pydantic v2, LangChain core test fakes, Aura's existing `TasksStore`, `TaskRecord`, `HookChain`, `TeamManager`, and compact modules.

---

## Chunk 1: Task State Invariants

### Task 1: Terminal Transitions Are Monotonic

**Files:**
- Modify: `aura/core/tasks/store.py`
- Modify: `tests/test_tasks_store.py`

- [ ] **Step 1: Write failing tests**

Add tests that prove terminal state cannot be overwritten and terminal listeners fire only once:

```python
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
```

Add equivalent focused tests for failed and cancelled first.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -q tests/test_tasks_store.py::test_terminal_transition_is_idempotent_after_completed
```

Expected: FAIL because later `mark_*` calls overwrite status and listeners fire again.

- [ ] **Step 3: Implement minimal store guard**

Add a private helper in `TasksStore`, for example:

```python
def _terminal_record(self, task_id: str) -> TaskRecord | None:
    rec = self._records.get(task_id)
    if rec is None or rec.status != "running":
        return None
    return rec
```

Use it in `mark_completed`, `mark_failed`, and `mark_cancelled`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest -q tests/test_tasks_store.py
```

Expected: all tests pass.

### Task 2: Task List Supports Teammates

**Files:**
- Modify: `aura/tools/task_list.py`
- Modify: `tests/test_task_list.py`

- [ ] **Step 1: Write failing test**

Add a teammate task and assert `kind="teammate"` filters it:

```python
def test_task_list_filters_teammates() -> None:
    store = TasksStore()
    store.create("worker", "p", kind="subagent")
    teammate = store.create("teammate: scout", "idle", kind="teammate")
    tool = TaskList(store=store)

    result = tool._run(kind="teammate")

    assert [t["id"] for t in result["tasks"]] == [teammate.id]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -q tests/test_task_list.py::test_task_list_filters_teammates
```

Expected: FAIL at pydantic/schema or filter validation because `teammate` is not allowed.

- [ ] **Step 3: Implement minimal schema update**

Change `_KindFilter` to include `"teammate"` and update tool descriptions.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest -q tests/test_task_list.py
```

Expected: all tests pass.

---

## Chunk 2: Task Observation and Compact Resilience

### Task 3: Observed Terminal Tasks Stop Reinjecting After Retrieval

**Files:**
- Modify: `aura/core/tasks/types.py`
- Modify: `aura/core/tasks/store.py`
- Modify: `aura/tools/task_get.py`
- Modify: `aura/tools/task_output.py`
- Modify: `aura/core/compact/compact.py`
- Modify: `tests/test_task_tools.py` or `tests/test_task_observability.py`
- Modify: `tests/test_compact_active_tasks.py`

- [ ] **Step 1: Write failing store/tool tests**

Add an observation field to the expected behavior first. Preferred public shape:

```python
def test_task_get_marks_terminal_task_observed() -> None:
    store = TasksStore()
    rec = store.create("x", "p")
    store.mark_completed(rec.id, "done")
    tool = TaskGet(store=store)

    result = tool._run(rec.id)

    assert result["observed_at"] is not None
    assert store.get(rec.id).observed_at == result["observed_at"]
```

Add equivalent `task_output` coverage for terminal results.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -q tests/test_task_observability.py::test_task_get_marks_terminal_task_observed
```

Expected: FAIL because no observed state exists.

- [ ] **Step 3: Implement observed state**

Add `observed_at: float | None = None` to `TaskRecord`.
Add `TasksStore.mark_observed(task_id) -> float | None`, which only stamps terminal tasks and preserves the first timestamp.
Call it from `task_get` and `task_output` only when `rec.status != "running"`.
Include `observed_at` in serialized task snapshots.

- [ ] **Step 4: Add compact failing test**

In `tests/test_compact_active_tasks.py`, assert:

- running task is preserved
- unobserved terminal tasks are preserved for `completed`, `failed`, and
  `cancelled`
- observed terminal tasks are not preserved for `completed`, `failed`, and
  `cancelled`

- [ ] **Step 5: Update compact**

In `_build_active_task_messages`, preserve:

```python
if rec.status == "running":
    keep = True
elif rec.status != "running" and rec.observed_at is None:
    keep = True
else:
    keep = False
```

- [ ] **Step 6: Verify GREEN**

Run:

```bash
uv run pytest -q tests/test_task_observability.py tests/test_task_tools.py tests/test_compact_active_tasks.py
```

Expected: all tests pass.

- [ ] **Step 7: Add mixed large-task component and integration coverage**

Add a focused test that creates several tasks in one `TasksStore`, marks one
running, one completed, one failed/timeout-like, observes one terminal task via
`task_get` or `task_output`, runs compact, and then asserts:

- `task_list` still reports the correct statuses
- the running task remains injected
- the unobserved failed/completed task remains injected
- the observed terminal task is not injected
- observed timestamps remain stable after repeated reads

Also add an integration-level test in `tests/integration/test_subagent_dag.py`
that drives the real task tools through an `Agent`/`FakeChatModel` flow:

- parent creates multiple tasks or uses existing integration fake subagents
- one task completes, one fails or is cancelled/timeout-like, one remains running
- parent observes one terminal task through `task_get` or `task_output`
- parent compacts
- parent can still call `task_list`/`task_get` and see correct statuses and
  observed state

### Task 4: Task Notifications Preserve Latest Events

**Files:**
- Modify: `aura/core/memory/context.py`
- Modify: `tests/test_task_observability.py` or `tests/test_context.py`

- [ ] **Step 1: Write failing test**

Create more than five task notifications and assert the rendered block contains the latest five and an omitted-earlier count.

- [ ] **Step 2: Verify RED**

Run the exact new test.

Expected: FAIL because current code uses `drained[:cap]`.

- [ ] **Step 3: Implement minimal change**

Use `drained[-cap:]` when the queue overflows and keep the omitted message as `(<N> more earlier)`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest -q tests/test_task_observability.py tests/test_context.py
```

Expected: all tests pass.

---

## Chunk 3: Teammate Lifecycle and Model Metadata

### Task 5: Teammate Model Spec Propagates

**Files:**
- Modify: `aura/core/teams/manager.py`
- Modify: `tests/test_teams_manager.py`

- [ ] **Step 1: Write failing tests**

Cover both `add_member` and `aadd_member`. This is required because the two
paths duplicate teammate record creation and child spawn behavior.

```python
def test_add_member_records_model_spec(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")

    member = mgr.add_member("scout", model_name="openai:gpt-4o-mini")
    task_id = mgr._member_task_ids[member.name]
    rec = mgr._tasks_store.get(task_id)

    assert rec is not None
    assert rec.model_spec == "openai:gpt-4o-mini"
```

Also assert the factory `spawn` call receives `model_spec=model_name` using the
existing fake/custom factory patterns in this test file. Add one sync
`add_member` test and one async `aadd_member` test.

- [ ] **Step 2: Verify RED**

Run the new tests.

Expected: FAIL because `model_spec` is not stored or passed.

- [ ] **Step 3: Implement minimal propagation**

When creating teammate records, pass `model_spec=model_name or self._factory.parent_model_spec`.
When spawning teammate child agents, pass `model_spec=model_name`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest -q tests/test_teams_manager.py tests/test_team_view_commands.py
```

Expected: all tests pass.

### Task 6: Teammate Tasks Reach Terminal States

**Files:**
- Modify: `aura/core/teams/manager.py`
- Modify: `aura/core/teams/runtime.py` if needed
- Modify: `tests/test_teams_manager.py`
- Modify: `tests/test_teams_runtime.py` if needed

- [ ] **Step 1: Write failing tests**

Pin the state table:

- natural runtime return -> `completed`
- graceful explicit removal -> `cancelled`
- force kill/team delete/session cleanup -> `cancelled`
- runtime exception -> `failed` with exception type and message

Cover both `add_member` and `aadd_member` where each path owns duplicated setup.
Start with manager-level tests because `TeamManager` owns `TaskRecord`.

- [ ] **Step 2: Verify RED**

Run each new test and confirm it fails for the expected missing terminal mark.

- [ ] **Step 3: Implement lifecycle marking**

Prefer small helper methods in `TeamManager`:

```python
def _mark_member_completed(self, name: str) -> None: ...
def _mark_member_cancelled(self, name: str) -> None: ...
def _mark_member_failed(self, name: str, exc: BaseException) -> None: ...
```

Wire them to in-process runtime task done callbacks and removal/delete cleanup paths. Do not overwrite terminal records; rely on `TasksStore` idempotency.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest -q tests/test_teams_manager.py tests/test_teams_runtime.py tests/test_teams_shutdown_handshake.py
```

Expected: all tests pass.

---

## Chunk 4: Permission/Subagent End-to-End Coverage and Lifecycle Flake

### Task 7: Parent/Child Permission Contracts Are Tested End-to-End

**Files:**
- Modify: `tests/test_c1_subagent_permission.py` or `tests/integration/test_subagent_dag.py`
- Modify production code only if the new tests expose a real gap.

- [ ] **Step 1: Write failing or proving tests**

Add full-path tests:

- parent agent in bypass emits `task_create`
- child executes a real tool
- result/journal/denials match the documented bypass contract
- deny rules still block under bypass if expected by current permission semantics
- disable_bypass clamps child behavior when parent config disables bypass
- nearby permission hook / config comments state the current contract clearly:
  bypass is a broad permission bypass, deny rules remain bypass-immune, and any
  stricter safety floor requires explicit tests plus migration notes

- [ ] **Step 2: Verify RED or prove existing behavior**

Run the exact new tests. If they pass immediately, keep them as coverage and do not change production code.

- [ ] **Step 3: Implement only if needed**

If a test fails due to missing inheritance of deny/ask/disable_bypass, pass the missing permission policy into `SubagentFactory` and child `make_permission_hook`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest -q tests/test_c1_subagent_permission.py tests/integration/test_subagent_dag.py tests/test_permission_deny_ask.py
```

Expected: all tests pass.

### Task 8: Lifecycle Hook Flake Is Isolated and Stabilized

**Files:**
- Modify: `tests/test_lifecycle_hooks.py`
- Modify: `aura/core/hooks/lifecycle.py` or lifecycle teardown code only if root cause is production behavior.

- [ ] **Step 1: Create a local reproduction**

Use pytest ordering around journal/lifecycle tests to reproduce:

```bash
uv run pytest -q tests/test_hooks.py tests/test_journal.py tests/test_journal_session_scope.py tests/test_lifecycle_hooks.py
```

If that does not reproduce, use repeated runs:

```bash
for i in 1 2 3 4 5; do uv run pytest -q tests/test_lifecycle_hooks.py::test_settings_json_can_register_external_hook_command || break; done
```

- [ ] **Step 2: Add diagnostic or regression test**

Do not add sleeps. Confirm whether the external command returns `None`, the hook does not run, or persistence stores the old prompt.

- [ ] **Step 3: Fix root cause**

Likely fixes include deterministic agent teardown, subprocess adapter cleanup, or test isolation around global journal/session state. Pick only the evidenced root cause.

- [ ] **Step 4: Verify**

Run:

```bash
uv run pytest -q tests/test_lifecycle_hooks.py
uv run pytest -q
```

Expected: full suite passes consistently.

---

## Chunk 5: Documentation and Release Gates

### Task 9: README Matches Current Task/Subagent Behavior

**Files:**
- Modify: `README.md`
- Modify: `Makefile` only if adding explicit test targets

- [ ] **Step 1: Write docs diff**

Update README sections for:

- task/subagent inheritance behavior
- recursion depth cap
- transcript and metadata files
- model override
- permission inheritance and current bypass semantics
- current bypass contract: broad permission bypass, deny rules remain
  bypass-immune, stricter safety floor would be a deliberate future change
- teammate current limitations if token/tool progress parity remains deferred

- [ ] **Step 2: Add or document test tiers**

Either add Make targets or document commands:

```make
test-fast:
	uv run pytest -q tests

test-integration:
	uv run pytest -q tests/integration

verify-real-llm:
	uv run python scripts/verify_real_llm.py
```

Only add targets that match existing project conventions and do not require credentials by default.

- [ ] **Step 3: Verify docs/targets**

Run:

```bash
uv run pytest -q tests/test_package.py tests/test_cli_entry.py
make -n check
```

Expected: commands are syntactically valid and no docs-related tests fail.

---

## Final Verification

- [ ] Run focused tests for changed areas:

```bash
uv run pytest -q tests/test_tasks_store.py tests/test_task_list.py tests/test_task_observability.py tests/test_task_tools.py tests/test_compact_active_tasks.py tests/test_teams_manager.py tests/test_teams_runtime.py tests/test_teams_shutdown_handshake.py tests/test_c1_subagent_permission.py tests/integration/test_subagent_dag.py tests/test_lifecycle_hooks.py
```

- [ ] Run lint:

```bash
uv run ruff check .
```

- [ ] Run type checks:

```bash
uv run mypy aura tests
```

- [ ] Run full suite:

```bash
uv run pytest -q
```

- [ ] Request final code review with the implementation diff.

- [ ] Commit only relevant files. Do not include unrelated desktop frontend changes.
