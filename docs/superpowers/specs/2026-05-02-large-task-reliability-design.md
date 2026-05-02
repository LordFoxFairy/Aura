# Large Task Reliability Design

Date: 2026-05-02

## Purpose

Raise Aura's reliability for long-running, multi-agent coding tasks. The target
is not to claim full Codex or Claude Code parity. The target is a practical
70% capability floor: Aura should be able to plan, delegate, observe, compact,
recover, and verify a large task without stale task state, hidden failures, or
context pollution derailing the session.

## Current Baseline

Aura already has a strong foundation:

- An explicit async agent loop with tool batching, permissions, hooks, compact,
  skills, MCP, and session persistence.
- Fire-and-forget subagents with isolated storage/context, transcript files,
  timeout, token tracking, summary tracking, recursive depth cap, and model
  override support.
- Background shell tasks and team/teammate orchestration.
- Quality gates through `make check`: `ruff`, `mypy`, and `pytest`.

Fresh verification during the audit:

- `uv sync --extra all --extra dev` succeeded.
- `uv run ruff check .` passed.
- `uv run mypy aura tests` passed.
- Task/subagent focused tests passed: `56 passed`.
- Additional quality subset passed: `52 passed`.
- Subagent DAG integration passed: `3 passed`.
- Full `uv run pytest -q` produced `2546 passed, 2 skipped, 1 failed`.

The single full-suite failure was
`tests/test_lifecycle_hooks.py::test_settings_json_can_register_external_hook_command`.
That test passed when run alone and passed in nearby related batches, so it is
treated as order-sensitive flakiness until proven otherwise.

## Non-Goals

- Do not rewrite the whole `Agent` object in one pass.
- Do not change the public CLI shape unless a test proves it is necessary.
- Do not broaden bypass mode silently. Bypass semantics must be explicit and
  tested.
- Do not touch unrelated desktop frontend work already present in the worktree.
- Do not introduce new providers, new UI surfaces, or a new orchestration
  framework.
- Do not make teammate token/tool-progress parity a hard requirement for this
  cycle. This cycle fixes lifecycle state, listing, and model propagation. If
  full teammate progress parity remains incomplete, document it as a current
  limitation.

## Design Principles

- Fix state correctness before polishing architecture.
- Prefer failing tests that pin real behavior over broad refactors.
- Keep changes small and independently verifiable.
- Preserve existing patterns: dataclass task records, hook-based control flow,
  pydantic tool schemas, and pytest-focused verification.
- Make large-task state observable through existing task/team/compact surfaces.

## Workstream 1: Test Stability and Safety Contracts

The first workstream establishes a trustworthy baseline.

1. Reproduce and isolate the lifecycle hook flake.
   - The suspected area is global state or test-order interaction around
     lifecycle hooks, journal session scope, async subprocess handling, or agent
     teardown.
   - The fix should address the root cause, not add sleeps.

2. Pin bypass and subagent permission behavior end-to-end.
   - Existing unit tests verify many hook-level paths.
   - Missing coverage is the full path: parent `Agent` mode/config -> model
     emits `task_create` -> child agent executes a tool -> permission decision,
     tool result, denials, and journal behavior stay consistent.

3. Clarify the bypass contract in docs and comments.
   - Current implementation treats bypass as a broad permission bypass, with
     deny rules still bypass-immune in the permission hook.
   - The design does not change that contract automatically. Any stricter
     safety floor must be introduced explicitly with tests and migration notes.

## Workstream 2: Task State Correctness

Large-task orchestration depends on task state being monotonic and trustworthy.

1. Make terminal transitions idempotent.
   - `TasksStore.mark_completed`, `mark_failed`, and `mark_cancelled` should not
     overwrite an already terminal record.
   - Terminal listeners should fire once per task.
   - This prevents late cancellation, timeout, or cleanup paths from corrupting
     a completed task.

2. Add focused tests for terminal state races.
   - Completed -> cancelled should remain completed.
   - Failed -> completed should remain failed.
   - Cancelled -> failed should remain cancelled.
   - Listener count should remain one.

3. Keep the implementation minimal.
   - A small store-level guard is preferred over scattering checks across every
     caller.

## Workstream 3: Teammate Task Parity

Teams are already modeled as task records, but teammate lifecycle state is not
as complete as subagent state.

1. Extend `task_list` to support `kind="teammate"`.
   - `TaskKind` already includes `teammate`.
   - The tool schema should match that domain model.

2. Ensure teammate records leave `running` when the teammate exits.
   - Runtime natural return not caused by explicit removal should mark the
     teammate task `completed`.
   - Graceful explicit removal through shutdown request/ack should mark the
     teammate task `cancelled`. The user intentionally stopped that teammate;
     it should not look like self-directed work completion.
   - Force kill, parent abort cascade, team delete, and session cleanup should
     mark the teammate task `cancelled`.
   - Runtime task exception should mark the teammate task `failed`, and
     `error` should include the exception type and message.
   - Both `add_member` and `aadd_member` paths must be covered because they
     currently duplicate teammate `TaskRecord` creation and child spawn logic.

3. Thread model selection consistently.
   - In-process teammate spawn should pass the requested `model_name` into the
     subagent factory as `model_spec`.
   - The associated `TaskRecord.model_spec` should show the same effective
     model surface that pane-backed teammates expose via CLI arguments.

4. Document teammate observability limits.
   - Full teammate tool-activity, token, and summary parity is deferred unless
     it falls out naturally from the lifecycle/model fixes.
   - The hard gate for this cycle is: teammate tasks leave `running`,
     `task_list(kind="teammate")` works, and `model_name` flows into
     `model_spec`.
   - Any remaining token/tool progress gap must be documented in README or the
     relevant current-limitations section.

## Workstream 4: Compact and Notification Resilience

Compact must reduce context pressure without erasing the operator's ability to
continue a large task.

1. Add an observed/retrieved marker for task results.
   - `TasksStore` owns observed terminal state. The preferred representation is
     an explicit field on `TaskRecord` if the value is part of the public task
     snapshot, or store-owned metadata if the value is internal only.
   - `task_get` and `task_output` are the only LLM-facing write entrances for
     this marker. When either returns a terminal task, it marks that task as
     observed.
   - Compact only reads the store-owned observed state. It must not infer
     observed status from transcript files, notifications, or history content.
   - Compact should continue reinjecting running tasks and unobserved terminal
     tasks.
   - Observed terminal tasks should not be repeatedly injected forever.

2. Keep task notifications useful under fan-out.
   - When more notifications exist than the cap, keep the latest entries, not
     the oldest entries.
   - Include a count of earlier omitted notifications so the model knows it is
     seeing a tail.

3. Cover mixed large-task behavior.
   - Concurrent children, partial completion, timeout, compact, and later
     `task_get`/`task_list` should compose without losing state.

## Workstream 5: Documentation and Gates

The documentation should match the actual system.

1. Update README subagent and task descriptions.
   - Current README text still describes older MVP behavior in places.
   - It should mention skill inheritance, task depth cap, transcript metadata,
     permissions, model overrides, and current limitations.

2. Add explicit test tiers.
   - Keep `make check` as the full local gate.
   - Add or document fast/integration/real-LLM gates so release validation is
     repeatable.
   - Real LLM checks can remain opt-in because they require credentials.

## Architecture

The design keeps the current architecture and strengthens its state contracts.

- `TasksStore` owns task lifecycle invariants.
- `TasksStore` owns terminal observed/retrieved state for task results.
- `task_get`, `task_output`, and `task_list` own LLM-facing task observation.
- `SubagentFactory` and `TeamManager` own spawn-time model and permission
  inheritance.
- `Context` and compact code own prompt reinjection policy.
- Tests define the safety envelope before production code changes.

This avoids a risky big-bang decomposition of `Agent`. After the reliability
work is green, small extractions such as `SubagentSupervisor` or task lifecycle
helpers can be considered only where they remove real duplication.

## Testing Strategy

Every behavior change should follow test-first implementation:

- Unit tests for `TasksStore` terminal idempotency.
- Tool schema tests for `task_list(kind="teammate")`.
- Team manager/runtime tests for teammate terminal state and model propagation.
- Integration tests for parent bypass/disable-bypass with real `task_create`
  and child tool execution through `FakeChatModel`.
- Compact tests for observed terminal tasks and notification tail behavior.
- Regression test or narrowed reproduction for the lifecycle hook flake.

Verification gates:

- Targeted pytest file or test case after each change.
- `uv run ruff check .`
- `uv run mypy aura tests`
- `uv run pytest -q`
- Optional release gate: real LLM or PTY scripts when credentials/environment
  are available.

## Acceptance Criteria

- Full test suite passes consistently, including the lifecycle hook test.
- Task terminal status is monotonic and listeners fire once.
- Teammate records can be listed and do not stay running after exit.
- Teammate model selection is reflected in spawn behavior and task metadata.
- Compact no longer repeatedly injects observed terminal task results.
- Notification overflow preserves the latest task events.
- README and test-gate documentation match the implemented behavior.
- No unrelated frontend changes are modified or committed.
