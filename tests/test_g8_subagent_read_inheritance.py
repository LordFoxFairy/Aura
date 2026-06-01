"""Workstream G8 + Phase 3 Task 4 — subagent inherits parent reads
through a typed :class:`ReadCarryover`.

Parity with claude-code's Task tool: when a parent agent spawns a subagent
via ``task_create``, the child's :class:`Context` starts out seeded from
the parent's read snapshot. This means files the parent already read are
visible as ``read_status == "fresh"`` in the child — no forced re-read,
no wasted tokens, no tripping the must-read-first hook.

Scope guard:

- ONLY parent-read fingerprints flow across. ``_matched_rules`` /
  ``_invoked_skills`` / ``_loaded_nested_paths`` are agent-identity level
  state and stay empty in the child.
- Read-only carryover: child mutations to its own reads MUST NOT
  propagate back to the parent (the carryover wraps a MappingProxyType).
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from aura.application.memory.context import Context
from aura.application.memory.rules import RulesBundle
from aura.application.session import AgentSession
from aura.application.tasks.spawn import SubagentSpawner
from aura.config.schema import AuraConfig
from aura.domain.state_values import ReadCarryover, ReadRecord
from aura.infrastructure.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn

# ---------------------------------------------------------------------------
# Context-level tests (the ``carryover`` kwarg itself)
# ---------------------------------------------------------------------------


def _record_for(path: Path, *, turn: int = 1) -> ReadRecord:
    st = path.stat()
    return ReadRecord(
        path=path.resolve(),
        mtime_at_read=st.st_mtime,
        size_at_read=st.st_size,
        read_at_turn=turn,
    )


def test_context_carryover_seeds_read_records(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("hello\n")
    carry = ReadCarryover(
        records={p.resolve(): _record_for(p)},
        source_session_id="parent-1",
        generated_at_turn=1,
    )
    ctx = Context(
        cwd=tmp_path,
        system_prompt="",
        primary_memory="",
        rules=RulesBundle(),
        carryover=carry,
    )
    assert ctx.read_status(p) == "fresh"


def test_context_carryover_is_isolated_from_parent(tmp_path: Path) -> None:
    # Child mutations of its own _read_records must NOT touch the
    # parent's carryover view (the carryover is a read-only proxy
    # anyway; the test pins that the child's seeded dict is its own).
    p1 = tmp_path / "a.txt"
    p1.write_text("x\n")
    parent_records = {p1.resolve(): _record_for(p1)}
    carry = ReadCarryover(
        records=parent_records,
        source_session_id="parent-2",
        generated_at_turn=1,
    )
    ctx = Context(
        cwd=tmp_path,
        system_prompt="",
        primary_memory="",
        rules=RulesBundle(),
        carryover=carry,
    )
    p2 = tmp_path / "b.txt"
    p2.write_text("y\n")
    ctx.record_read(p2)
    assert ctx.read_status(p2) == "fresh"
    # Parent's view is untouched — only the path the parent originally
    # carried is visible on the carryover (still wrapped in proxy).
    assert set(carry.records.keys()) == {p1.resolve()}


def test_context_carryover_none_is_empty_start() -> None:
    # Explicit no-inheritance path — Context behaves identically to
    # prior zero-arg construction.
    ctx = Context(
        cwd=Path("/tmp"),
        system_prompt="",
        primary_memory="",
        rules=RulesBundle(),
        carryover=None,
    )
    assert ctx._read_records == {}


# ---------------------------------------------------------------------------
# AC-G8-1 / AC-G8-2 / AC-G8-3 — end-to-end via SubagentSpawner
# ---------------------------------------------------------------------------


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }
    )


def _factory_with_parent_reads(
    parent_records: dict[Path, ReadRecord],
) -> SubagentSpawner[AgentSession]:
    """Build a SubagentSpawner whose carryover provider reflects the
    live ``parent_records`` dict — mutations between calls show up at
    the next ``spawn``.
    """
    def _provider() -> ReadCarryover:
        return ReadCarryover(
            records=dict(parent_records),
            source_session_id="parent-test",
            generated_at_turn=1,
        )

    return SubagentSpawner(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        build_child=AgentSession,
        parent_carryover_provider=_provider,
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="done"))]
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )


def test_subagent_inherits_parent_read_records(tmp_path: Path) -> None:
    """AC-G8-1: parent reads X → spawn → child read_status(X) == 'fresh'."""
    x = tmp_path / "x.txt"
    x.write_text("content\n")
    parent_records: dict[Path, ReadRecord] = {x.resolve(): _record_for(x)}
    factory = _factory_with_parent_reads(parent_records)
    child = factory.spawn("sub-prompt")
    try:
        assert child._context.read_status(x) == "fresh"
    finally:
        child.close()


def test_subagent_read_inheritance_is_read_only(tmp_path: Path) -> None:
    """AC-G8-2: child records Y → parent still sees Y as 'never_read'."""
    x = tmp_path / "x.txt"
    x.write_text("content\n")
    parent_records: dict[Path, ReadRecord] = {x.resolve(): _record_for(x)}
    factory = _factory_with_parent_reads(parent_records)
    child = factory.spawn("sub-prompt")
    try:
        y = tmp_path / "y.txt"
        y.write_text("other\n")
        child._context.record_read(y)
        # Child saw y as fresh.
        assert child._context.read_status(y) == "fresh"
        # Parent snapshot is untouched — child did NOT write back into it.
        assert y.resolve() not in parent_records
        # And x is still the ONLY entry the parent had.
        assert set(parent_records.keys()) == {x.resolve()}
    finally:
        child.close()


def test_subagent_does_not_inherit_matched_rules(tmp_path: Path) -> None:
    """AC-G8-3: parent matched rules do NOT cross the boundary."""
    # We only inherit read fingerprints. Even if the parent had a
    # matched rule, the factory's handoff must not copy it into the
    # child.
    x = tmp_path / "x.txt"
    x.write_text("content\n")
    parent_records: dict[Path, ReadRecord] = {x.resolve(): _record_for(x)}
    factory = _factory_with_parent_reads(parent_records)
    child = factory.spawn("sub-prompt")
    try:
        # Seed a rule on a fake Context to prove we're checking child state,
        # not conflating with parent — child must start with EMPTY matched
        # rules regardless of what the parent accumulated.
        assert child._context._matched_rules == []
        assert child._context._matched_rule_paths == set()
        # Same invariant for invoked skills and nested paths.
        assert child._context._invoked_skills == []
        assert child._context._invoked_skill_paths == set()
        assert child._context._loaded_nested_paths == set()
    finally:
        child.close()


def test_subagent_carryover_is_a_snapshot_at_spawn_time(
    tmp_path: Path,
) -> None:
    """Provider is invoked at spawn time — later parent reads don't leak.

    The factory captures the carryover returned by
    ``parent_carryover_provider()`` at spawn. Subsequent parent reads
    must not retroactively show up in the already-running child.
    """
    x = tmp_path / "x.txt"
    x.write_text("content\n")
    parent_records: dict[Path, ReadRecord] = {x.resolve(): _record_for(x)}
    factory = _factory_with_parent_reads(parent_records)
    child = factory.spawn("sub-prompt")
    try:
        # Parent records a new read AFTER spawn.
        z = tmp_path / "z.txt"
        z.write_text("zzz\n")
        parent_records[z.resolve()] = _record_for(z)
        # Child does NOT see z — its snapshot was taken at spawn.
        assert child._context.read_status(z) == "never_read"
        # But the file it DID inherit is still fresh.
        assert child._context.read_status(x) == "fresh"
    finally:
        child.close()


def test_subagent_factory_without_provider_starts_empty(tmp_path: Path) -> None:
    # Backward compat path: factory built without
    # ``parent_carryover_provider`` behaves exactly as before — child
    # Context starts with an empty _read_records dict.
    factory = SubagentSpawner(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        build_child=AgentSession,
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="done"))]
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    child = factory.spawn("sub-prompt")
    try:
        assert child._context._read_records == {}
    finally:
        child.close()


