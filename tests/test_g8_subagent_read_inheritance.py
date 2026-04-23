"""Workstream G8 — subagent inherits parent ``Context._read_records``.

Parity with claude-code's Task tool: when a parent agent spawns a subagent
via ``task_create``, the child's :class:`Context` starts out with a shallow
copy of the parent's ``_read_records`` map. This means files the parent
already read are visible as ``read_status == "fresh"`` in the child — no
forced re-read, no wasted tokens, no tripping the must-read-first hook.

Scope guard:

- ONLY ``_read_records`` is inherited. ``_matched_rules`` /
  ``_invoked_skills`` / ``_loaded_nested_paths`` are agent-identity level
  state and stay empty in the child.
- Shallow copy: child mutations to its own ``_read_records`` MUST NOT
  propagate back to the parent (dict is not shared).
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from aura.config.schema import AuraConfig
from aura.core.memory.context import Context, _ReadRecord
from aura.core.memory.rules import RulesBundle
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from tests.conftest import FakeChatModel, FakeTurn

# ---------------------------------------------------------------------------
# Context-level tests (the ``inherited_reads`` kwarg itself)
# ---------------------------------------------------------------------------


def test_context_inherited_reads_copied_into_read_records(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("hello\n")
    st = p.stat()
    parent_records: dict[Path, _ReadRecord] = {
        p.resolve(): _ReadRecord(mtime=st.st_mtime, size=st.st_size, partial=False),
    }
    ctx = Context(
        cwd=tmp_path,
        system_prompt="",
        primary_memory="",
        rules=RulesBundle(),
        inherited_reads=parent_records,
    )
    assert ctx.read_status(p) == "fresh"


def test_context_inherited_reads_is_shallow_copy(tmp_path: Path) -> None:
    # Mutating the child's _read_records must NOT touch the parent's dict.
    p1 = tmp_path / "a.txt"
    p1.write_text("x\n")
    st1 = p1.stat()
    parent_records: dict[Path, _ReadRecord] = {
        p1.resolve(): _ReadRecord(mtime=st1.st_mtime, size=st1.st_size, partial=False),
    }
    ctx = Context(
        cwd=tmp_path,
        system_prompt="",
        primary_memory="",
        rules=RulesBundle(),
        inherited_reads=parent_records,
    )
    p2 = tmp_path / "b.txt"
    p2.write_text("y\n")
    ctx.record_read(p2)
    assert ctx.read_status(p2) == "fresh"
    # Parent dict is untouched.
    assert p2.resolve() not in parent_records
    assert len(parent_records) == 1


def test_context_inherited_reads_none_is_empty_start() -> None:
    # Explicit no-inheritance path — Context behaves identically to prior
    # zero-arg construction.
    ctx = Context(
        cwd=Path("/tmp"),
        system_prompt="",
        primary_memory="",
        rules=RulesBundle(),
        inherited_reads=None,
    )
    assert ctx._read_records == {}


# ---------------------------------------------------------------------------
# AC-G8-1 / AC-G8-2 / AC-G8-3 — end-to-end via SubagentFactory
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
    parent_reads: dict[Path, _ReadRecord],
) -> SubagentFactory:
    return SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        parent_read_records_provider=lambda: parent_reads,
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="done"))]
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )


def test_subagent_inherits_parent_read_records(tmp_path: Path) -> None:
    """AC-G8-1: parent reads X → spawn → child read_status(X) == 'fresh'."""
    x = tmp_path / "x.txt"
    x.write_text("content\n")
    st = x.stat()
    parent_reads: dict[Path, _ReadRecord] = {
        x.resolve(): _ReadRecord(mtime=st.st_mtime, size=st.st_size, partial=False),
    }
    factory = _factory_with_parent_reads(parent_reads)
    child = factory.spawn("sub-prompt")
    try:
        assert child._context.read_status(x) == "fresh"
    finally:
        child.close()


def test_subagent_read_inheritance_is_read_only(tmp_path: Path) -> None:
    """AC-G8-2: child records Y → parent still sees Y as 'never_read'."""
    x = tmp_path / "x.txt"
    x.write_text("content\n")
    st = x.stat()
    parent_reads: dict[Path, _ReadRecord] = {
        x.resolve(): _ReadRecord(mtime=st.st_mtime, size=st.st_size, partial=False),
    }
    factory = _factory_with_parent_reads(parent_reads)
    child = factory.spawn("sub-prompt")
    try:
        y = tmp_path / "y.txt"
        y.write_text("other\n")
        child._context.record_read(y)
        # Child saw y as fresh.
        assert child._context.read_status(y) == "fresh"
        # Parent snapshot is untouched — child did NOT write back into it.
        assert y.resolve() not in parent_reads
        # And x is still the ONLY entry the parent had.
        assert set(parent_reads.keys()) == {x.resolve()}
    finally:
        child.close()


def test_subagent_does_not_inherit_matched_rules(tmp_path: Path) -> None:
    """AC-G8-3: parent matched rules do NOT cross the boundary."""
    # We only inherit _read_records. Even if the parent had a matched rule,
    # the factory's handoff must not copy it into the child.
    x = tmp_path / "x.txt"
    x.write_text("content\n")
    st = x.stat()
    parent_reads: dict[Path, _ReadRecord] = {
        x.resolve(): _ReadRecord(mtime=st.st_mtime, size=st.st_size, partial=False),
    }
    factory = _factory_with_parent_reads(parent_reads)
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


def test_subagent_inherited_reads_is_a_snapshot_at_spawn_time(
    tmp_path: Path,
) -> None:
    """Provider is invoked at spawn time — later parent reads don't leak.

    The factory captures the dict returned by
    ``parent_read_records_provider()`` at spawn. Subsequent mutations to the
    parent's live dict must not retroactively show up in the child.
    """
    x = tmp_path / "x.txt"
    x.write_text("content\n")
    st = x.stat()
    parent_reads: dict[Path, _ReadRecord] = {
        x.resolve(): _ReadRecord(mtime=st.st_mtime, size=st.st_size, partial=False),
    }
    factory = _factory_with_parent_reads(parent_reads)
    child = factory.spawn("sub-prompt")
    try:
        # Parent records a new read AFTER spawn.
        z = tmp_path / "z.txt"
        z.write_text("zzz\n")
        stz = z.stat()
        parent_reads[z.resolve()] = _ReadRecord(
            mtime=stz.st_mtime, size=stz.st_size, partial=False,
        )
        # Child does NOT see z — its snapshot was taken at spawn.
        assert child._context.read_status(z) == "never_read"
        # But the file it DID inherit is still fresh.
        assert child._context.read_status(x) == "fresh"
    finally:
        child.close()


def test_subagent_factory_without_provider_starts_empty(tmp_path: Path) -> None:
    # Backward compat path: factory built without ``parent_read_records_provider``
    # behaves exactly as before — child Context starts with an empty
    # _read_records dict.
    factory = SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
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


