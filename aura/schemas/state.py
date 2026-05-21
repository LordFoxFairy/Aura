"""对话生命周期的累计状态 — 随 AgentLoop 共享引用，跨 turn 存活。

Phase 1 — Loop redesign — introduces :class:`LoopSlots` plus the
supporting frozen data types (:class:`TokenStats`,
:class:`SkillRestrictLease`). These replaced the untyped
``state.custom: dict[str, Any]`` scratchpad; Phase 1 Task 7 deleted
the ``custom`` field outright. New transient state MUST land on a
typed :class:`LoopSlots` field — there is no untyped escape hatch.

Type-only references (``Denial``, ``AskerResponse``) are
``TYPE_CHECKING`` imports so the leaf invariant — *nothing under
``aura/schemas`` imports any other ``aura`` module at runtime* —
remains intact (`aura/schemas/__init__.py` enforces this by
construction).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias

from aura.schemas.todos import TodoItem

if TYPE_CHECKING:
    # Real types — only seen by type checkers. Kept out of runtime
    # imports to preserve the ``aura.schemas`` leaf invariant
    # (``aura/schemas/__init__.py`` enforces that nothing under
    # ``aura/schemas`` reaches into other ``aura`` modules at runtime).
    from aura.capabilities.skills_runtime.types import Skill
    from aura.core.permissions.decision import Decision
    from aura.core.permissions.denials import PermissionDenial as Denial
else:
    # Runtime fallbacks — needed because :class:`LoopState` (a stdlib
    # dataclass) is used as a pydantic field type on stateful tools
    # (see ``aura/tools/todo_write.py``); when pydantic introspects the
    # chain ``LoopState → LoopSlots`` it must resolve every annotation
    # name. Aliasing to :data:`Any` here gives pydantic a resolvable
    # name without dragging the real modules into ``aura.schemas`` at
    # runtime. The ``if TYPE_CHECKING`` branch above keeps mypy strict.
    Denial = Any
    Decision = Any
    Skill = Any


# A canonical signature string (`<tool_name>::<json-args>`) used by the
# permission hook's per-turn ResolveOnce dedup cache. Today it's a
# ``str`` produced by ``aura.core.hooks.permission._dedup_key``;
# promoting to a TypeAlias documents the contract without forcing
# every consumer through a wrapper.
PermissionKey: TypeAlias = str

# The cached permission outcome — a ``(decision, feedback)`` tuple. The
# permission hook stores ``user_accept`` / ``user_deny`` outcomes here
# so a same-signature follow-up call within the same turn reuses the
# decision instead of re-prompting. ``feedback`` is the asker's free-form
# rationale string (often empty); kept alongside the decision so the
# audit emit path can replay it identically.
PermissionDedupEntry: TypeAlias = "tuple[Decision, str]"


@dataclass(frozen=True)
class TokenStats:
    """Per-session cumulative + last-turn token usage.

    Owned by :func:`aura.core.hooks.budget.make_usage_tracking_hook` —
    the post-model writer ``replace``-s the slot every turn (Task 3
    migration replaced the legacy untyped scratchpad dict with this
    typed slot; readers in ``transport/wire.py``, ``commands/stats.py``,
    and the REPL bottom toolbar now go through
    ``state.slots.token_stats``). All fields default to zero so the
    empty :class:`TokenStats` is a valid starting state.

    Frozen so a stale snapshot held by the renderer cannot retroactively
    re-attribute tokens to a different turn — the writer ``replace``-s
    on every update.
    """

    last_input_tokens: int = 0
    last_output_tokens: int = 0
    last_cache_read_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    turn_count: int = 0


@dataclass(frozen=True)
class SkillRestrictLease:
    """Public counterpart of the runtime ``_RestrictEntry`` in
    :mod:`aura.capabilities.skills_runtime.restrict`.

    Phase 1 introduces the public type so :class:`LoopSlots` can name
    it; Task 6 migrates the skill loader (writer) and the permission
    hook (reader) onto this type and retires ``_RestrictEntry``.

    Frozen — once a lease is recorded, its install turn and whitelist
    must not mutate (audit consumers must see the value at install
    time, not a later overwrite).

    Multiplicity note: a single skill installs zero or one lease per
    turn, but multiple skills can stack leases concurrently. The
    ``LoopSlots.skill_restrict_leases`` slot therefore holds a
    ``list[SkillRestrictLease]`` (default empty) — each entry expires
    independently when ``LoopState.turn_count`` advances past its
    ``install_turn`` (see :func:`aura.capabilities.skills_runtime.restrict._active_leases`).
    """

    install_turn: int
    tools: frozenset[str]


@dataclass(frozen=True)
class BuddyState:
    """Status-bar pet observer state — mood + last-event metadata.

    Owned by the buddy hooks in :mod:`cli.buddy`. The spec §3.1
    sketch named the slot ``mood: str``; in practice the state machine
    needs three coupled fields (mood label + last event timestamp +
    sticky-worry flag) to avoid flicker and preserve worry across
    turns. Bundling them on one frozen value keeps the spec's "one
    writer per slot" contract while giving the buddy enough room to
    encode its full state machine.

    Defaults match the pre-migration "no events fired yet" shape:
    ``mood="idle"``, ``last_event_ts=0.0``, ``had_recent_error=False``
    — reading :func:`cli.buddy.get_mood` on a fresh
    :class:`LoopState` returns ``"idle"`` as before.
    """

    mood: str = "idle"
    last_event_ts: float = 0.0
    had_recent_error: bool = False


@dataclass(frozen=True)
class ReadRecord:
    """Snapshot of a successful file read — the audit identity for the
    must-read-first invariant.

    Phase 3 §3 promotes the previously-private ``_ReadRecord`` (in
    :mod:`aura.core.memory.context`) to a public schema type so the
    subagent factory can hand parent records to the child via
    :class:`ReadCarryover` without leaking the core type. The four
    fields capture *what was read* (``path``), *the on-disk identity at
    read time* (``mtime_at_read`` + ``size_at_read``), and *when in the
    parent's lifecycle* (``read_at_turn``) — enough to re-validate
    freshness later without needing the original file content.

    Frozen — once recorded, the (path, mtime, size, turn) tuple IS the
    audit fact; rebinding any field would silently rewrite history.
    Task 4 migrates the core's progressive ``_read_records`` map onto
    this same type; until then the two coexist (the core's record stays
    private while consumers go through ``ReadCarryover``).
    """

    path: Path
    mtime_at_read: float
    size_at_read: int
    read_at_turn: int


@dataclass(frozen=True)
class ReadCarryover:
    """Immutable bag of parent-agent read records handed to a subagent.

    Phase 3 §3 — replaces the ad-hoc ``inherited_reads: dict[Path,
    _ReadRecord]`` parameter on :class:`Context` with a typed,
    self-validating contract. The stale-record risk in the legacy dict
    (parent reads ``f.py`` at turn 1, child edits ``f.py`` at turn 10
    after the parent's record went stale) is closed by
    :meth:`is_fresh`, which re-stats the file at access time.

    Frozen — the carryover IS the parent's read snapshot at spawn time;
    any change of mind requires constructing a new value (Task 4 wires
    this through the factory).

    Fields:

    - ``records`` — read-only :class:`~collections.abc.Mapping` of
      resolved ``Path`` → :class:`ReadRecord`. The constructor wraps
      whatever mapping is passed in a :class:`MappingProxyType` so a
      subagent that naively does ``carry.records[p] = ...`` fails with
      :class:`TypeError` instead of silently polluting the parent's
      view. (The wrapper is stored on a frozen dataclass via
      ``object.__setattr__`` in ``__post_init__``.)
    - ``source_session_id`` — parent's session id (``None`` for tests
      and the empty-default carryover). Surfaced in audit / debug
      output so a stale-read prompt can name *which* parent's read is
      no longer trustworthy.
    - ``generated_at_turn`` — parent turn at which the carryover was
      taken. The subagent may use this to phrase a re-read message
      ("parent read this 4 turns ago, file changed since").
    """

    records: Mapping[Path, ReadRecord]
    source_session_id: str | None
    generated_at_turn: int

    def __post_init__(self) -> None:
        # Wrap the records mapping in a read-only proxy so consumers
        # cannot mutate it. ``object.__setattr__`` is the only way to
        # rebind on a frozen dataclass; this happens once at
        # construction, after which the value is permanent.
        if not isinstance(self.records, MappingProxyType):
            normalized = {
                path.expanduser().resolve(strict=False): record
                for path, record in self.records.items()
            }
            object.__setattr__(self, "records", MappingProxyType(normalized))

    def is_fresh(self, path: Path) -> bool:
        """Return True iff ``path`` is in ``records`` AND the on-disk
        file still matches the recorded ``(mtime, size)``.

        Re-stats the file on every call — the cost (one ``stat()``
        syscall per inherited read at edit-prompt time) is acceptable
        because subagent carryover sets are small (≤20 files typical,
        per spec §9). False on missing-record, missing-file, or any
        mismatch; the only path to True is "record present AND file
        unchanged on disk".
        """
        resolved = path.expanduser().resolve(strict=False)
        record = self.records.get(resolved)
        if record is None:
            return False
        try:
            stat = os.stat(resolved)
        except OSError:
            return False
        return (
            stat.st_mtime == record.mtime_at_read
            and stat.st_size == record.size_at_read
        )


@dataclass(frozen=True)
class LoopSlots:
    """Typed slot bag for per-session transient state.

    Phase 1 replaced the prior untyped ``LoopState.custom: dict[str, Any]``
    scratchpad with this typed dataclass; Task 7 deleted the legacy dict
    outright. New transient state MUST land here as a typed field, not on
    a side-channel.

    Spec §3.1 — exactly 11 named slots, one writer per slot. Frozen so
    the slot identity is stable across the turn (the loop refers to
    ``state.slots`` once at turn-start and trusts the value); mutations
    flow through :func:`dataclasses.replace`. Mutable container fields
    (lists, dicts) may still be mutated in place — ``frozen=True`` only
    blocks rebinding the attribute, not in-place ``list.append`` or
    ``list.clear`` on the contained collection. This is intentional and
    matches spec §4 step 1 (``slots.turn_denials.clear()`` at turn
    start).

    Field ownership (one writer per slot — see spec §3.1 migration map):

    - ``token_stats``                  — :func:`make_usage_tracking_hook`
    - ``turn_denials``                 — :func:`make_permission_hook`
    - ``todos``                        — ``todo_write`` tool
    - ``ask_pending``                  — :class:`HookChain`
    - ``perm_dedup_cache``             — :func:`make_permission_hook`
    - ``preserved_invoked_skills``     — :class:`Compactor`
    - ``invoked_skills``               — :meth:`Context.record_skill_invocation`
    - ``consecutive_compact_failures`` — :class:`Compactor`
    - ``active_team``                  — ``/team`` slash commands
    - ``buddy``                        — buddy observer hooks
    - ``skill_restrict_leases``        — skill loader (`install_restrict_lease`)
    """

    token_stats: TokenStats = field(default_factory=TokenStats)
    turn_denials: list[Denial] = field(default_factory=list)
    todos: list[TodoItem] = field(default_factory=list)
    ask_pending: bool = False
    perm_dedup_cache: dict[PermissionKey, PermissionDedupEntry] = field(default_factory=dict)
    preserved_invoked_skills: list[Skill] = field(default_factory=list)
    invoked_skills: list[Skill] = field(default_factory=list)
    consecutive_compact_failures: int = 0
    active_team: str | None = None
    buddy: BuddyState = field(default_factory=BuddyState)
    skill_restrict_leases: list[SkillRestrictLease] = field(default_factory=list)


@dataclass
class LoopState:
    # turn_count 在每次 _invoke_model 入口前 +1，pre_model hook 看到的是"即将开始的第 N 轮"。
    turn_count: int = 0
    # total_tokens_used 由 make_usage_tracking_hook 在 post_model 阶段填入，loop 本身不写。
    total_tokens_used: int = 0
    # Typed slot bag for per-session transient state (Phase 1).
    # Frozen (see :class:`LoopSlots`); writers use :func:`dataclasses.replace`
    # to swap fields. The attribute itself is rebound
    # (``state.slots = replace(state.slots, ...)``), so :class:`LoopState`
    # stays a non-frozen dataclass.
    #
    # Phase 1 Task 7 deleted the prior untyped ``custom: dict[str, Any]``
    # scratchpad. New transient state MUST land on a typed slot here;
    # there is no untyped escape hatch. The
    # ``tests/test_no_state_custom.py`` invariant test guards against
    # silent re-introduction.
    slots: LoopSlots = field(default_factory=LoopSlots)

    def reset(self) -> None:
        # 必须原地 mutate：AgentLoop 持有同一个 LoopState 引用，新建对象不会被 loop 感知。
        # ``slots`` is intentionally NOT reset here — typed slot owners
        # (Agent.clear_session, Loop.run_turn, todo_write, etc.) clear
        # their own slots in place at the appropriate lifecycle event.
        # Wiping every slot here would destroy state owners legitimately
        # carry across /clear (e.g. cumulative token stats for the
        # status bar).
        self.turn_count = 0
        self.total_tokens_used = 0
