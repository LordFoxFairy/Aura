"""对话生命周期的累计状态 — 随 AgentLoop 共享引用，跨 turn 存活。

Phase 1 — Loop redesign — introduces :class:`LoopSlots` plus the
supporting frozen data types (:class:`TokenStats`,
:class:`SkillRestrictLease`). These replace the untyped
``state.custom: dict[str, Any]`` scratchpad. The ``custom`` field
stays in :class:`LoopState` for now; Tasks 3-7 migrate consumers
key-by-key, then Task 7 deletes it.

Type-only references (``Denial``, ``AskerResponse``) are
``TYPE_CHECKING`` imports so the leaf invariant — *nothing under
``aura/schemas`` imports any other ``aura`` module at runtime* —
remains intact (`aura/schemas/__init__.py` enforces this by
construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

from aura.schemas.todos import TodoItem

if TYPE_CHECKING:
    # Real types — only seen by type checkers. Kept out of runtime
    # imports to preserve the ``aura.schemas`` leaf invariant
    # (``aura/schemas/__init__.py`` enforces that nothing under
    # ``aura/schemas`` reaches into other ``aura`` modules at runtime).
    from aura.core.hooks.permission import AskerResponse
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
    AskerResponse = Any


# A canonical signature string (`<tool_name>::<json-args>`) used by the
# permission hook's per-turn ResolveOnce dedup cache. Today it's a
# ``str`` produced by ``aura.core.hooks.permission._dedup_key``;
# promoting to a TypeAlias documents the contract without forcing
# every consumer through a wrapper.
PermissionKey: TypeAlias = str


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
    :mod:`aura.core.skills.restrict`.

    Phase 1 introduces the public type so :class:`LoopSlots` can name
    it; Task 6 migrates the skill loader (writer) and the permission
    hook (reader) onto this type and retires ``_RestrictEntry``.

    Frozen — once a lease is recorded, its install turn and whitelist
    must not mutate (audit consumers must see the value at install
    time, not a later overwrite).
    """

    install_turn: int
    tools: frozenset[str]


@dataclass(frozen=True)
class LoopSlots:
    """Typed replacement for ``LoopState.custom: dict[str, Any]``.

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
    - ``mood``                         — buddy ``pre_model`` hook
    - ``skill_restrict_lease``         — skill loader (`install_restrict_lease`)
    """

    token_stats: TokenStats = field(default_factory=TokenStats)
    turn_denials: list[Denial] = field(default_factory=list)
    todos: list[TodoItem] = field(default_factory=list)
    ask_pending: bool = False
    perm_dedup_cache: dict[PermissionKey, AskerResponse] = field(default_factory=dict)
    preserved_invoked_skills: list[str] = field(default_factory=list)
    invoked_skills: list[str] = field(default_factory=list)
    consecutive_compact_failures: int = 0
    active_team: str | None = None
    mood: str = "neutral"
    skill_restrict_lease: SkillRestrictLease | None = None


@dataclass
class LoopState:
    # turn_count 在每次 _invoke_model 入口前 +1，pre_model hook 看到的是"即将开始的第 N 轮"。
    turn_count: int = 0
    # total_tokens_used 由 make_usage_tracking_hook 在 post_model 阶段填入，loop 本身不写。
    total_tokens_used: int = 0
    # custom — per-session transient scratchpad for hooks / tools.
    #
    # Legitimate keys currently in use (contract lock — any new slot
    # here MUST land with a matching docstring update and a justified
    # owner, not silently):
    #
    # - denials sink: :data:`aura.core.permissions.denials.DENIALS_SINK_KEY`
    #   (G5). Shared list reference between the permission hook
    #   (writer) and :class:`aura.core.agent.Agent` (owner + reader
    #   via ``last_turn_denials()``).
    # - ``"todos"`` — populated by ``todo_write`` tool; read by
    #   compact / system prompt assembly.
    #
    # The token-usage scratchpad key was migrated out of this dict by
    # Phase 1 / Task 3; it now lives on the typed
    # :class:`LoopSlots.token_stats` slot.
    #
    # Do NOT add new transient slots for one-shot hook→loop signalling:
    # G4 removed the last per-call decision side-channel in favor of
    # :class:`aura.core.hooks.PreToolOutcome` direct-return. New
    # lifecycle data should ride typed return values, not a dict slot
    # here.
    #
    # Phase 1 deprecation note: ``custom`` is being migrated to
    # :class:`LoopSlots` key-by-key (Tasks 3-6) and removed in Task 7.
    # New code MUST NOT add keys here; use a typed slot on
    # :class:`LoopSlots` instead.
    custom: dict[str, Any] = field(default_factory=dict)
    # Typed slot bag — replaces ``custom`` as Tasks 3-6 migrate
    # consumers key-by-key. Frozen (see :class:`LoopSlots`); writers
    # use :func:`dataclasses.replace` to swap fields. The attribute
    # itself is rebound (``state.slots = replace(state.slots, ...)``),
    # so :class:`LoopState` stays a non-frozen dataclass.
    slots: LoopSlots = field(default_factory=LoopSlots)

    def reset(self) -> None:
        # 必须原地 mutate：AgentLoop 持有同一个 LoopState 引用，新建对象不会被 loop 感知。
        self.turn_count = 0
        self.total_tokens_used = 0
        self.custom.clear()
