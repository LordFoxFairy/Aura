"""SubagentFactory — build an :class:`Agent` instance per task.

Inheritance rules (matches claude-code's Task tool):

- Own :class:`LoopState`, own :class:`Context`, own storage. The subagent
  MUST NOT observe or mutate the parent's history; spawning is not
  conversation continuation.
- Shares the parent's :class:`AuraConfig` (providers / router) + the parent's
  model spec, so the router alias (e.g. ``default``) resolves to the same
  concrete model. Resolving a fresh model per subagent is deliberate — the
  chat model classes are stateful (seen_bound_tools etc.) and sharing one
  across agents risks cross-talk.
- Round 7R per-spawn override: ``spawn(model_spec=...)`` swaps the
  inherited spec for this child only. ``None`` (the default) keeps the
  inherited parent spec; an explicit string runs through
  :func:`llm.make_model_for_spec` so router aliases (``"haiku"``) and
  ``provider:model`` form both resolve. Unknown specs raise
  :class:`UnknownModelSpecError` from :meth:`validate_model_spec` — the
  task_create tool calls this BEFORE creating a TaskRecord so a typo
  doesn't strand an orphan record in ``running``.
- Skills ARE inherited: the parent's pre-loaded :class:`SkillRegistry` is
  handed through to the child's Agent constructor so the subagent has exact
  parity with parent skill set without a redundant disk scan.
- MCP servers ARE inherited at the config level: child's ``mcp_servers``
  list matches parent's. Each Agent still runs its own ``aconnect`` —
  langchain-mcp-adapters spawns a fresh session per ``get_tools`` call, so
  parent and subagent end up with INDEPENDENT MCP connections to the same
  servers. That's the simplest + correct-enough design.
- Recursion guard: ``task_create`` / ``task_output`` / ``task_get`` /
  ``task_list`` / ``task_stop`` are stripped from the child's
  tools.enabled — a subagent CANNOT spawn or inspect further subagents
  in 0.5.x (dispatch not wired into child Agents yet).
- Parent budget hooks are NOT inherited. Safety hooks (bash_safety +
  must_read_first) are re-installed inside the child's ``__init__`` via
  the same code path as the parent, so the subagent is safe even though
  parent-authored hooks don't cross the boundary.
- Permission IS inherited — via a freshly-built hook that reuses the
  parent's :class:`RuleSet` + :class:`SafetyPolicy` + live mode +
  optional deny_rules / ask_rules, but hands the child a private
  :class:`SessionRuleSet` and a :class:`SubagentAutoDenyAsker` (C1,
  parity with claude-code's ``shouldAvoidPermissionPrompts: true`` —
  subagents have no UI so any would-be ask path silently denies).
  Plan / accept_edits modes don't make sense on a non-interactive
  subagent; those collapse to ``default``. ``bypass`` is the one mode
  that *does* inherit verbatim.

Storage defaults to an in-memory sqlite connection so the subagent's
transcript doesn't pollute the parent's on-disk session DB. Tests inject a
``storage_factory`` explicitly; production wiring uses the default.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from langchain_core.language_models import BaseChatModel

from aura.config.schema import AuraConfig, ToolsConfig
from aura.core import llm
from aura.core.hooks import HookChain
from aura.core.hooks.permission import make_permission_hook
from aura.core.memory.context import _ReadRecord
from aura.core.permissions.mode import Mode
from aura.core.permissions.safety import DEFAULT_SAFETY, SafetyPolicy
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.core.permissions.subagent_asker import SubagentAutoDenyAsker
from aura.core.persistence.storage import SessionStorage
from aura.core.skills import SkillRegistry
from aura.core.tasks.agent_types import get_agent_type

if TYPE_CHECKING:
    from aura.core.agent import Agent

# Singleton — stateless; sharing one instance across every subagent +
# every tool call is cheap and matches the "no I/O" contract.
_SUBAGENT_AUTO_DENY_ASKER = SubagentAutoDenyAsker()

# Tools the child must NEVER see — recursion guard + inspection-of-self
# isolation. Stripping ``task_get`` / ``task_list`` / ``task_stop`` from
# the child mirrors claude-code's "subagents cannot dispatch further
# subagents" rule + closes the loophole where a child could observe its
# own running record (the child's TasksStore is fresh + empty, so the
# inspection tools would just return ``unknown task_id`` for every id;
# stripping them is cleaner than letting the LLM hallucinate state that
# isn't there).
_RECURSION_BAN: frozenset[str] = frozenset({
    "task_create",
    "task_output",
    "task_get",
    "task_list",
    "task_stop",
})


def _default_storage() -> SessionStorage:
    # sqlite3.connect(":memory:") works; Path(":memory:").parent == Path(".")
    # which already exists so the mkdir in SessionStorage is a no-op.
    return SessionStorage(Path(":memory:"))


class SubagentFactory:
    """Create a standalone Agent for a single subagent run."""

    def __init__(
        self,
        parent_config: AuraConfig,
        parent_model_spec: str,
        *,
        parent_skills: SkillRegistry | None = None,
        parent_read_records_provider: (
            Callable[[], Mapping[Path, _ReadRecord]] | None
        ) = None,
        parent_ruleset: RuleSet | None = None,
        parent_safety: SafetyPolicy | None = None,
        parent_mode_provider: Callable[[], str] | None = None,
        parent_session: SessionRuleSet | None = None,
        parent_deny_rules: RuleSet | None = None,
        parent_ask_rules: RuleSet | None = None,
        model_factory: Callable[[], BaseChatModel] | None = None,
        storage_factory: Callable[[], SessionStorage] | None = None,
    ) -> None:
        # ``parent_read_records_provider`` — called at each ``spawn`` to
        # snapshot the parent Agent's live ``Context._read_records`` map.
        # Threaded through to the child's :class:`Context` as
        # ``inherited_reads`` so files the parent already read show up as
        # ``read_status == "fresh"`` in the child (Workstream G8). ``None``
        # disables inheritance — child starts with an empty read map
        # (legacy behavior, kept for the handful of tests that build a
        # factory without a parent Agent reference).
        #
        # ``parent_ruleset`` / ``parent_safety`` / ``parent_mode_provider``
        # (C1) — the triplet that lets ``spawn`` assemble a permission
        # hook identical in spirit to the parent's. ``None`` on all three
        # disables the child permission hook (legacy path for tests that
        # build a factory without any permission wiring). ``parent_session``
        # is NOT inherited — it's recorded here purely so a caller that
        # wants to verify "child.session is fresh / not parent.session"
        # has something to compare against. ``spawn`` ignores it.
        #
        # ``parent_deny_rules`` / ``parent_ask_rules`` (Round 3C) — the
        # parent's deny + ask layered rulesets. Threaded into
        # :func:`make_permission_hook` so the child enforces the SAME
        # deny / ask matrix. ``None`` means "inherit no extra layered
        # rules" — the make_permission_hook signature accepts only
        # ``rules=`` today, so deny / ask flow through that same param
        # (see _build_permission_hook).
        self._parent_config = parent_config
        self._parent_model_spec = parent_model_spec
        self._parent_skills = parent_skills
        self._parent_read_records_provider = parent_read_records_provider
        self._parent_ruleset = parent_ruleset
        self._parent_safety = parent_safety
        self._parent_mode_provider = parent_mode_provider
        self._parent_session = parent_session
        self._parent_deny_rules = parent_deny_rules
        self._parent_ask_rules = parent_ask_rules
        self._model_factory = model_factory
        self._storage_factory = storage_factory or _default_storage

    @property
    def parent_model_spec(self) -> str:
        """Read-only view of the inherited parent spec.

        Surfaced so ``task_create`` can read the spec the child WOULD
        run on without an override + pin it onto the TaskRecord at
        create time. Returning the stored value (rather than re-routing
        through ``cfg.router``) keeps the property a cheap dict get.
        """
        return self._parent_model_spec

    def validate_model_spec(self, spec: str) -> None:
        """Raise :class:`UnknownModelSpecError` if ``spec`` cannot resolve.

        Called by ``task_create`` BEFORE the TaskRecord is created so a
        typo / unknown alias / unknown provider surfaces as a clean
        ToolError rather than stranding an orphan record in
        ``running``. Pure validation: no side effects, no SDK
        construction.
        """
        # ``llm.resolve`` is sync + does only dict lookups; no SDK
        # touch. It raises :class:`UnknownModelSpecError` on any
        # unknown alias / unknown provider.
        llm.resolve(spec, cfg=self._parent_config)

    def spawn(
        self,
        prompt: str,
        allowed_tools: list[str] | None = None,
        *,
        agent_type: str = "general-purpose",
        task_id: str | None = None,
        model_spec: str | None = None,
    ) -> Agent:
        # Import locally to avoid a circular import: Agent's module pulls in
        # aura.tools.task_create, which pulls in this factory.
        from aura.core.agent import Agent

        # Resolve the subagent flavor first — any unknown name raises
        # ValueError with the valid set, which the calling tool
        # (``task_create``) surfaces to the LLM as a ToolError.
        type_def = get_agent_type(agent_type)

        # Build the effective allowlist. Layered precedence:
        #   1. general-purpose (empty ``type_def.tools``) → inherit parent
        #      tool set unchanged, just strip the recursion-guard tools.
        #   2. Restricted type → intersect with parent's enabled set. If any
        #      declared allowed-tool is missing from the parent, raise —
        #      silently dropping would hand the subagent a broken prompt
        #      (the suffix promises tools the child can't see).
        parent_enabled = list(self._parent_config.tools.enabled)
        if type_def.tools:
            missing = type_def.tools - set(parent_enabled)
            if missing:
                raise ValueError(
                    f"agent_type {agent_type!r} requires tools "
                    f"{sorted(missing)} but parent has not enabled them; "
                    "enable them on the parent config or pick a different "
                    "agent_type."
                )
            effective_allow: set[str] | None = set(type_def.tools)
        else:
            effective_allow = None  # inherit-all sentinel

        # Clone the parent config but strip subagent-forbidden tools — a
        # subagent can't spawn further subagents in 0.5.x (dispatch not
        # wired into child Agents yet). MCP servers ARE inherited so the
        # subagent has parity with parent's external tool set.
        # ``allowed_tools`` (legacy kwarg) and ``effective_allow`` (derived
        # from agent_type) both act as narrowing filters; both must pass.
        child_tools = ToolsConfig(
            enabled=[
                name for name in parent_enabled
                if name not in _RECURSION_BAN
                and (allowed_tools is None or name in allowed_tools)
                and (effective_allow is None or name in effective_allow)
            ]
        )
        child_cfg = self._parent_config.model_copy(
            update={"tools": child_tools}
        )
        # Per-spawn model resolution. Precedence:
        #   1. Explicit ``model_factory`` (test injection — full bypass).
        #   2. ``model_spec`` kwarg (Round 7R) → make_model_for_spec.
        #   3. Inherited ``parent_model_spec`` → resolve + create.
        if self._model_factory is not None:
            model = self._model_factory()
        elif model_spec is not None:
            # ``make_model_for_spec`` runs through llm.resolve + llm.create
            # — same path as the parent's startup model build. Raises
            # :class:`UnknownModelSpecError` on bad spec; propagates so
            # the caller (run_task) marks the record failed. task_create
            # validates BEFORE creating the record so the failure here
            # is reserved for genuine post-create races (config swapped
            # mid-run, etc.).
            model = llm.make_model_for_spec(model_spec, self._parent_config)
        else:
            provider, model_name = llm.resolve(
                self._parent_model_spec, cfg=self._parent_config,
            )
            model = llm.create(provider, model_name)
        storage = self._storage_factory()
        # Snapshot the parent's read records RIGHT NOW. ``dict(...)`` over
        # whatever the provider returns pins a shallow copy at spawn time so
        # subsequent parent reads don't retroactively enter the child's map
        # (and child record_read calls don't write back into the parent's
        # live dict). _ReadRecord is frozen, so value-level sharing is
        # harmless. None provider → None passed through → child starts
        # empty, matching prior behavior.
        inherited_reads: dict[Path, _ReadRecord] | None
        if self._parent_read_records_provider is not None:
            inherited_reads = dict(self._parent_read_records_provider())
        else:
            inherited_reads = None

        # C1 — assemble a permission hook for the child. The child needs
        # the parent's rules (so default-allows + user rules propagate)
        # and the parent's safety policy (so protected paths still
        # block), but gets a FRESH ``SessionRuleSet`` — session rules
        # approved inside the child MUST NOT leak back to the parent —
        # and the auto-deny asker (any would-be prompt silently denies).
        #
        # Mode inheritance rule: ``bypass`` rides through (if the user
        # explicitly opted into bypass they meant it for the whole tree),
        # ``plan`` / ``accept_edits`` collapse to ``default`` (the child
        # has no way to exit plan mode interactively; the parent's plan
        # gate already blocked whatever spawned this subagent if it was
        # meant to be dry-run), and any other value maps to ``default``.
        #
        # If the factory was built without permission wiring (any of the
        # three params is ``None``), skip the hook entirely — existing
        # tests / SDK callers that never set up permissions get the
        # legacy zero-hook behaviour.
        child_session = SessionRuleSet()
        child_hooks: HookChain | None = None
        child_mode: str = "default"
        if (
            self._parent_ruleset is not None
            and self._parent_safety is not None
            and self._parent_mode_provider is not None
        ):
            parent_mode = self._parent_mode_provider()
            child_mode = "bypass" if parent_mode == "bypass" else "default"
            # Freeze the resolved child mode into the hook's closure.
            # Re-reading ``parent_mode_provider`` at hook fire time would
            # let a mid-turn parent mode flip (shift+tab) bleed into the
            # child — the parity contract says mode is decided at spawn.
            _resolved_mode: Mode = "bypass" if parent_mode == "bypass" else "default"
            perm_hook = make_permission_hook(
                asker=_SUBAGENT_AUTO_DENY_ASKER,
                session=child_session,
                rules=self._parent_ruleset,
                project_root=self._parent_config.resolved_storage_path().parent,
                mode=_resolved_mode,
                safety=self._parent_safety or DEFAULT_SAFETY,
            )
            child_hooks = HookChain(pre_tool=[perm_hook])

        # Storage race fix (audit Tier S): every subagent MUST have its own
        # session_id. The old literal ``"subagent"`` made two concurrent
        # children share a storage key — ``SessionStorage.save`` is
        # DELETE-then-INSERT, so whichever child flushed last wiped the
        # other's transcript. It also made ``session="subagent"`` in every
        # journal event, destroying forensic traceability.
        #
        # Prefer the caller-supplied ``task_id`` — it ties the session to
        # the observable :class:`TaskRecord` so ``/tasks`` + journal +
        # storage all line up. Fallback to a short uuid for legacy callers
        # (``factory.spawn("prompt")`` without kwargs) so the invariant
        # "every child has a unique session" holds unconditionally.
        # Storage layer parameterises the string safely, so no sanitization
        # is needed on task_id.
        child_session_id = (
            f"subagent-{task_id}"
            if task_id is not None
            else f"subagent-{uuid4().hex[:8]}"
        )
        return Agent(
            config=child_cfg,
            model=model,
            storage=storage,
            hooks=child_hooks,
            session_id=child_session_id,
            session_rules=child_session,
            pre_loaded_skills=self._parent_skills,
            system_prompt_suffix=type_def.system_prompt_suffix,
            inherited_reads=inherited_reads,
            mode=child_mode,
        )
