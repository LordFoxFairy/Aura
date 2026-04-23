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
- Skills ARE inherited: the parent's pre-loaded :class:`SkillRegistry` is
  handed through to the child's Agent constructor so the subagent has exact
  parity with parent skill set without a redundant disk scan.
- MCP servers ARE inherited at the config level: child's ``mcp_servers``
  list matches parent's. Each Agent still runs its own ``aconnect`` —
  langchain-mcp-adapters spawns a fresh session per ``get_tools`` call, so
  parent and subagent end up with INDEPENDENT MCP connections to the same
  servers. That's the simplest + correct-enough design.
- Recursion guard: ``task_create`` / ``task_output`` are stripped from the
  child's tools.enabled — a subagent CANNOT spawn further subagents in
  0.5.x (dispatch not wired into child Agents yet). The inspection tools
  ``task_get`` / ``task_list`` / ``task_stop`` ARE inherited: they operate
  on the shared :class:`TasksStore` held by the parent Agent, so a
  subagent can poll its siblings. Each child gets its own TasksStore
  instance in practice (spawn builds a fresh Agent, which builds a fresh
  store), so ``task_get("sibling-id")`` from a subagent will see an empty
  store and return ``unknown task_id`` — that's intentional: siblings
  aren't visible across the parent/child boundary, and we don't want to
  leak parent state into the child. Keeping the tools enabled means the
  LLM inside the subagent doesn't hallucinate "maybe task_get exists"
  without a way to verify.
- Parent hooks (permission, budget) are NOT inherited. Safety hooks
  (bash_safety + must_read_first) are re-installed inside the child's
  ``__init__`` via the same code path as the parent, so the subagent is
  safe even though parent-authored hooks don't cross the boundary.

Storage defaults to an in-memory sqlite connection so the subagent's
transcript doesn't pollute the parent's on-disk session DB. Tests inject a
``storage_factory`` explicitly; production wiring uses the default.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from aura.config.schema import AuraConfig, ToolsConfig
from aura.core import llm
from aura.core.memory.context import _ReadRecord
from aura.core.persistence.storage import SessionStorage
from aura.core.skills import SkillRegistry
from aura.core.tasks.agent_types import get_agent_type

if TYPE_CHECKING:
    from aura.core.agent import Agent


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
        self._parent_config = parent_config
        self._parent_model_spec = parent_model_spec
        self._parent_skills = parent_skills
        self._parent_read_records_provider = parent_read_records_provider
        self._model_factory = model_factory
        self._storage_factory = storage_factory or _default_storage

    def spawn(
        self,
        prompt: str,
        allowed_tools: list[str] | None = None,
        *,
        agent_type: str = "general-purpose",
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
                if name not in {"task_create", "task_output"}
                and (allowed_tools is None or name in allowed_tools)
                and (effective_allow is None or name in effective_allow)
            ]
        )
        child_cfg = self._parent_config.model_copy(
            update={"tools": child_tools}
        )
        if self._model_factory is not None:
            model = self._model_factory()
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
        return Agent(
            config=child_cfg,
            model=model,
            storage=storage,
            session_id="subagent",
            pre_loaded_skills=self._parent_skills,
            system_prompt_suffix=type_def.system_prompt_suffix,
            inherited_reads=inherited_reads,
        )
