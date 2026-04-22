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
  0.5.x (dispatch not wired into child Agents yet).
- Parent hooks (permission, budget) are NOT inherited. Safety hooks
  (bash_safety + must_read_first) are re-installed inside the child's
  ``__init__`` via the same code path as the parent, so the subagent is
  safe even though parent-authored hooks don't cross the boundary.

Storage defaults to an in-memory sqlite connection so the subagent's
transcript doesn't pollute the parent's on-disk session DB. Tests inject a
``storage_factory`` explicitly; production wiring uses the default.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from aura.config.schema import AuraConfig, ToolsConfig
from aura.core import llm
from aura.core.persistence.storage import SessionStorage
from aura.core.skills import SkillRegistry

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
        model_factory: Callable[[], BaseChatModel] | None = None,
        storage_factory: Callable[[], SessionStorage] | None = None,
    ) -> None:
        self._parent_config = parent_config
        self._parent_model_spec = parent_model_spec
        self._parent_skills = parent_skills
        self._model_factory = model_factory
        self._storage_factory = storage_factory or _default_storage

    def spawn(
        self, prompt: str, allowed_tools: list[str] | None = None,
    ) -> Agent:
        # Import locally to avoid a circular import: Agent's module pulls in
        # aura.tools.task_create, which pulls in this factory.
        from aura.core.agent import Agent

        # Clone the parent config but strip subagent-forbidden tools — a
        # subagent can't spawn further subagents in 0.5.x (dispatch not
        # wired into child Agents yet). MCP servers ARE inherited so the
        # subagent has parity with parent's external tool set.
        child_tools = ToolsConfig(
            enabled=[
                name for name in self._parent_config.tools.enabled
                if name not in {"task_create", "task_output"}
                and (allowed_tools is None or name in allowed_tools)
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
        return Agent(
            config=child_cfg,
            model=model,
            storage=storage,
            session_id="subagent",
            pre_loaded_skills=self._parent_skills,
        )
