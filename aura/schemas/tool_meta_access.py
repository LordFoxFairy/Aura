"""Unified metadata-access helper — projects ``ToolMetadata`` to a dict view.

Every Aura-registered tool sets ``aura_metadata: ToolMetadata`` (enforced
by ``ToolRegistry.register`` at Task 4). This helper exposes the typed
fields through a flat ``dict`` so legacy callsites (loop, hooks, CLI
renderer) keep working with ``meta_dict(tool).get(...)`` patterns.

The legacy ``tool.metadata`` fallback is gone — Aura no longer writes to
``BaseTool.metadata`` (LangChain still owns that field, but Aura's
contract is exclusively ``aura_metadata``). A tool without
``aura_metadata`` returns ``{}`` so ``.get(...)`` is always safe.

Lives under ``aura.schemas`` rather than ``aura.tools`` so consumers
inside ``aura.core.permissions`` (which feeds ``aura.tools`` indirectly
through agent wiring) can import it without triggering an import cycle
through ``aura.tools.__init__``.
"""

from __future__ import annotations

from typing import Any

from aura.schemas.tool import ToolMetadata


def meta_dict(tool: Any) -> dict[str, Any]:
    """Return tool metadata as a legacy-shaped dict.

    Reads from ``tool.aura_metadata`` (the typed ``ToolMetadata`` set by
    every Aura-registered tool) and projects its fields back to the
    legacy dict keys, including ``max_result_size_chars`` (the budget
    hook's per-tool truncation override) and ``is_search_command``
    (derived from ``capability_flags``).

    Returns ``{}`` when ``aura_metadata`` is missing so
    ``meta_dict(tool).get(...)`` is always safe.
    """
    aura_meta = getattr(tool, "aura_metadata", None)
    if isinstance(aura_meta, ToolMetadata):
        return {
            "is_read_only": aura_meta.is_read_only,
            "is_destructive": aura_meta.is_destructive,
            "is_concurrency_safe": aura_meta.is_concurrency_safe,
            "rule_matcher": aura_meta.rule_matcher,
            "args_preview": aura_meta.args_preview,
            "timeout_sec": aura_meta.timeout_sec,
            "max_result_size_chars": aura_meta.max_result_size_chars,
            "is_search_command": "search_command" in aura_meta.capability_flags,
        }
    return {}
