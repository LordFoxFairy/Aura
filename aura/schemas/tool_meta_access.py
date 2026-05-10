"""Unified metadata-access helper — Phase 2 Task 3 bridge.

Every builtin tool now sets ``aura_metadata: ToolMetadata``. This module
provides ``meta_dict(tool)`` which projects the typed ``ToolMetadata`` fields
back to the legacy dict shape so all loop / hook / CLI readers continue to
work unchanged through Phase 2.

The primary read path is ``tool.aura_metadata`` (set on all 21 migrated
builtins). The fallback to ``tool.metadata`` (the legacy dict) is removed
for the 7 typed fields; however, the "legacy-only" key ``max_result_size_chars``
is still promoted from ``tool.metadata`` for tools that set both
(e.g. ``build_tool`` test helpers), because it has no ``ToolMetadata`` field
to carry it. ``is_search_command`` is projected from ``capability_flags``.
Task 4 will enforce ``ToolMetadata`` at registration, at which point
``max_result_size_chars`` can be added as a proper field or the budget hook
can be taught to read it from a different source.

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
    all migrated builtins) and projects its fields back to the legacy dict
    keys.

    The ``tool.metadata`` fallback is removed for the 7 typed fields.
    ``max_result_size_chars`` is the sole exception: it is not in
    ``ToolMetadata`` (spec §3 limits the typed surface to 7 fields), so
    we promote it from ``tool.metadata`` when present. This lets
    ``build_tool(max_result_size_chars=...)`` test helpers continue to
    exercise the budget hook's per-tool override path; Task 4 will add a
    proper typed field or remove the need for this bridge entirely.

    Returns ``{}`` when neither source is present so
    ``meta_dict(tool).get(...)`` is always safe.
    """
    aura_meta = getattr(tool, "aura_metadata", None)
    if isinstance(aura_meta, ToolMetadata):
        # ``max_result_size_chars`` is a legacy-only key not in ToolMetadata;
        # fall back to tool.metadata only for this one key so build_tool
        # test helpers can still exercise the budget hook's per-tool cap.
        legacy = getattr(tool, "metadata", None)
        legacy_max = legacy.get("max_result_size_chars") if isinstance(legacy, dict) else None
        return {
            "is_read_only": aura_meta.is_read_only,
            "is_destructive": aura_meta.is_destructive,
            "is_concurrency_safe": aura_meta.is_concurrency_safe,
            "rule_matcher": aura_meta.rule_matcher,
            "args_preview": aura_meta.args_preview,
            "timeout_sec": aura_meta.timeout_sec,
            "max_result_size_chars": legacy_max,
            "is_search_command": "search_command" in aura_meta.capability_flags,
        }
    # No aura_metadata — legacy dict path for test-only tools and MCP
    # adapter tools not yet migrated in Task 4.
    legacy = getattr(tool, "metadata", None)
    return legacy if isinstance(legacy, dict) else {}
