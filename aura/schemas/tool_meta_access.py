"""Transitional metadata-access helper — Phase 2 Task 2 bridge.

During the Phase 2 tools migration, individual builtin tools move from
the legacy ``metadata: dict`` (built by :func:`aura.schemas.tool.tool_metadata`)
to a typed ``aura_metadata: ToolMetadata`` attribute. The migration ships
one tool at a time (``read_file`` in Task 2, the remaining 20 in Task 3),
so for the duration of the migration the loop, hook, and CLI readers
must transparently accept *both* shapes.

This module is the single bridge. Every consumer that used to read
``(tool.metadata or {}).get("is_concurrency_safe", False)`` now reads
``meta_dict(tool).get("is_concurrency_safe", False)`` — same shape, same
default, same call site, but the dict is built from ``aura_metadata`` if
that's the canonical source on this tool.

Lives under ``aura.schemas`` rather than ``aura.tools`` so consumers
inside ``aura.core.permissions`` (which feeds ``aura.tools`` indirectly
through agent wiring) can import it without triggering an import cycle
through ``aura.tools.__init__``.

Task 4 deletes this module along with the legacy ``tool_metadata(...)``
dict helper. Until then, keep it tiny and obvious — one function, one
purpose, no caching, no configuration.
"""

from __future__ import annotations

from typing import Any

from aura.schemas.tool import ToolMetadata


def meta_dict(tool: Any) -> dict[str, Any]:
    """Return tool metadata as a legacy-shaped dict.

    Prefers ``tool.aura_metadata`` (the typed ``ToolMetadata`` set by
    migrated tools) and projects its fields back to the legacy dict
    keys. Legacy-only keys not represented in ``ToolMetadata``
    (``max_result_size_chars``, ``is_search_command``) are filled with
    safe defaults so callers reading them via ``.get(key, default)``
    behave identically across both shapes.

    Falls back to ``tool.metadata`` (the legacy dict) when no
    ``aura_metadata`` is set — that's the path the 20 unmigrated
    builtins still take. Returns ``{}`` if neither is present so
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
            # Legacy-only keys — ToolMetadata doesn't carry them as
            # named fields (spec §3 keeps the typed surface to 7 fields).
            # ``is_search_command`` is projected from the
            # ``"search_command"`` capability flag so the renderer's
            # search-fold drift test continues to read the correct value
            # off a migrated tool's metadata; the only legacy key
            # without a typed counterpart is ``max_result_size_chars``
            # which defaults to None.
            "max_result_size_chars": None,
            "is_search_command": "search_command" in aura_meta.capability_flags,
        }
    legacy = getattr(tool, "metadata", None)
    return legacy if isinstance(legacy, dict) else {}
