"""Project ToolMetadata to a flat dict view."""

from __future__ import annotations

from typing import Any

from aura.domain.tool import ToolMetadata


def meta_dict(tool: Any) -> dict[str, Any]:
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
