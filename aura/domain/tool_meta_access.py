"""Read a tool's attached ToolMetadata: typed object or flat dict view."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from aura.domain.tool import HasAuraMetadata, ToolMetadata


def aura_metadata(tool: BaseTool) -> ToolMetadata | None:
    if isinstance(tool, HasAuraMetadata) and isinstance(
        tool.aura_metadata, ToolMetadata
    ):
        return tool.aura_metadata
    return None


def meta_dict(tool: BaseTool) -> dict[str, Any]:
    meta = aura_metadata(tool)
    if meta is None:
        return {}
    return {
        "is_read_only": meta.is_read_only,
        "is_destructive": meta.is_destructive,
        "is_concurrency_safe": meta.is_concurrency_safe,
        "rule_matcher": meta.rule_matcher,
        "args_preview": meta.args_preview,
        "timeout_sec": meta.timeout_sec,
        "max_result_size_chars": meta.max_result_size_chars,
        "is_search_command": "search_command" in meta.capability_flags,
    }
