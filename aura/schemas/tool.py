"""Tool execution protocol — ToolResult / ToolError / ToolMetadata."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

ToolRuleMatcher = Callable[[dict[str, Any], str], bool]
ToolArgsPreview = Callable[[dict[str, Any]], str]
ToolFlagResolver = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    display: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    invalid: bool
    reason: str = ""


@dataclass(frozen=True)
class ToolMetadata:
    is_read_only: bool
    is_destructive: bool | ToolFlagResolver
    is_concurrency_safe: bool
    rule_matcher: ToolRuleMatcher | None
    args_preview: ToolArgsPreview | None
    timeout_sec: float | None
    max_result_size_chars: int | None = None
    capability_flags: frozenset[str] = field(default_factory=frozenset)


class ToolError(Exception):
    """Raise from a tool ``_run`` / ``_arun`` to surface a user-facing error."""


def tool_metadata(
    *,
    is_read_only: bool | ToolFlagResolver = False,
    is_destructive: bool | ToolFlagResolver = False,
    is_concurrency_safe: bool = False,
    max_result_size_chars: int | None = None,
    rule_matcher: ToolRuleMatcher | None = None,
    args_preview: ToolArgsPreview | None = None,
    timeout_sec: float | None = None,
    is_search_command: bool = False,
) -> dict[str, Any]:
    return {
        "is_read_only": is_read_only,
        "is_destructive": is_destructive,
        "is_concurrency_safe": is_concurrency_safe,
        "max_result_size_chars": max_result_size_chars,
        "rule_matcher": rule_matcher,
        "args_preview": args_preview,
        "timeout_sec": timeout_sec,
        "is_search_command": is_search_command,
    }


def resolve_is_destructive(
    metadata: dict[str, Any] | None,
    args: dict[str, Any],
) -> bool:
    # Classifier exceptions fail-safe to True (ambiguous ≙ destructive).
    return _resolve_flag(metadata, "is_destructive", args, fail_safe=True)


def resolve_is_read_only(
    metadata: dict[str, Any] | None,
    args: dict[str, Any],
) -> bool:
    return _resolve_flag(metadata, "is_read_only", args, fail_safe=False)


def _resolve_flag(
    metadata: dict[str, Any] | None,
    key: str,
    args: dict[str, Any],
    *,
    fail_safe: bool,
) -> bool:
    if metadata is None:
        return False
    raw = metadata.get(key, False)
    if callable(raw):
        try:
            return bool(raw(args))
        except Exception:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
            return fail_safe
    return bool(raw)
