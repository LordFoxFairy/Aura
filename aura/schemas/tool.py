"""Tool execution protocol — ``ToolResult`` / ``ToolError`` / ``tool_metadata``.

Every Aura tool returns a ``ToolResult`` to the loop (either directly from a
successful call or wrapped by the loop after catching an exception), and may
raise ``ToolError`` from its body to report a user-facing failure.

``tool_metadata(...)`` returns the dict every Aura tool carries on
``BaseTool.metadata``. Flags live in ``metadata`` (not as dedicated fields)
because LangChain's ``bind_tools`` only sends ``name``/``description``/
``args_schema`` to the model; anything extra we stash in ``metadata`` stays
local to the loop and hook layer — read via ``(tool.metadata or {}).get(...)``.

Keys:

- ``is_read_only``        — ``bool`` OR ``(args) -> bool`` — purely informational
                             as of 2026-04-21 (Plan B refactor). Used by the CLI
                             prompt tag ("read-only"). Permission decisions flow
                             through rules (see
                             ``aura.core.permissions.defaults.DEFAULT_ALLOW_RULES``)
                             and the safety policy, NOT this flag. When a callable
                             is supplied, resolve via :func:`resolve_is_read_only`
                             so callers can't accidentally read the raw bool slot
                             and miss the input-aware classification.
- ``is_destructive``      — ``bool`` OR ``(args) -> bool`` — permission gate +
                             safety rail must fire. A callable lets a tool
                             distinguish a safe ``bash("ls")`` from a destructive
                             ``bash("rm -rf /")`` at call time. Resolve via
                             :func:`resolve_is_destructive` — direct ``.get()``
                             lookups miss the callable branch and see the raw
                             function object as truthy, which would misclassify
                             every invocation as destructive.
- ``is_concurrency_safe`` — loop may batch with siblings under ``asyncio.gather``
- ``max_result_size_chars`` — post-tool budget hook truncation threshold
- ``rule_matcher``        — ``(args, content) -> bool`` — how this tool decides
                             whether a pattern rule like ``"bash(npm test)"``
                             covers a given invocation. ``None`` means the tool
                             supports only tool-wide rules.
- ``args_preview``        — ``(args) -> str`` — one-line preview of this call's
                             arguments, rendered by the CLI permission prompt.
                             ``None`` falls back to the tool name.
- ``timeout_sec``         — ``float | None`` — hard deadline (seconds) the loop
                             wraps around ``tool.ainvoke``. ``None`` = no
                             deadline (tools with their own internal timeout
                             ladder, e.g. bash, set this to ``None`` so the
                             outer wrapper doesn't stack).
- ``is_search_command``   — ``bool`` — renderer folds long output for
                             search/read tools so the REPL isn't flooded by a
                             1000-line grep. False = render as usual.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ToolRuleMatcher = Callable[[dict[str, Any], str], bool]
ToolArgsPreview = Callable[[dict[str, Any]], str]
# Input-aware classifier for ``is_destructive`` / ``is_read_only``. Matches
# claude-code's ``isDestructive(input)`` pattern: same tool object, different
# invocations, different classification — ``bash("ls")`` is a read, ``bash("rm
# -rf /")`` is destructive. A static ``bool`` remains valid for tools whose
# classification doesn't depend on args (e.g. ``read_file`` is always a read).
ToolFlagResolver = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    display: str | None = None


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
    """Build the metadata dict every Aura tool carries on ``metadata``.

    ``is_destructive`` / ``is_read_only`` accept either a static ``bool``
    (backward-compatible — the existing ~20 tool definitions all pass bools)
    or a callable ``(args) -> bool``. A callable is evaluated per-invocation
    by :func:`resolve_is_destructive` / :func:`resolve_is_read_only`, letting
    a tool like ``bash`` report a safe ``ls`` as non-destructive while still
    flagging ``rm -rf`` as destructive.
    """
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
    """Resolve a (possibly input-aware) ``is_destructive`` flag to a bool.

    - ``metadata is None`` → False (no metadata ≙ no claim).
    - Static bool → cast and return (``bool(False)`` and ``bool(True)``
      cover the legacy shape).
    - Callable → invoke with ``args``. If it raises, fail-safe to
      ``True`` — an unclassifiable command should be gated like a
      destructive one, not waved through.

    Callers MUST use this helper rather than ``metadata.get("is_destructive")``
    directly — a callable is truthy under ``bool()`` regardless of what it
    would return for the current args, so a naive ``.get()`` would route every
    bash invocation through the destructive path (including ``bash("ls")``).
    """
    return _resolve_flag(metadata, "is_destructive", args, fail_safe=True)


def resolve_is_read_only(
    metadata: dict[str, Any] | None,
    args: dict[str, Any],
) -> bool:
    """Resolve a (possibly input-aware) ``is_read_only`` flag to a bool.

    Same shape as :func:`resolve_is_destructive`; differs only in the
    fail-closed direction — if the classifier raises, return ``False``
    (a tool that can't prove it's read-only is not read-only).
    """
    return _resolve_flag(metadata, "is_read_only", args, fail_safe=False)


def _resolve_flag(
    metadata: dict[str, Any] | None,
    key: str,
    args: dict[str, Any],
    *,
    fail_safe: bool,
) -> bool:
    """Shared resolver for callable-or-bool metadata flags.

    ``fail_safe`` is the value returned when the classifier raises — pick
    the direction that treats ambiguity as the safer option for *this*
    flag (``True`` for is_destructive, ``False`` for is_read_only).
    """
    if metadata is None:
        return False
    raw = metadata.get(key, False)
    if callable(raw):
        try:
            return bool(raw(args))
        except Exception:  # noqa: BLE001 — classifier bugs must not crash the gate
            return fail_safe
    return bool(raw)
