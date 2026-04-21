"""Must-read-first invariant for ``edit_file`` and ``write_file``.

Mirrors claude-code's ``FileEditTool.ts:275–287`` (errorCode 6) and
``FileWriteTool.ts:280–294`` (file-unchanged guard): before modifying a file
on disk, it MUST have been read in the current session AND must not have
drifted since. Without this guard the model can mutate based on a stale
view and silently corrupt user files.

Enforcement is a ``PreToolHook`` closure over a session-scoped ``Context``
reference. ``Context`` carries the ``_read_records`` map of
(mtime, size) fingerprints; ``AgentLoop`` calls ``Context.record_read``
after each successful ``read_file`` invocation. Staleness is compared via
mtime+size — lighter than claude-code's content-hash approach but
equivalent in practice for real mutations (they change at least one).

Scope (matches claude-code):
  - ``edit_file`` is always gated; ``old_str == ""`` + non-existent path
    is an explicit bypass for new-file creation via edit.
  - ``write_file`` is gated ONLY when the target already exists. Pure
    creation (path does not exist on disk) passes through — there is no
    prior content that could drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool

from aura.core.hooks import PreToolHook
from aura.core.memory.context import Context
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult

_ReadStatus = Literal["never_read", "stale", "partial"]


def _error_text(tool_name: str, reason: _ReadStatus, path: Path) -> str:
    if tool_name == "write_file":
        if reason == "stale":
            return (
                f"file has changed since last read. re-read before overwriting. "
                f"(path={path})"
            )
        if reason == "partial":
            return (
                f"file was only partially read. read_file({path}) fully "
                f"(offset=0, limit=None) before overwriting."
            )
        return (
            f"file has not been read yet. read_file({path}) before overwriting."
        )
    # edit_file
    if reason == "stale":
        return (
            f"file has changed since last read. re-read before editing. "
            f"(path={path})"
        )
    if reason == "partial":
        return (
            f"file was only partially read. read_file({path}) fully "
            f"(offset=0, limit=None) before edit."
        )
    return (
        f"file has not been read yet. read_file({path}) before edit."
    )


def make_must_read_first_hook(context: Context) -> PreToolHook:
    async def _hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **_: Any,
    ) -> ToolResult | None:
        if tool.name not in ("edit_file", "write_file"):
            return None

        raw = args.get("path")
        if not isinstance(raw, str) or not raw:
            # Tool's own arg-validation (pydantic schema) will reject —
            # don't pre-empt that error with a less specific one.
            return None

        try:
            resolved = Path(raw).resolve()
        except OSError:
            # Let the tool's own error path surface (e.g. "not found").
            return None

        if tool.name == "edit_file":
            # Mirror claude-code: allow new-file creation via empty old_str
            # without a prior read — there is nothing on disk to have read.
            # Narrowly scoped: requires old_str == "" AND path does not exist.
            if args.get("old_str") == "" and not resolved.exists():
                return None
        else:  # write_file
            # File-unchanged guard only applies when there's a file to be
            # unchanged. Pure creation always passes through.
            if not resolved.exists():
                return None

        status = context.read_status(resolved)
        if status == "fresh":
            return None

        # Lazy import — same pattern as aura/core/memory/rules.py to keep
        # the journal dependency out of the module-load path.
        from aura.core.persistence import journal

        journal.write(
            "must_read_first_blocked",
            tool=tool.name,
            path=str(resolved),
            reason=status,
        )
        return ToolResult(ok=False, error=_error_text(tool.name, status, resolved))

    return _hook
