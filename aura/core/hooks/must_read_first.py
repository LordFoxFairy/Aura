"""Must-read-first invariant for ``edit_file``.

Mirrors claude-code's ``FileEditTool.ts:275–287`` (errorCode 6): before an
edit, the file MUST have been read in the same session AND must not have
changed on disk since. Without this guard the model can edit based on stale
assumptions and silently corrupt user files.

Enforcement is a ``PreToolHook`` closure over a session-scoped ``Context``
reference. ``Context`` carries the ``_read_records`` map of
(mtime, size) fingerprints; ``AgentLoop`` calls ``Context.record_read``
after each successful ``read_file`` invocation. Staleness is compared via
mtime+size — lighter than claude-code's content-hash approach but
equivalent in practice for real edits (a mutation changes at least one).

Scope (matches claude-code):
  - fires ONLY for ``edit_file``
  - ``write_file`` is deliberately NOT gated (creation / overwrite)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from aura.core.hooks import PreToolHook
from aura.core.memory.context import Context
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult


def make_must_read_first_hook(context: Context) -> PreToolHook:
    async def _hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **_: Any,
    ) -> ToolResult | None:
        if tool.name != "edit_file":
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

        # Mirror claude-code: allow new-file creation via empty old_str
        # without a prior read — there is nothing on disk to have read.
        # Narrowly scoped: requires old_str == "" AND path does not exist.
        if args.get("old_str") == "" and not resolved.exists():
            return None

        status = context.read_status(resolved)
        if status == "fresh":
            return None

        # Lazy import — same pattern as aura/core/memory/rules.py to keep
        # the journal dependency out of the module-load path.
        from aura.core.persistence import journal

        journal.write(
            "must_read_first_blocked",
            tool="edit_file",
            path=str(resolved),
            reason=status,
        )
        if status == "stale":
            error = (
                f"file has changed since last read. re-read before editing. "
                f"(path={resolved})"
            )
        else:  # "never_read"
            error = (
                f"file has not been read yet. read_file({resolved}) before "
                f"edit. (path={resolved})"
            )
        return ToolResult(ok=False, error=error)

    return _hook
