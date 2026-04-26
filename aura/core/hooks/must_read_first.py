"""Must-read-first invariant for ``edit_file``, ``write_file``, and ``bash``.

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
  - ``bash`` (F-04-011, audit closure): gated when the command contains a
    file-mutation idiom against an existing file. The detector covers
    ``sed -i`` / ``sed --in-place``, output redirects (``> path`` /
    ``>> path``), and ``tee`` / ``tee -a``. Pure creation (target does
    not exist) passes through. The detector is regex-based and
    deliberately conservative: pipelines that obfuscate the target via
    ``$(cmd)`` substitution or `eval` slip past — those are residual
    gaps the upstream classifier (F-04-003) would close. False positives
    are not a risk here: any path the gate flags as stale is a real
    case where the model is about to overwrite a file it never read.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool

from aura.core.hooks import PRE_TOOL_PASSTHROUGH, PreToolHook, PreToolOutcome
from aura.core.memory.context import Context
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult

_ReadStatus = Literal["never_read", "stale", "partial"]


# F-04-011 — split a bash command into pipeline / sequence segments. We
# treat each segment independently so ``cat a.txt | sed -i ... b.txt``
# inspects the sed segment alone (no false positive from the cat read).
# Naïve split on the canonical separators; subshell ``$()`` bodies are
# residual gaps shared with the F-04-003 classifier.
_SEGMENT_SPLIT = re.compile(r"[|;&]+")


def _has_inplace_flag(token: str) -> bool:
    """``-i`` / ``-Ei`` / ``--in-place`` / ``--in-place=.bak`` etc."""
    if token == "--in-place" or token.startswith("--in-place="):
        return True
    if token.startswith("-") and not token.startswith("--"):
        # Combined short flags like ``-iE``: ``i`` anywhere counts.
        return "i" in token[1:]
    return False


def _last_non_option_token(tokens: list[str]) -> str | None:
    for tok in reversed(tokens):
        if not tok.startswith("-"):
            return tok
    return None


def _extract_bash_mutation_targets(command: str) -> list[str]:
    """Return paths the bash command appears to mutate.

    Empty list = "pure-read command, pass through". Caller treats each
    returned path as "must have been freshly read or hook blocks".
    Detector covers the canonical idioms documented in the module
    docstring; pipelines that obfuscate the target via ``$(cmd)`` or
    ``eval`` slip past — F-04-003's classifier would close that gap.
    """
    targets: list[str] = []
    for segment in _SEGMENT_SPLIT.split(command):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            # Unbalanced quoting — skip this segment, don't false-block.
            continue
        if not tokens:
            continue

        # 1. sed -i  /  sed --in-place
        # ``sed`` may be a path like ``/usr/bin/sed`` — match basename.
        for i, tok in enumerate(tokens):
            if tok == "sed" or tok.endswith("/sed"):
                rest = tokens[i + 1:]
                if any(_has_inplace_flag(t) for t in rest):
                    last = _last_non_option_token(rest)
                    if last is not None:
                        targets.append(last)
                break

        # 2. tee <file>  /  tee -a <file>
        for i, tok in enumerate(tokens):
            if tok == "tee" or tok.endswith("/tee"):
                rest = tokens[i + 1:]
                last = _last_non_option_token(rest)
                if last is not None:
                    targets.append(last)
                break

        # 3. > path  /  >> path  — output redirect.
        # ``shlex`` keeps ``>`` and ``>>`` as separate tokens. Skip
        # /dev/* (always a sink) and process substitutions.
        for i, tok in enumerate(tokens):
            if tok in (">", ">>"):
                if i + 1 >= len(tokens):
                    continue
                target = tokens[i + 1]
                if target.startswith("/dev/") or target.startswith("("):
                    continue
                # ``2>`` and friends arrive as ``2>`` from shlex too;
                # explicit fd-prefixed forms like ``2>&1`` remain a
                # single token so harmless.
                targets.append(target)
            elif re.fullmatch(r"\d*>{1,2}", tok) and not tok.endswith("&"):
                # Combined fd-redirect like ``2>`` or ``1>>`` token.
                if i + 1 >= len(tokens):
                    continue
                target = tokens[i + 1]
                if target.startswith("/dev/") or target.startswith("("):
                    continue
                # ``2>&1`` is a single token; ``2>file`` may also collapse.
                targets.append(target)

    return targets


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
    ) -> PreToolOutcome:
        if tool.name not in ("edit_file", "write_file", "bash"):
            return PRE_TOOL_PASSTHROUGH

        # Lazy import — same pattern as aura/core/memory/rules.py to keep
        # the journal dependency out of the module-load path.
        from aura.core.persistence import journal

        if tool.name == "bash":
            command = args.get("command")
            if not isinstance(command, str) or not command:
                return PRE_TOOL_PASSTHROUGH
            for raw_target in _extract_bash_mutation_targets(command):
                try:
                    resolved = Path(raw_target).resolve()
                except OSError:
                    continue
                # Pure creation: nothing on disk to have read.
                if not resolved.exists():
                    continue
                status = context.read_status(resolved)
                if status == "fresh":
                    continue
                journal.write(
                    "must_read_first_blocked",
                    tool="bash",
                    path=str(resolved),
                    reason=status,
                    command=command,
                )
                return PreToolOutcome(
                    short_circuit=ToolResult(
                        ok=False,
                        error=(
                            f"bash command would mutate {resolved} but it has "
                            f"not been read this session ({status}). "
                            f"read_file({resolved}) before running."
                        ),
                    ),
                    decision=None,
                )
            return PRE_TOOL_PASSTHROUGH

        raw = args.get("path")
        if not isinstance(raw, str) or not raw:
            # Tool's own arg-validation (pydantic schema) will reject —
            # don't pre-empt that error with a less specific one.
            return PRE_TOOL_PASSTHROUGH

        try:
            resolved = Path(raw).resolve()
        except OSError:
            # Let the tool's own error path surface (e.g. "not found").
            return PRE_TOOL_PASSTHROUGH

        if tool.name == "edit_file":
            # Mirror claude-code: allow new-file creation via empty old_str
            # without a prior read — there is nothing on disk to have read.
            # Narrowly scoped: requires old_str == "" AND path does not exist.
            if args.get("old_str") == "" and not resolved.exists():
                return PRE_TOOL_PASSTHROUGH
        else:  # write_file
            # File-unchanged guard only applies when there's a file to be
            # unchanged. Pure creation always passes through.
            if not resolved.exists():
                return PRE_TOOL_PASSTHROUGH

        status = context.read_status(resolved)
        if status == "fresh":
            return PRE_TOOL_PASSTHROUGH

        journal.write(
            "must_read_first_blocked",
            tool=tool.name,
            path=str(resolved),
            reason=status,
        )
        return PreToolOutcome(
            short_circuit=ToolResult(
                ok=False, error=_error_text(tool.name, status, resolved),
            ),
            decision=None,
        )

    return _hook
