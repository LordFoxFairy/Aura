"""Must-read-first gate: mutation requires prior session read + matching (mtime, size).

Scope: edit_file always; write_file only if target exists; bash on detected mutation
idioms (sed -i, redirects, tee). Subshell/eval obfuscation slips past (shared gap).
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool

from aura.application.hooks import PreToolHook
from aura.application.memory.context import Context
from aura.domain.permission.decision import Decision
from aura.domain.permission.outcome import Allow, Replace
from aura.domain.tool import ToolResult
from aura.schemas.state import LoopState

_ReadStatus = Literal["never_read", "stale", "partial"]

_SEGMENT_SPLIT = re.compile(r"[|;&]+")


def _has_inplace_flag(token: str) -> bool:
    if token == "--in-place" or token.startswith("--in-place="):
        return True
    if token.startswith("-") and not token.startswith("--"):
        return "i" in token[1:]
    return False


def _last_non_option_token(tokens: list[str]) -> str | None:
    for tok in reversed(tokens):
        if not tok.startswith("-"):
            return tok
    return None


def _extract_bash_mutation_targets(command: str) -> list[str]:
    """Paths the bash command appears to mutate; empty list = no detected mutation."""
    targets: list[str] = []
    for segment in _SEGMENT_SPLIT.split(command):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            continue
        if not tokens:
            continue

        for i, tok in enumerate(tokens):
            if tok == "sed" or tok.endswith("/sed"):
                rest = tokens[i + 1:]
                if any(_has_inplace_flag(t) for t in rest):
                    last = _last_non_option_token(rest)
                    if last is not None:
                        targets.append(last)
                break

        for i, tok in enumerate(tokens):
            if tok == "tee" or tok.endswith("/tee"):
                rest = tokens[i + 1:]
                last = _last_non_option_token(rest)
                if last is not None:
                    targets.append(last)
                break

        for i, tok in enumerate(tokens):
            if tok in (">", ">>"):
                if i + 1 >= len(tokens):
                    continue
                target = tokens[i + 1]
                if target.startswith("/dev/") or target.startswith("("):
                    continue
                targets.append(target)
            elif (match := re.fullmatch(r"(\d*)?(>{1,2})(.*)", tok)):
                suffix = match.group(3)
                if suffix.startswith("&"):
                    continue
                if suffix:
                    target = suffix
                else:
                    if i + 1 >= len(tokens):
                        continue
                    target = tokens[i + 1]
                if target.startswith("/dev/") or target.startswith("("):
                    continue
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
        state: LoopState,  # noqa: ARG001  # Protocol kw arg; unused
        **_: Any,
    ) -> Allow | Replace:
        if tool.name not in ("edit_file", "write_file", "bash"):
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))

        from aura.infrastructure.persistence import journal

        if tool.name == "bash":
            command = args.get("command")
            if not isinstance(command, str) or not command:
                return Allow(decision=Decision(allow=True, reason="mode_bypass"))
            for raw_target in _extract_bash_mutation_targets(command):
                try:
                    resolved = Path(raw_target).resolve()
                except OSError:
                    continue
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
                return Replace(
                    result=ToolResult(
                        ok=False,
                        error=(
                            f"bash command would mutate {resolved} but it has "
                            f"not been read this session ({status}). "
                            f"read_file({resolved}) before running."
                        ),
                    ),
                    decision=Decision(allow=False, reason="safety_blocked"),
                )
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))

        raw = args.get("path")
        if not isinstance(raw, str) or not raw:
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))

        try:
            resolved = Path(raw).resolve()
        except OSError:
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))

        if tool.name == "edit_file":
            # New-file creation bypass: empty old_str + missing path.
            if args.get("old_str") == "" and not resolved.exists():
                return Allow(decision=Decision(allow=True, reason="mode_bypass"))
        elif not resolved.exists():
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))

        status = context.read_status(resolved)
        if status == "fresh":
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))

        journal.write(
            "must_read_first_blocked",
            tool=tool.name,
            path=str(resolved),
            reason=status,
        )
        return Replace(
            result=ToolResult(
                ok=False, error=_error_text(tool.name, status, resolved),
            ),
            decision=Decision(allow=False, reason="safety_blocked"),
        )

    return _hook
