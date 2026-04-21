"""bash tool — run a shell command with a timeout."""

from __future__ import annotations

import subprocess
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.core.permissions.matchers import exact_match_on
from aura.schemas.tool import ToolError, tool_metadata

_DEFAULT_TIMEOUT = 30
_MAX_OUTPUT_BYTES = 30_000


class BashParams(BaseModel):
    command: str = Field(description="Shell command to run via /bin/sh -c.")
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT,
        ge=1,
        le=600,
        description="Timeout in seconds (1-600).",
    )


def _preview(args: dict[str, Any]) -> str:
    return f"command: {args.get('command', '')}"


def _cap_stream(data: str) -> tuple[str, bool]:
    """Tail-preserving byte cap; returns (possibly-truncated text, was_truncated)."""
    encoded = data.encode("utf-8")
    total = len(encoded)
    if total <= _MAX_OUTPUT_BYTES:
        return data, False
    kept = encoded[-_MAX_OUTPUT_BYTES:]
    dropped = total - len(kept)
    tail = kept.decode("utf-8", errors="replace")
    marker = f"… ({dropped} bytes truncated; showing last {_MAX_OUTPUT_BYTES} of {total})\n"
    return marker + tail, True


class Bash(BaseTool):
    name: str = "bash"
    description: str = (
        "Run a shell command with a timeout. Returns stdout, stderr, exit_code, "
        "and truncated (True when stdout or stderr exceeded the per-stream byte cap; "
        "truncation keeps the tail and prepends a marker noting the dropped byte count)."
    )
    args_schema: type[BaseModel] = BashParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_destructive=True,
        rule_matcher=exact_match_on("command"),
        args_preview=_preview,
    )

    def _run(self, command: str, timeout: int = _DEFAULT_TIMEOUT) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"timeout after {timeout}s: {exc}") from exc
        stdout, stdout_truncated = _cap_stream(completed.stdout)
        stderr, stderr_truncated = _cap_stream(completed.stderr)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": completed.returncode,
            "truncated": stdout_truncated or stderr_truncated,
        }


bash: BaseTool = Bash()
