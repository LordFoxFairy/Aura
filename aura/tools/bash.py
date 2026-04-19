"""bash tool — run a shell command with a timeout."""

from __future__ import annotations

import subprocess

from pydantic import BaseModel, Field

from aura.tools.base import AuraTool, ToolResult, build_tool

_DEFAULT_TIMEOUT = 30


class BashParams(BaseModel):
    command: str = Field(description="Shell command to run via /bin/sh -c.")
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT,
        ge=1,
        le=600,
        description="Timeout in seconds (1-600).",
    )


def _run(params: BashParams) -> ToolResult:
    try:
        completed = subprocess.run(
            params.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=params.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ToolResult(ok=False, error=f"timeout after {params.timeout}s: {exc}")
    return ToolResult(
        ok=True,
        output={
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        },
    )


bash: AuraTool = build_tool(
    name="bash",
    description="Run a shell command with a timeout. Returns stdout, stderr, and exit_code.",
    input_model=BashParams,
    call=_run,
    is_destructive=True,
)
