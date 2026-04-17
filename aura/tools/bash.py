"""bash built-in tool — run a shell command with a timeout."""
from __future__ import annotations

import asyncio
import subprocess

from pydantic import BaseModel, Field

from aura.tools.base import AuraTool, ToolResult  # noqa: F401

_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 600


class BashParams(BaseModel):
    command: str = Field(description="Shell command to run via /bin/sh -c.")
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT,
        ge=1,
        le=_MAX_TIMEOUT,
        description="Timeout in seconds (1-600).",
    )


class BashTool:
    name: str = "bash"
    description: str = (
        "Run a shell command with a timeout. Returns stdout, stderr, and exit_code."
    )
    input_model: type[BaseModel] = BashParams
    is_read_only: bool = False
    is_destructive: bool = True
    is_concurrency_safe: bool = False  # shell commands can race on shared resources

    async def acall(self, params: BaseModel) -> ToolResult:
        assert isinstance(params, BashParams)
        return await asyncio.to_thread(self._run_sync, params.command, params.timeout)

    def _run_sync(self, command: str, timeout: int) -> ToolResult:
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
            return ToolResult(
                ok=False,
                error=f"timeout after {timeout}s: {exc}",
            )
        return ToolResult(
            ok=True,
            output={
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "exit_code": completed.returncode,
            },
        )
