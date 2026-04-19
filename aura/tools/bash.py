"""bash tool — run a shell command with a timeout."""

from __future__ import annotations

import subprocess
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.tools.base import ToolError, tool_metadata

_DEFAULT_TIMEOUT = 30


class BashParams(BaseModel):
    command: str = Field(description="Shell command to run via /bin/sh -c.")
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT,
        ge=1,
        le=600,
        description="Timeout in seconds (1-600).",
    )


class Bash(BaseTool):
    name: str = "bash"
    description: str = "Run a shell command with a timeout. Returns stdout, stderr, and exit_code."
    args_schema: type[BaseModel] = BashParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_destructive=True,
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
        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }


bash: BaseTool = Bash()
