"""Pure data types shared across the bash-safety modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Reason = Literal[
    "zsh_dangerous_command",
    "cr_outside_double_quote",
    "malformed_with_separator",
    "cd_git_compound",
    "command_substitution",
    "pipe_to_shell",
    "sed_inplace_system_path",
    "redirect_to_system_path",
    "destructive_removal",
    "world_writable_chmod",
    "root_chown",
    "exec_destructive",
    "obfuscated_execution",
]


@dataclass(frozen=True)
class BashSafetyViolation:
    # ``detail`` is cited verbatim in ToolResult.error surfaced to the model.
    reason: Reason
    detail: str
