"""Permissions config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StatusLineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = ""
    timeout_ms: int = 500
    enabled: bool = True

    @field_validator("timeout_ms")
    @classmethod
    def _clamp_timeout(cls, v: int) -> int:
        if v < 50:
            return 50
        if v > 5000:
            return 5000
        return v

    @property
    def is_active(self) -> bool:
        return self.enabled and bool(self.command.strip())


class PermissionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["default", "bypass", "plan", "accept_edits"] = "default"
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    safety_exempt: list[str] = Field(default_factory=list)
    statusline: StatusLineConfig | None = None
    prompt_timeout_sec: float | None = Field(
        default=300.0,
        description=(
            "Seconds to wait for prompt response before treating as denial. "
            "None = wait forever; default 300 (5 minutes)."
        ),
    )
    disable_bypass: bool = Field(
        default=False,
        description="When true, refuse all attempts to enter bypass mode.",
    )
