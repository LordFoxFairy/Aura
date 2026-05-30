"""Permissions schema + Outcome tagged union."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aura.domain.tool import ToolResult

if TYPE_CHECKING:
    # Runtime import would cycle aura.schemas → aura.application.
    from aura.application.permission.decision import Decision


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


@dataclass(frozen=True)
class Allow:
    decision: Decision

    def __post_init__(self) -> None:
        if not self.decision.allow:
            raise ValueError(
                "Allow requires a Decision with allow=True; "
                f"got reason={self.decision.reason!r}, allow=False"
            )


@dataclass(frozen=True)
class Block:
    decision: Decision

    def __post_init__(self) -> None:
        if self.decision.allow:
            raise ValueError(
                "Block requires a Decision with allow=False; "
                f"got reason={self.decision.reason!r}, allow=True"
            )


@dataclass(frozen=True)
class Ask:
    reason: str

    def __post_init__(self) -> None:
        if not self.reason or not self.reason.strip():
            raise ValueError("Ask requires a non-empty reason")


@dataclass(frozen=True)
class Replace:
    result: ToolResult
    decision: Decision

    def __post_init__(self) -> None:
        # Defensive None check: callers can bypass the type system (tests do).
        if self.result is None:  # pyright: ignore[reportUnnecessaryComparison]  # ToolResult is non-Optional in types, but tests construct via raw kwargs that bypass it.
            raise ValueError("Replace requires a non-None ToolResult")
        if self.decision.allow:
            raise ValueError(
                "Replace requires a Decision with allow=False "
                "(tool was not invoked); "
                f"got reason={self.decision.reason!r}, allow=True"
            )


Outcome = Allow | Block | Ask | Replace


_ASKER_CHOICES: frozenset[str] = frozenset(
    {"yes", "yes-always", "no", "no-always"}
)


@dataclass(frozen=True)
class AskerPrompt:
    tool: str
    args_preview: str
    rule_hint: str
    is_destructive: bool
    request_id: str

    def __post_init__(self) -> None:
        if not self.tool:
            raise ValueError("AskerPrompt requires a non-empty tool name")
        if not self.request_id:
            raise ValueError(
                "AskerPrompt requires a non-empty request_id "
                "(used for IPC correlation)"
            )


@dataclass(frozen=True)
class AskerResponse:
    choice: Literal["yes", "yes-always", "no", "no-always"]
    request_id: str

    def __post_init__(self) -> None:
        if self.choice not in _ASKER_CHOICES:
            raise ValueError(
                f"AskerResponse.choice must be one of {sorted(_ASKER_CHOICES)!r}; "
                f"got {self.choice!r}"
            )
        if not self.request_id:
            raise ValueError(
                "AskerResponse requires a non-empty request_id "
                "(must echo AskerPrompt.request_id for IPC correlation)"
            )
